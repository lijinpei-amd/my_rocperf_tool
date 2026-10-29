#pragma once

#include "my_rocperf_tool/inst_kind.h"

#include "rocprofiler-sdk/callback_tracing.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Support/Error.h"
#include "llvm/TargetParser/Triple.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace llvm {
class MCAsmInfo;
class MCRegisterInfo;
class MemoryBuffer;
class Target;
}  // namespace llvm

namespace my_rocperf_tool {

struct SubTargetKey {
  const llvm::Target* target;
  llvm::Triple triple;
  std::string mcpu;
  std::string features;

  bool operator==(const SubTargetKey& other) const {
    return target == other.target && triple == other.triple &&
           mcpu == other.mcpu && features == other.features;
  }
};
}  // namespace my_rocperf_tool

namespace llvm {
template <>
struct DenseMapInfo<my_rocperf_tool::SubTargetKey> {
  using TargetInfoTy = DenseMapInfo<const Target*>;

  static my_rocperf_tool::SubTargetKey getEmptyKey() {
    return my_rocperf_tool::SubTargetKey{
        TargetInfoTy::getEmptyKey(), llvm::Triple{}, {}, {}};
  }

  static my_rocperf_tool::SubTargetKey getTombstoneKey() {
    return my_rocperf_tool::SubTargetKey{
        TargetInfoTy::getTombstoneKey(), llvm::Triple{}, {}, {}};
  }

  static unsigned getHashValue(const my_rocperf_tool::SubTargetKey& val) {
    auto hash_triple = [](const llvm::Triple& triple) {
      return llvm::hash_combine(
          triple.getArch(), triple.getSubArch(), triple.getVendor(),
          triple.getOS(), triple.getEnvironment(), triple.getObjectFormat());
    };
    return llvm::hash_combine(val.target, hash_triple(val.triple), val.mcpu,
                              val.features);
  }

  static bool isEqual(const my_rocperf_tool::SubTargetKey& LHS,
                      const my_rocperf_tool::SubTargetKey& RHS) {
    return LHS == RHS;
  }
};

}  // namespace llvm

namespace my_rocperf_tool {

class Disassembler;
class SubTargetContext;

// One loaded code object, answering questions about the instruction at an
// ELF vaddr. Addresses here are ELF vaddrs throughout: the trace decoder
// resolves a hardware PC to {code object, vaddr} before the tool sees it, so
// nothing downstream needs the object's runtime load address.
//
// The per-vaddr queries take the vaddr of an instruction's first byte in one
// of the object's .text sections, and assert otherwise. A section is decoded
// whole on its first query, after which a query costs a section bounds check
// and a slot index. Not thread-safe; see Disassembler.
class ObjectFileInfo {
 public:
  ObjectFileInfo(ObjectFileInfo&&);
  ~ObjectFileInfo();

  // Encoded size in bytes: 4, or more for an instruction that carries a
  // literal or a 64-bit encoding.
  uint32_t size_at(uint64_t vaddr) const;
  InstKind kind_at(uint64_t vaddr) const;
  // The instruction as LLVM prints it, trimmed. The trace decoder classifies
  // instructions by this text, so its exact bytes are part of the contract.
  std::string text_at(uint64_t vaddr) const;
  // The same text, copied into `buf` only if it fits. Returns its length
  // either way, so a caller can grow its buffer and ask again without a heap
  // allocation per instruction.
  size_t copy_text_at(uint64_t vaddr, llvm::MutableArrayRef<char> buf) const;

  // The vaddrs of the instructions starting in [first, last], in order,
  // walking from the one at `first`. Empty unless a single .text section
  // holds both ends.
  std::vector<uint64_t> inst_vaddrs(uint64_t first, uint64_t last) const;

 private:
  friend class Disassembler;
  // One decoded instruction, or the continuation of a multi-dword one.
  struct Slot;
  // A .text section and, once decoded, its slots by dword.
  struct Section;

  ObjectFileInfo(std::unique_ptr<llvm::MemoryBuffer> buffer,
                 std::vector<Section> sections,
                 const SubTargetContext& sub_target_ctx);
  // Fails with "<buffer name>: <reason>": not an ELF file, not AMDGPU, or an
  // ISA (e_flags machine) missing from comgr-isa-metadata.def.
  static llvm::Expected<ObjectFileInfo> load(
      Disassembler& disas, std::unique_ptr<llvm::MemoryBuffer> buffer);

  Section* find_section(uint64_t vaddr) const;
  void ensure_section_decoded(Section& sec) const;
  void decode_all_sections() const;
  // The decoded instruction at vaddr, found through hot_section.
  const Slot& slot_at(uint64_t vaddr) const;

  std::unique_ptr<llvm::MemoryBuffer> buffer;
  // Sorted by vaddr. Decoding fills them through the const queries.
  mutable std::vector<Section> sections;
  // The section the last query hit. It points into the vector's heap block,
  // so it survives a move of this object.
  mutable Section* hot_section = nullptr;
  // Owned by the Disassembler's cache, keyed on this object's ISA.
  const SubTargetContext* sub_target_ctx;
};

// Owns the code objects of one run and the MC stacks that decode them. Needs no
// setup: the first construction in the process registers the AMDGPU target
// with LLVM.
//
// Not thread-safe; give each thread its own Disassembler. Every query fills
// caches, even through a const ObjectFileInfo, and the MC stack keeps per-call
// state.
//
// Never relocated, hence neither copyable nor movable: every SubTargetContext's
// MCContext holds a pointer to mc_options, which a move would leave dangling.
// Construct it where it will stay, e.g. behind a unique_ptr or in place in a
// node-based map.
class Disassembler {
 public:
  Disassembler();
  ~Disassembler();
  Disassembler(const Disassembler&) = delete;
  Disassembler& operator=(const Disassembler&) = delete;

  // Loads a code object, from a file or a live load record, under `id`. Fails
  // with "<path or uri>: <reason>" if it cannot be read, is not an AMDGPU ELF
  // object, or has an ISA missing from comgr-isa-metadata.def; `id` then
  // stays unknown. An id already present keeps its code object, and the new
  // one is not read.
  llvm::Error add_code_object(uint64_t id, const std::string& file_path);
  llvm::Error add_code_object(
      const rocprofiler_callback_tracing_code_object_load_data_t& load_data);

  // Null if no code object was loaded under `id`. The pointer stays valid for
  // the lifetime of this Disassembler.
  const ObjectFileInfo* find_code_object(uint64_t id) const;

  // Decodes every .text section of every code object now instead of on its
  // first query.
  void decode_all() const;

 private:
  friend class ObjectFileInfo;
  friend class SubTargetContext;

  llvm::Error add_buffer(uint64_t id,
                         std::unique_ptr<llvm::MemoryBuffer> buffer);
  // Returns the cached MC stack for this ISA, building it on first request.
  // The reference stays valid for the lifetime of this Disassembler.
  const SubTargetContext& get_sub_target_context(llvm::StringRef mcpu,
                                                 llvm::StringRef features);

  // Node-based, so find_code_object's pointers survive later additions. Ids
  // come from the trace, so any value must be a safe lookup -- which rules out
  // DenseMap, whose reserved keys a corrupt trace could name.
  std::unordered_map<uint64_t, ObjectFileInfo> object_files;
  llvm::DenseMap<SubTargetKey, std::unique_ptr<SubTargetContext>>
      sub_target_contexts;
  llvm::Triple triple;
  const llvm::Target* target = nullptr;
  // Defaults, not llvm::mc::InitMCTargetOptionsFromFlags(): that reads cl::opts
  // which exist only once a RegisterMCTargetOptionsFlags is constructed, and
  // this tool exposes no LLVM options.
  llvm::MCTargetOptions mc_options;
  std::unique_ptr<llvm::MCRegisterInfo> mc_reg_info;
  std::unique_ptr<llvm::MCAsmInfo> mc_asm_info;
};

}  // namespace my_rocperf_tool
