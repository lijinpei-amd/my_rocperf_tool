#pragma once

#include "rocprofiler-sdk/callback_tracing.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>
#include <vector>

namespace my_rocperf_tool {

struct SubTargetKey {
  llvm::Target *target;
  llvm::Triple triple;
  std::string mcpu;
  std::string features;

  bool operator==(const SubTargetKey &other) const {
    return target == other.target && triple == other.triple &&
           mcpu == other.mcpu && features == other.features;
  }
};
} // namespace my_rocperf_tool

namespace llvm {
template <> struct DenseMapInfo<my_rocperf_tool::SubTargetKey> {
  using TargetInfoTy = DenseMapInfo<Target *>;

  static my_rocperf_tool::SubTargetKey getEmptyKey() {
    return my_rocperf_tool::SubTargetKey{
        TargetInfoTy::getEmptyKey(), llvm::Triple{}, {}, {}};
  }

  static my_rocperf_tool::SubTargetKey getTombstoneKey() {
    return my_rocperf_tool::SubTargetKey{
        TargetInfoTy::getTombstoneKey(), llvm::Triple{}, {}, {}};
  }

  static unsigned getHashValue(const my_rocperf_tool::SubTargetKey &val) {
    auto hash_triple = [](const llvm::Triple &triple) {
      return llvm::hash_combine(
          triple.getArch(), triple.getSubArch(), triple.getVendor(),
          triple.getOS(), triple.getEnvironment(), triple.getObjectFormat());
    };
    return llvm::hash_combine(val.target, hash_triple(val.triple), val.mcpu,
                              val.features);
  }

  static bool isEqual(const my_rocperf_tool::SubTargetKey &LHS,
                      const my_rocperf_tool::SubTargetKey &RHS) {
    return LHS == RHS;
  }
};

} // namespace llvm

namespace my_rocperf_tool {

class Disassembler;

class SymbolIndex {
  struct SymbolInfo {
    llvm::StringRef name;
    uint64_t value;
    uint64_t size;
    const void *symbol;
  };
  std::vector<SymbolInfo> symbols;
  llvm::StringMap<const SymbolInfo *> symbol_map;

public:
  void clear() {
    symbols.clear();
    symbol_map.clear();
  }
  template <class ELFT>
  void add_symbol(const typename ELFT::Sym &symbol, llvm::StringRef strtab) {
    if (symbol.getType() != llvm::ELF::STT_FUNC) {
      return;
    }
    auto symbol_name = symbol.getName(strtab);
    assert(symbol_name);
    symbols.push_back(
        SymbolInfo{*symbol_name, symbol.st_value, symbol.st_size, &symbol});
  }
  void finish_adding() {
    llvm::sort(symbols, [](const SymbolInfo &LHS, const SymbolInfo &RHS) {
      return LHS.value < RHS.value;
    });
    for (const auto &symbol_info : symbols) {
      symbol_map.try_emplace(symbol_info.name, &symbol_info);
    }
  }
  const void *find_symbol_by_name(llvm::StringRef name) const {
    auto iter = symbol_map.find(name);
    return iter == symbol_map.end() ? nullptr : iter->second->symbol;
  }
  const void *find_symbol_contain_address(uint64_t addr) const {
    auto iter = llvm::upper_bound(
        symbols, addr,
        [](uint64_t LHS, const SymbolInfo &RHS) { return LHS < RHS.value; });
    if (iter == symbols.begin()) {
      return nullptr;
    }
    iter = std::prev(iter);
    if (iter->value + iter->size < addr) {
      return iter->symbol;
    }
    return nullptr;
  }
};

struct DecodedSlot {
  llvm::MCInst inst;
  uint32_t inst_size = 0; // 0 = continuation of previous multi-dword instruction
};

struct CachedSection {
  uint64_t start_addr;
  uint64_t byte_size;
  uint64_t file_offset;
  bool is_decoded = false;
  std::vector<DecodedSlot> slots;
};

class ObjectFileInfo {
  template <class ELFT>
  bool process_metadata_note(const typename ELFT::Note &note,
                             llvm::msgpack::DocNode &root);
  template <class ELFT>
  void disassemble(Disassembler &disas,
                   const llvm::object::ELFObjectFile<ELFT> &elf_obj);
  template <class ELFT>
  void scan_elf(const llvm::object::ELFObjectFile<ELFT> &elf_obj);
  template <class ELFT>
  void scan_section(const typename ELFT::Shdr &section,
                    const llvm::object::ELFFile<ELFT> &elf_file);

  void initialize_mc(Disassembler &disas);
  void ensure_section_decoded(CachedSection &sec) const;
  mutable std::vector<CachedSection> inst_cache;
  mutable CachedSection *hot_section = nullptr;

public:
  ObjectFileInfo(
      Disassembler &disas,
      const rocprofiler_callback_tracing_code_object_load_data_t &load_data);
  ObjectFileInfo(Disassembler &disas, const std::string &file_path,
                 uint64_t load_base);
  void init_elf(Disassembler &disas);
  const llvm::MCInst &decode_at(uint64_t addr, uint64_t &inst_size) const;
  void decode_all_sections() const;

  llvm::StringRef processor;
  bool sram_ecc_supported = false;
  bool xnack_supported = false;
  uint64_t load_base;
  uint64_t text_sec_offset;
  uint64_t text_sec_address;
  uint64_t text_sec_size;
  std::unique_ptr<llvm::MemoryBuffer> memory_buffer;
  std::unique_ptr<llvm::object::ObjectFile> object_file;
  llvm::MCSubtargetInfo *sub_target = nullptr;
  std::unique_ptr<llvm::MCContext> mc_ctx;
  std::unique_ptr<llvm::MCDisassembler> mc_dis_asm;
  std::unique_ptr<llvm::MCInstrInfo> mc_instr_info;
  std::unique_ptr<llvm::MCRegisterInfo> mc_reg_info;
  std::unique_ptr<llvm::MCCodeEmitter> mc_code_emitter;
  std::unique_ptr<llvm::MCAsmBackend> mc_asm_backend;
  std::unique_ptr<llvm::MCInstPrinter> inst_printer;
  std::string inst_str;
  std::unique_ptr<llvm::raw_string_ostream> inst_stream =
      std::make_unique<llvm::raw_string_ostream>(inst_str);
  std::unique_ptr<llvm::formatted_raw_ostream> fout =
      std::make_unique<llvm::formatted_raw_ostream>(*inst_stream);
  std::unique_ptr<llvm::MCStreamer> streamer;

  SymbolIndex symbol_index;
};

class Disassembler {
  llvm::DenseMap<uint64_t, ObjectFileInfo> object_files;
  llvm::DenseMap<SubTargetKey, std::unique_ptr<llvm::MCSubtargetInfo>>
      subtargets;
  llvm::Triple triple;
  const llvm::Target *target = nullptr;
  llvm::MCTargetOptions mc_options;
  std::unique_ptr<llvm::MCRegisterInfo> mc_reg_info;
  std::unique_ptr<llvm::MCAsmInfo> mc_asm_info;

public:
  Disassembler();
  llvm::MCSubtargetInfo *get_sub_target(llvm::StringRef mcpu,
                                        llvm::StringRef features);
  bool addCodeObject(uint64_t id, const std::string &file_path,
                     uint64_t load_base) {
    return object_files.try_emplace(id, *this, file_path, load_base).second;
  }
  bool addCodeObject(
      const rocprofiler_callback_tracing_code_object_load_data_t &load_data) {
    return object_files.try_emplace(load_data.code_object_id, *this, load_data)
        .second;
  }
  bool registerSymbol(
      const rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t
          &symbol_register) {
    return false;
  }
  ObjectFileInfo &get_object_file_by_id(uint64_t id) {
    auto iter = object_files.find(id);
    assert(iter != object_files.end());
    return iter->second;
  }
  const llvm::DenseMap<uint64_t, ObjectFileInfo> &get_object_files() const {
    return object_files;
  }
  const llvm::Triple &getTriple() const { return triple; }
  const llvm::Target *getTarget() const { return target; }
  const llvm::MCTargetOptions &getMCOptions() const { return mc_options; }
  const llvm::MCRegisterInfo &getMCRegisterInfo() const { return *mc_reg_info; }
  const llvm::MCAsmInfo &getMCAsmInfo() const { return *mc_asm_info; }
};

} // namespace my_rocperf_tool
