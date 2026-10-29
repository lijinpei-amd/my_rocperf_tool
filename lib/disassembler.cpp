#include "my_rocperf_tool/disassembler.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "rocprofiler-sdk/callback_tracing.h"

#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace {
struct IsaInfo {
  llvm::StringRef IsaName;
  llvm::StringRef Processor;
  bool SrameccSupported;
  bool XnackSupported;
  unsigned ElfMachine;
  bool TrapHandlerEnabled;
  bool ImageSupport;
  unsigned LDSSize;
  unsigned LDSBankCount;
  unsigned EUsPerCU;
  unsigned MaxWavesPerCU;
  unsigned MaxFlatWorkGroupSize;
  unsigned SGPRAllocGranule;
  unsigned TotalNumSGPRs;
  unsigned AddressableNumSGPRs;
  unsigned VGPRAllocGranule;
  unsigned TotalNumVGPRs;
  // TODO: Update this to AvailableNumVGPRs to be more accurate
  unsigned AddressableNumVGPRs;
} isa_infos[] = {
#define HANDLE_ISA(TARGET_TRIPLE, PROCESSOR, SRAMECC_SUPPORTED,                \
                   XNACK_SUPPORTED, ELF_MACHINE, TRAP_HANDLER_ENABLED,         \
                   IMAGE_SUPPORT, LDS_SIZE, LDS_BANK_COUNT, EUS_PER_CU,        \
                   MAX_WAVES_PER_CU, MAX_FLAT_WORK_GROUP_SIZE,                 \
                   SGPR_ALLOC_GRANULE, TOTAL_NUM_SGPRS, ADDRESSABLE_NUM_SGPRS, \
                   VGPR_ALLOC_GRANULE, TOTAL_NUM_VGPRS, ADDRESSABLE_NUM_VGPRS) \
  {TARGET_TRIPLE "-" PROCESSOR,                                                \
   PROCESSOR,                                                                  \
   SRAMECC_SUPPORTED,                                                          \
   XNACK_SUPPORTED,                                                            \
   llvm::ELF::ELF_MACHINE,                                                     \
   TRAP_HANDLER_ENABLED,                                                       \
   IMAGE_SUPPORT,                                                              \
   LDS_SIZE,                                                                   \
   LDS_BANK_COUNT,                                                             \
   EUS_PER_CU,                                                                 \
   MAX_WAVES_PER_CU,                                                           \
   MAX_FLAT_WORK_GROUP_SIZE,                                                   \
   SGPR_ALLOC_GRANULE,                                                         \
   TOTAL_NUM_SGPRS,                                                            \
   ADDRESSABLE_NUM_SGPRS,                                                      \
   VGPR_ALLOC_GRANULE,                                                         \
   TOTAL_NUM_VGPRS,                                                            \
   ADDRESSABLE_NUM_VGPRS},
#include "my_rocperf_tool/data/comgr-isa-metadata.def"
};

}  // namespace

namespace my_rocperf_tool {

// The MC objects that vary with the ISA -- everything downstream of one
// (mcpu, features) pair. Code objects sharing an ISA share one of these; the
// owning Disassembler caches them by SubTargetKey, so the whole stack is built
// once per gfx target rather than once per loaded executable. So is the kind
// table: an opcode is classified once per ISA for the Disassembler's lifetime.
//
// Not thread-safe: print_inst writes a shared scratch buffer, kind_of fills
// its table on first use, and the MC disassembler carries per-call state. That
// is fine as used, because each worker thread owns a Disassembler and
// therefore its own contexts.
class SubTargetContext {
 public:
  SubTargetContext(const Disassembler& disas, llvm::StringRef mcpu,
                   llvm::StringRef features);

  const llvm::MCDisassembler& get_mc_disassembler() const {
    return *mc_dis_asm;
  }

  // Renders one instruction to text, trimmed. The result points into a scratch
  // buffer that the next call overwrites -- copy it to keep it longer.
  llvm::StringRef print_inst(const llvm::MCInst& inst) const;

  InstKind kind_of(unsigned opcode) const {
    auto& kind = kinds[opcode];
    if (LLVM_UNLIKELY(kind < 0)) kind = static_cast<int8_t>(classify(opcode));
    return static_cast<InstKind>(kind);
  }

 private:
  InstKind classify(unsigned opcode) const;

  // An InstKind per opcode, -1 until first queried. Bytes rather than
  // std::optional, so filling a table of every opcode is one memset.
  mutable std::vector<int8_t> kinds;
  // Declaration order is lifetime order. The streamer holds a
  // formatted_raw_ostream over inst_stream, an inst printer and code emitter
  // over mc_instr_info/mc_ctx, and an asm backend over sub_target, so it must
  // be destroyed before any of them.
  mutable std::string inst_str;
  std::unique_ptr<llvm::raw_string_ostream> inst_stream;
  std::unique_ptr<llvm::MCSubtargetInfo> sub_target;
  std::unique_ptr<llvm::MCInstrInfo> mc_instr_info;
  std::unique_ptr<llvm::MCContext> mc_ctx;
  std::unique_ptr<llvm::MCDisassembler> mc_dis_asm;
  std::unique_ptr<llvm::MCStreamer> streamer;
};

SubTargetContext::SubTargetContext(const Disassembler& disas,
                                   llvm::StringRef mcpu,
                                   llvm::StringRef features)
    : inst_stream(std::make_unique<llvm::raw_string_ostream>(inst_str)) {
  const auto* target = disas.target;
  const auto& triple = disas.triple;
  const auto& asm_info = *disas.mc_asm_info;
  const auto& reg_info = *disas.mc_reg_info;

  sub_target.reset(target->createMCSubtargetInfo(triple, mcpu, features));
  assert(sub_target);
  mc_instr_info.reset(target->createMCInstrInfo());
  kinds.assign(mc_instr_info->getNumOpcodes(), -1);
  mc_ctx = std::make_unique<llvm::MCContext>(triple, &asm_info, &reg_info,
                                             sub_target.get(), nullptr,
                                             &disas.mc_options);
  mc_dis_asm.reset(target->createMCDisassembler(*sub_target, *mc_ctx));

  // The streamer takes ownership of all four; only it ever touches them.
  auto fout = std::make_unique<llvm::formatted_raw_ostream>(*inst_stream);
  std::unique_ptr<llvm::MCInstPrinter> printer(
      target->createMCInstPrinter(triple, asm_info.getAssemblerDialect(),
                                  asm_info, *mc_instr_info, reg_info));
  std::unique_ptr<llvm::MCCodeEmitter> code_emitter(
      target->createMCCodeEmitter(*mc_instr_info, *mc_ctx));
  std::unique_ptr<llvm::MCAsmBackend> asm_backend(
      target->createMCAsmBackend(*sub_target, reg_info, disas.mc_options));
  streamer.reset(target->createAsmStreamer(
      *mc_ctx, std::move(fout), std::move(printer), std::move(code_emitter),
      std::move(asm_backend)));
}

llvm::StringRef SubTargetContext::print_inst(const llvm::MCInst& inst) const {
  inst_str.clear();
  streamer->emitInstruction(inst, *sub_target);
  return llvm::StringRef(inst_str).trim();
}

InstKind SubTargetContext::classify(unsigned opcode) const {
  auto name = mc_instr_info->getName(opcode);
  if (name.starts_with("V_MFMA")) return InstKind::Mfma;
  // LLVM AMDGPU decorates opcode names with a lowercase subtarget tag
  // ("S_BARRIER_vi", "S_BARRIER_gfx10", ...). The gfx12 split-barrier
  // family is distinct: S_BARRIER_WAIT_*, S_BARRIER_SIGNAL_*, etc. -- an
  // uppercase mnemonic word follows S_BARRIER_. Match the classic
  // s_barrier across subtargets without catching the split family.
  constexpr llvm::StringRef kPrefix = "S_BARRIER_";
  if (name == "S_BARRIER" ||
      (name.starts_with(kPrefix) && name.size() > kPrefix.size() &&
       std::islower(static_cast<unsigned char>(name[kPrefix.size()]))))
    return InstKind::ClassicBarrier;
  return InstKind::Other;
}

struct ObjectFileInfo::Slot {
  llvm::MCInst inst;
  uint32_t inst_size = 0;  // 0 = continuation of previous multi-dword inst
};

struct ObjectFileInfo::Section {
  uint64_t start_addr;
  uint64_t byte_size;
  // Points into the owning ObjectFileInfo's buffer.
  const uint8_t* bytes;
  bool is_decoded = false;
  std::vector<Slot> slots;

  bool contains(uint64_t vaddr) const {
    return vaddr >= start_addr && vaddr - start_addr < byte_size;
  }
};

ObjectFileInfo::ObjectFileInfo(std::unique_ptr<llvm::MemoryBuffer> buffer,
                               std::vector<Section> sections,
                               const SubTargetContext& sub_target_ctx)
    : buffer(std::move(buffer)),
      sections(std::move(sections)),
      sub_target_ctx(&sub_target_ctx) {}

ObjectFileInfo::ObjectFileInfo(ObjectFileInfo&&) = default;
ObjectFileInfo::~ObjectFileInfo() = default;

llvm::Expected<ObjectFileInfo> ObjectFileInfo::load(
    Disassembler& disas, std::unique_ptr<llvm::MemoryBuffer> buffer) {
  auto fail = [&](const llvm::Twine& reason) {
    return llvm::createStringError(buffer->getBufferIdentifier() + ": " +
                                   reason);
  };
  auto obj_or_err =
      llvm::object::ObjectFile::createELFObjectFile(buffer->getMemBufferRef());
  if (!obj_or_err) return fail(llvm::toString(obj_or_err.takeError()));
  // ELF32/64 and LE/BE differ only in the header types, which the base class
  // already reads.
  const auto& elf = llvm::cast<llvm::object::ELFObjectFileBase>(**obj_or_err);
  if (elf.getEMachine() != llvm::ELF::EM_AMDGPU)
    return fail("not an AMDGPU code object (e_machine " +
                llvm::Twine(elf.getEMachine()) + ")");

  unsigned march = elf.getPlatformFlags() & llvm::ELF::EF_AMDGPU_MACH;
  auto isa = llvm::find_if(isa_infos, [&](const IsaInfo& isa_info) {
    return isa_info.ElfMachine == march;
  });
  if (isa == std::end(isa_infos))
    return fail("unknown AMDGPU ISA (EF_AMDGPU_MACH 0x" +
                llvm::utohexstr(march) + ")");

  std::vector<Section> sections;
  for (const auto& section : elf.sections()) {
    auto name = section.getName();
    if (!name) return fail(llvm::toString(name.takeError()));
    if (*name != ".text") continue;
    auto contents = section.getContents();
    if (!contents) return fail(llvm::toString(contents.takeError()));
    sections.push_back(
        {section.getAddress(), contents->size(), contents->bytes_begin()});
  }
  llvm::sort(sections, [](const Section& a, const Section& b) {
    return a.start_addr < b.start_addr;
  });
  const auto& sub_target_ctx = disas.get_sub_target_context(isa->Processor, "");
  return ObjectFileInfo(std::move(buffer), std::move(sections), sub_target_ctx);
}

ObjectFileInfo::Section* ObjectFileInfo::find_section(uint64_t vaddr) const {
  auto it = llvm::upper_bound(
      sections, vaddr,
      [](uint64_t a, const Section& s) { return a < s.start_addr; });
  if (it == sections.begin()) return nullptr;
  --it;
  return it->contains(vaddr) ? &*it : nullptr;
}

void ObjectFileInfo::ensure_section_decoded(Section& sec) const {
  if (LLVM_LIKELY(sec.is_decoded)) return;
  sec.slots.resize(sec.byte_size / 4);

  uint64_t offset = 0;
  while (offset < sec.byte_size) {
    uint64_t inst_size;
    llvm::ArrayRef<uint8_t> bytes{sec.bytes + offset, sec.byte_size - offset};
    auto& slot = sec.slots[offset / 4];
    // Decode at the ELF vaddr, which is also the space every queried address
    // lives in -- the trace decoder subtracts the code object's load delta
    // before reporting a PC. MC only consults this address to symbolize
    // operands, and no MCSymbolizer is installed, so it currently has no
    // effect on the decoded instruction. Installing one would make the choice
    // of address space observable, and vaddrs are what the rest of the tool
    // reports; runtime addresses would need the delta added back.
    auto status = sub_target_ctx->get_mc_disassembler().getInstruction(
        slot.inst, inst_size, bytes, sec.start_addr + offset, llvm::nulls());
    assert(status == llvm::MCDisassembler::Success);
    (void)status;
    slot.inst_size = static_cast<uint32_t>(inst_size);
    offset += inst_size;
  }
  sec.is_decoded = true;
}

const ObjectFileInfo::Slot& ObjectFileInfo::slot_at(uint64_t vaddr) const {
  if (LLVM_UNLIKELY(!(hot_section && hot_section->contains(vaddr)))) {
    hot_section = find_section(vaddr);
    assert(hot_section && "vaddr outside every .text section");
  }
  ensure_section_decoded(*hot_section);
  size_t slot_idx = (vaddr - hot_section->start_addr) / 4;
  assert(slot_idx < hot_section->slots.size());
  const auto& slot = hot_section->slots[slot_idx];
  assert(slot.inst_size > 0 && "vaddr inside a multi-dword instruction");
  return slot;
}

uint32_t ObjectFileInfo::size_at(uint64_t vaddr) const {
  return slot_at(vaddr).inst_size;
}

InstKind ObjectFileInfo::kind_at(uint64_t vaddr) const {
  return sub_target_ctx->kind_of(slot_at(vaddr).inst.getOpcode());
}

std::string ObjectFileInfo::text_at(uint64_t vaddr) const {
  return sub_target_ctx->print_inst(slot_at(vaddr).inst).str();
}

size_t ObjectFileInfo::copy_text_at(uint64_t vaddr,
                                    llvm::MutableArrayRef<char> buf) const {
  auto text = sub_target_ctx->print_inst(slot_at(vaddr).inst);
  if (text.size() <= buf.size())
    std::memcpy(buf.data(), text.data(), text.size());
  return text.size();
}

std::vector<uint64_t> ObjectFileInfo::inst_vaddrs(uint64_t first,
                                                  uint64_t last) const {
  std::vector<uint64_t> vaddrs;
  const auto* sec = find_section(first);
  if (!sec || !sec->contains(last)) return vaddrs;
  // `last` lies inside the section, so the walk cannot run off its end.
  for (uint64_t vaddr = first; vaddr <= last; vaddr += size_at(vaddr))
    vaddrs.push_back(vaddr);
  return vaddrs;
}

void ObjectFileInfo::decode_all_sections() const {
  for (auto& sec : sections) ensure_section_decoded(sec);
}

namespace {
// Registers only what the MC stack is built from: the target itself, its MC
// layer (which also supplies the inst printer, code emitter, asm backend and
// asm target streamer) and its MC disassembler. TargetRegistry is not
// thread-safe and workers construct Disassemblers concurrently, hence once per
// process.
void register_amdgpu_target() {
  static std::once_flag once;
  std::call_once(once, [] {
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUDisassembler();
  });
}
}  // namespace

Disassembler::Disassembler() {
  register_amdgpu_target();
  triple = llvm::Triple(llvm::Triple::normalize("amdgcn-amd-amdhsa"));
  std::string error;
  target = llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target)
    llvm::reportFatalInternalError(
        llvm::Twine("cannot look up the AMDGPU target: ") + error);
  mc_reg_info.reset(target->createMCRegInfo(triple));
  mc_asm_info.reset(target->createMCAsmInfo(*mc_reg_info, triple, mc_options));
}

Disassembler::~Disassembler() = default;

llvm::Error Disassembler::add_code_object(uint64_t id,
                                          const std::string& file_path) {
  if (object_files.count(id)) return llvm::Error::success();
  auto buffer = llvm::MemoryBuffer::getFile(file_path, /*IsText=*/false,
                                            /*RequiresNullTerminator=*/false);
  if (!buffer)
    return llvm::createStringError(
        buffer.getError(), file_path + ": " + buffer.getError().message());
  return add_buffer(id, std::move(*buffer));
}

llvm::Error Disassembler::add_code_object(
    const rocprofiler_callback_tracing_code_object_load_data_t& load_data) {
  const uint64_t id = load_data.code_object_id;
  if (object_files.count(id)) return llvm::Error::success();
  std::unique_ptr<llvm::MemoryBuffer> buffer;
  switch (load_data.storage_type) {
    case ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE: {
      // -1 is LLVM's "stat the descriptor" sentinel for the file size.
      // load_size measures the loaded image in vaddr space and bears no
      // relation to the file's length, which also carries the non-alloc
      // sections and the section header table. It usually falls short, which
      // truncates the buffer past the section headers and fails the ELF parse.
      auto open_file = llvm::MemoryBuffer::getOpenFile(
          load_data.storage_file, load_data.uri, -1, false, true);
      if (!open_file)
        return llvm::createStringError(
            open_file.getError(),
            llvm::Twine(load_data.uri) + ": " + open_file.getError().message());
      buffer = std::move(*open_file);
      break;
    }
    case ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY: {
      buffer = llvm::MemoryBuffer::getMemBufferCopy(
          llvm::StringRef(reinterpret_cast<const char*>(load_data.memory_base),
                          load_data.memory_size),
          load_data.uri);
      break;
    }
    default:
      return llvm::createStringError(
          llvm::Twine(load_data.uri) +
          ": unsupported code object storage type " +
          llvm::Twine(static_cast<int>(load_data.storage_type)));
  }
  return add_buffer(id, std::move(buffer));
}

llvm::Error Disassembler::add_buffer(
    uint64_t id, std::unique_ptr<llvm::MemoryBuffer> buffer) {
  auto loaded = ObjectFileInfo::load(*this, std::move(buffer));
  if (!loaded) return loaded.takeError();
  object_files.try_emplace(id, std::move(*loaded));
  return llvm::Error::success();
}

const ObjectFileInfo* Disassembler::find_code_object(uint64_t id) const {
  auto it = object_files.find(id);
  return it != object_files.end() ? &it->second : nullptr;
}

void Disassembler::decode_all() const {
  for (const auto& [id, obj] : object_files) obj.decode_all_sections();
}

const SubTargetContext& Disassembler::get_sub_target_context(
    llvm::StringRef mcpu, llvm::StringRef features) {
  auto [iter, inserted] = sub_target_contexts.try_emplace(
      SubTargetKey{target, triple, mcpu.str(), features.str()});
  if (inserted)
    iter->second = std::make_unique<SubTargetContext>(*this, mcpu, features);
  return *iter->second;
}

}  // namespace my_rocperf_tool
