#include "my_rocperf_tool/disassembler.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCTargetOptionsCommandFlags.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "rocprofiler-sdk/callback_tracing.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>

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

SubTargetContext::SubTargetContext(const Disassembler& disas,
                                   llvm::StringRef mcpu,
                                   llvm::StringRef features)
    : inst_stream(std::make_unique<llvm::raw_string_ostream>(inst_str)) {
  const auto* target = disas.getTarget();
  const auto& triple = disas.getTriple();
  const auto& asm_info = disas.getMCAsmInfo();
  const auto& reg_info = disas.getMCRegisterInfo();

  sub_target.reset(target->createMCSubtargetInfo(triple, mcpu, features));
  assert(sub_target);
  mc_instr_info.reset(target->createMCInstrInfo());
  mc_ctx = std::make_unique<llvm::MCContext>(triple, &asm_info, &reg_info,
                                             sub_target.get(), nullptr,
                                             &disas.getMCOptions());
  mc_dis_asm.reset(target->createMCDisassembler(*sub_target, *mc_ctx));

  // The streamer takes ownership of all four; only it ever touches them.
  auto fout = std::make_unique<llvm::formatted_raw_ostream>(*inst_stream);
  std::unique_ptr<llvm::MCInstPrinter> printer(
      target->createMCInstPrinter(triple, asm_info.getAssemblerDialect(),
                                  asm_info, *mc_instr_info, reg_info));
  std::unique_ptr<llvm::MCCodeEmitter> code_emitter(
      target->createMCCodeEmitter(*mc_instr_info, *mc_ctx));
  std::unique_ptr<llvm::MCAsmBackend> asm_backend(
      target->createMCAsmBackend(*sub_target, reg_info, disas.getMCOptions()));
  streamer.reset(target->createAsmStreamer(
      *mc_ctx, std::move(fout), std::move(printer), std::move(code_emitter),
      std::move(asm_backend)));
}

llvm::StringRef SubTargetContext::print_inst(const llvm::MCInst& inst) const {
  inst_str.clear();
  streamer->emitInstruction(inst, *sub_target);
  return llvm::StringRef(inst_str).trim();
}

template <class ELFT>
void ObjectFileInfo::scan_elf(
    const llvm::object::ELFObjectFile<ELFT>& elf_obj) {
  const auto& elf_file = elf_obj.getELFFile();
  auto elf_header = elf_file.getHeader();
  unsigned march = elf_header.e_flags & llvm::ELF::EF_AMDGPU_MACH;
  auto iter = llvm::find_if(isa_infos, [&](const IsaInfo& isa_info) {
    return isa_info.ElfMachine == march;
  });
  if (iter == std::end(isa_infos)) {
    return;
  }
  processor = iter->Processor;
  sram_ecc_supported = iter->SrameccSupported;
  xnack_supported = iter->XnackSupported;

  auto sections = elf_file.sections();
  assert(sections);
  for (auto section : *sections) scan_section<ELFT>(section, elf_file);
}

template <typename ELFT>
void ObjectFileInfo::scan_section(const typename ELFT::Shdr& section,
                                  const llvm::object::ELFFile<ELFT>& elf_file) {
  auto sec_name = elf_file.getSectionName(section);
  assert(sec_name);
  if (*sec_name == ".symtab") {
    auto section_symbols = elf_file.symbols(&section);
    assert(section_symbols);
    auto symbol_strtab = elf_file.getStringTableForSymtab(section);
    assert(symbol_strtab);
    for (auto& symbol : *section_symbols) {
      symbol_index.add_symbol<ELFT>(symbol, *symbol_strtab);
    }
  }
  if (*sec_name == ".text") {
    text_sec_offset = section.sh_offset;
    text_sec_address = section.sh_addr;
    text_sec_size = section.sh_size;
    inst_cache.push_back({section.sh_addr, section.sh_size, section.sh_offset});
  }
}

template <class ELFT>
void ObjectFileInfo::disassemble(
    Disassembler& disas, const llvm::object::ELFObjectFile<ELFT>& elf_obj) {
  return;
  const auto& elf_file = elf_obj.getELFFile();
  auto sections = elf_file.sections();
  assert(sections);
  for (auto section : *sections) {
    auto sec_name = elf_file.getSectionName(section);
    assert(sec_name);
    if (*sec_name == ".text") {
      auto buffer = memory_buffer->getBuffer();
      auto section_buffer =
          buffer.drop_front(section.sh_offset).take_front(section.sh_size);
      llvm::ArrayRef<unsigned char> bytes{section_buffer.bytes_begin(),
                                          section_buffer.size()};
      uint64_t virt_addr = section.sh_addr;
      while (!bytes.empty()) {
        llvm::MCInst inst;
        uint64_t inst_size;
        auto status = sub_target_ctx->get_mc_disassembler().getInstruction(
            inst, inst_size, bytes, virt_addr, llvm::nulls());
        if (status == llvm::MCDisassembler::Success) {
          sub_target_ctx->print_inst(inst);
        } else {
          assert(false);
        }
        virt_addr += inst_size;
        bytes = bytes.drop_front(inst_size);
      }
    }
  }
}

void ObjectFileInfo::initialize_mc(Disassembler& disas) {
  llvm::sort(inst_cache, [](const CachedSection& a, const CachedSection& b) {
    return a.start_addr < b.start_addr;
  });
  sub_target_ctx = &disas.get_sub_target_context(processor, "");
}

void ObjectFileInfo::init_elf(Disassembler& disas) {
  auto obj_or_err =
      llvm::object::ObjectFile::createELFObjectFile(*memory_buffer);
  assert(obj_or_err);
  object_file = std::move(*obj_or_err);
  if (auto* elf32_le =
          llvm::dyn_cast<llvm::object::ELF32LEObjectFile>(object_file.get())) {
    scan_elf(*elf32_le);
    initialize_mc(disas);
    disassemble(disas, *elf32_le);
  } else if (auto* elf64_le = llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(
                 object_file.get())) {
    scan_elf(*elf64_le);
    initialize_mc(disas);
    disassemble(disas, *elf64_le);
  } else if (auto* elf32_be = llvm::dyn_cast<llvm::object::ELF32BEObjectFile>(
                 object_file.get())) {
    scan_elf(*elf32_be);
    initialize_mc(disas);
    disassemble(disas, *elf32_be);
  } else if (auto* elf64_be = llvm::dyn_cast<llvm::object::ELF64BEObjectFile>(
                 object_file.get())) {
    scan_elf(*elf64_be);
    initialize_mc(disas);
    disassemble(disas, *elf64_be);
  } else {
    assert(false);
  }
}

ObjectFileInfo::ObjectFileInfo(
    Disassembler& disas,
    const rocprofiler_callback_tracing_code_object_load_data_t& load_data) {
  switch (load_data.storage_type) {
    case ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE: {
      auto open_file =
          llvm::MemoryBuffer::getOpenFile(load_data.storage_file, load_data.uri,
                                          load_data.load_size, false, true);
      memory_buffer = std::move(*open_file);
      break;
    }
    case ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY: {
      memory_buffer = llvm::MemoryBuffer::getMemBufferCopy(
          llvm::StringRef(reinterpret_cast<const char*>(load_data.memory_base),
                          load_data.memory_size),
          load_data.uri);
      break;
    }
    default:
      break;
  }
  init_elf(disas);
}

ObjectFileInfo::ObjectFileInfo(Disassembler& disas,
                               const std::string& file_path) {
  auto file_or_err =
      llvm::MemoryBuffer::getFileOrSTDIN(file_path, false, false);
  assert(file_or_err);
  memory_buffer = std::move(*file_or_err);
  init_elf(disas);
}

void ObjectFileInfo::ensure_section_decoded(CachedSection& sec) const {
  if (LLVM_LIKELY(sec.is_decoded)) return;
  sec.slots.resize(sec.byte_size / 4);

  const auto* base =
      reinterpret_cast<const unsigned char*>(memory_buffer->getBufferStart()) +
      sec.file_offset;
  uint64_t remaining = memory_buffer->getBufferSize() - sec.file_offset;

  uint64_t offset = 0;
  while (offset < sec.byte_size) {
    uint64_t inst_size;
    llvm::ArrayRef<unsigned char> bytes{base + offset, remaining - offset};
    auto& slot = sec.slots[offset / 4];
    // Decode at the ELF vaddr, which is also the space every address handed
    // to decode_at lives in -- the trace decoder subtracts the code object's
    // load delta before reporting a PC. MC only consults this address to
    // symbolize operands, and no MCSymbolizer is installed, so it currently
    // has no effect on the decoded instruction. Installing one would make the
    // choice of address space observable, and vaddrs are what the rest of the
    // tool reports; runtime addresses would need the delta added back.
    auto status = sub_target_ctx->get_mc_disassembler().getInstruction(
        slot.inst, inst_size, bytes, sec.start_addr + offset, llvm::nulls());
    assert(status == llvm::MCDisassembler::Success);
    (void)status;
    slot.inst_size = static_cast<uint32_t>(inst_size);
    offset += inst_size;
  }
  sec.is_decoded = true;
}

const llvm::MCInst& ObjectFileInfo::decode_at(uint64_t addr,
                                              uint64_t& inst_size) const {
  // Fast path: check last accessed section.
  if (LLVM_UNLIKELY(
          !(hot_section && addr >= hot_section->start_addr &&
            addr < hot_section->start_addr + hot_section->byte_size))) {
    // Binary search for the section containing addr.
    auto it = llvm::upper_bound(
        inst_cache, addr,
        [](uint64_t a, const CachedSection& s) { return a < s.start_addr; });
    assert(it != inst_cache.begin());
    --it;
    assert(addr >= it->start_addr && addr < it->start_addr + it->byte_size);
    hot_section = &*it;
  }
  ensure_section_decoded(*hot_section);
  size_t slot_idx = (addr - hot_section->start_addr) / 4;
  assert(slot_idx < hot_section->slots.size());
  const auto& slot = hot_section->slots[slot_idx];
  assert(slot.inst_size > 0);
  inst_size = slot.inst_size;
  return slot.inst;
}

void ObjectFileInfo::decode_all_sections() const {
  for (auto& sec : inst_cache) ensure_section_decoded(sec);
}

Disassembler::Disassembler() {
  triple = llvm::Triple(llvm::Triple::normalize("amdgcn-amd-amdhsa"));
  std::string error;
  target = llvm::TargetRegistry::lookupTarget(triple, error);
  mc_options = llvm::mc::InitMCTargetOptionsFromFlags();
  mc_reg_info.reset(target->createMCRegInfo(triple));
  mc_asm_info.reset(target->createMCAsmInfo(*mc_reg_info, triple, mc_options));
}

const SubTargetContext& Disassembler::get_sub_target_context(
    llvm::StringRef mcpu, llvm::StringRef features) {
  assert(target);
  auto [iter, inserted] = sub_target_contexts.try_emplace(
      SubTargetKey{target, triple, mcpu.str(), features.str()});
  if (inserted)
    iter->second = std::make_unique<SubTargetContext>(*this, mcpu, features);
  return *iter->second;
}

}  // namespace my_rocperf_tool
