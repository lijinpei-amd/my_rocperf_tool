#include "my_rocperf_tool/disassembler.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCTargetOptionsCommandFlags.h"
#include "llvm/Support/Format.h"
#include "llvm/TargetParser/Triple.h"

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

} // namespace

namespace my_rocperf_tool {

template <class ELFT>
bool process_metadata_note(const typename ELFT::Note &note,
                           llvm::msgpack::DocNode &root) {
  return true;
}

template <class ELFT>
void ObjectFileInfo::scan_elf(
    const llvm::object::ELFObjectFile<ELFT> &elf_obj) {
  const auto &elf_file = elf_obj.getELFFile();
  auto elf_header = elf_file.getHeader();
  unsigned march = elf_header.e_flags & llvm::ELF::EF_AMDGPU_MACH;
  auto iter = llvm::find_if(isa_infos, [&](const IsaInfo &isa_info) {
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
  for (auto section : *sections)
    scan_section<ELFT>(section, elf_file);
}

template <typename ELFT>
void ObjectFileInfo::scan_section(const typename ELFT::Shdr &section,
                                  const llvm::object::ELFFile<ELFT> &elf_file) {
  auto sec_name = elf_file.getSectionName(section);
  assert(sec_name);
  if (*sec_name == ".symtab") {
    auto section_symbols = elf_file.symbols(&section);
    assert(section_symbols);
    auto symbol_strtab = elf_file.getStringTableForSymtab(section);
    assert(symbol_strtab);
    for (auto &symbol : *section_symbols) {
      symbol_index.add_symbol<ELFT>(symbol, *symbol_strtab);
    }
  }
  if (*sec_name == ".text") {
    text_sec_offset = section.sh_offset;
    text_sec_address = section.sh_addr;
    text_sec_size = section.sh_size;
  }
}

template <class ELFT>
void ObjectFileInfo::disassemble(
    Disassembler &disas, const llvm::object::ELFObjectFile<ELFT> &elf_obj) {
  return;
  const auto &elf_file = elf_obj.getELFFile();
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
      uint64_t virt_addr = load_base;
      while (!bytes.empty()) {
        llvm::MCInst inst;
        uint64_t inst_size;
        auto status = mc_dis_asm->getInstruction(inst, inst_size, bytes,
                                                 virt_addr, llvm::nulls());
        if (status == llvm::MCDisassembler::Success) {
          streamer->emitInstruction(inst, *sub_target);
        } else {
          assert(false);
        }
        virt_addr += inst_size;
        bytes = bytes.drop_front(inst_size);
      }
    }
  }
  inst_str.clear();
}

void ObjectFileInfo::initialize_mc(Disassembler &disas) {
  auto *target = disas.getTarget();
  auto &triple = disas.getTriple();
  sub_target = disas.get_sub_target(processor, "");
  mc_ctx = std::make_unique<llvm::MCContext>(
      triple, &disas.getMCAsmInfo(), &disas.getMCRegisterInfo(), sub_target,
      nullptr, &disas.getMCOptions());
  mc_dis_asm.reset(target->createMCDisassembler(*sub_target, *mc_ctx));

  mc_instr_info.reset(target->createMCInstrInfo());

  mc_code_emitter.reset(target->createMCCodeEmitter(*mc_instr_info, *mc_ctx));

  mc_asm_backend.reset(target->createMCAsmBackend(
      *sub_target, disas.getMCRegisterInfo(), disas.getMCOptions()));
  inst_printer.reset(target->createMCInstPrinter(
      triple, disas.getMCAsmInfo().getAssemblerDialect(), disas.getMCAsmInfo(),
      *mc_instr_info, disas.getMCRegisterInfo()));
  streamer.reset(target->createAsmStreamer(
      *mc_ctx, std::move(fout), std::move(inst_printer),
      std::move(mc_code_emitter), std::move(mc_asm_backend)));
}

void ObjectFileInfo::init_elf(Disassembler &disas) {
  auto obj_or_err =
      llvm::object::ObjectFile::createELFObjectFile(*memory_buffer);
  assert(obj_or_err);
  object_file = std::move(*obj_or_err);
  if (auto *elf32_le =
          llvm::dyn_cast<llvm::object::ELF32LEObjectFile>(object_file.get())) {
    scan_elf(*elf32_le);
    initialize_mc(disas);
    disassemble(disas, *elf32_le);
  } else if (auto *elf64_le = llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(
                 object_file.get())) {
    scan_elf(*elf64_le);
    initialize_mc(disas);
    disassemble(disas, *elf64_le);
  } else if (auto *elf32_be = llvm::dyn_cast<llvm::object::ELF32BEObjectFile>(
                 object_file.get())) {
    scan_elf(*elf32_be);
    initialize_mc(disas);
    disassemble(disas, *elf32_be);
  } else if (auto *elf64_be = llvm::dyn_cast<llvm::object::ELF64BEObjectFile>(
                 object_file.get())) {
    scan_elf(*elf64_be);
    initialize_mc(disas);
    disassemble(disas, *elf64_be);
  } else {
    assert(false);
  }
}

ObjectFileInfo::ObjectFileInfo(
    Disassembler &disas,
    const rocprofiler_callback_tracing_code_object_load_data_t &load_data)
    : load_base(load_data.load_base) {
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
        llvm::StringRef(reinterpret_cast<const char *>(load_data.memory_base),
                        load_data.memory_size),
        load_data.uri);
    break;
  }
  default:
    break;
  }
  init_elf(disas);
}

ObjectFileInfo::ObjectFileInfo(Disassembler &disas,
                               const std::string &file_path, uint64_t load_base)
    : load_base(load_base) {
  auto file_or_err =
      llvm::MemoryBuffer::getFileOrSTDIN(file_path, false, false);
  assert(file_or_err);
  memory_buffer = std::move(*file_or_err);
  init_elf(disas);
}

llvm::MCInst ObjectFileInfo::decode_at(uint64_t addr,
                                       uint64_t &inst_size) const {
  auto buffer_size = memory_buffer->getBufferSize();
  assert(addr >= text_sec_address && addr < text_sec_address + text_sec_size);
  uint64_t file_offset = addr - text_sec_address + text_sec_offset;
  llvm::MCInst inst;
  llvm::ArrayRef<unsigned char> bytes{
      (unsigned char *)memory_buffer->getBufferStart() + file_offset,
      buffer_size};
  auto status = mc_dis_asm->getInstruction(inst, inst_size, bytes,
                                           addr + load_base, llvm::nulls());
  assert(status == llvm::MCDisassembler::Success);
  (void)status;
  return inst;
}

Disassembler::Disassembler() {
  triple = llvm::Triple(llvm::Triple::normalize("amdgcn-amd-amdhsa"));
  std::string error;
  target = llvm::TargetRegistry::lookupTarget(triple, error);
  mc_options = llvm::mc::InitMCTargetOptionsFromFlags();
  mc_reg_info.reset(target->createMCRegInfo(triple));
  mc_asm_info.reset(target->createMCAsmInfo(*mc_reg_info, triple, mc_options));
}

llvm::MCSubtargetInfo *Disassembler::get_sub_target(llvm::StringRef mcpu,
                                                    llvm::StringRef features) {
  assert(target);
  auto res = target->createMCSubtargetInfo(triple, mcpu, features);
  assert(res);
  return res;
}

} // namespace my_rocperf_tool
