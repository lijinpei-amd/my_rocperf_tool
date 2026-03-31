#include "my_rocperf_tool/att_output_dir.h"
#include "my_rocperf_tool/disassembler.h"
#include "my_rocperf_tool/init_llvm.h"
#include "my_rocperf_tool/inst_statistics.h"

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

#if ROCPROF_TRACE_DECODER_NEW
#include "trace_decoder_api.h"
#else
#include "rocprof_trace_decoder.h"
#endif

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCInst.h"

#include <iostream>
#include <optional>
#include <string>

ABSL_FLAG(std::optional<std::string>, att_output_dir, std::nullopt,
          "output file dir");

struct trace_decoder_context {
  bool first_run;
  std::unique_ptr<char[]> att_file_content;
  size_t att_file_size;
  my_rocperf_tool::Disassembler disas;
  my_rocperf_tool::InstStatistics stats;
};

uint64_t att_decoder_se_data_callback(uint8_t **buffer, uint64_t *buffer_size,
                                      void *userdata) {
  auto &ctx = *reinterpret_cast<trace_decoder_context *>(userdata);
  if (!ctx.first_run) {
    return 0;
  }
  ctx.first_run = false;
  uint64_t data_size = ctx.att_file_size;
  *buffer_size = data_size;
  *buffer = (uint8_t *)ctx.att_file_content.get();
  return data_size;
}

rocprofiler_thread_trace_decoder_status_t att_decoder_isa_callback(
    char *isa_instruction, uint64_t *isa_memory_size, uint64_t *isa_size,
    rocprofiler_thread_trace_decoder_pc_t pc, void *userdata) {
  assert(pc.code_object_id != 0);
  auto &ctx = *reinterpret_cast<trace_decoder_context *>(userdata);
  auto &obj_file = ctx.disas.get_object_file_by_id(pc.code_object_id);
  uint64_t inst_size;
  auto mc_inst = obj_file.decode_at(pc.address, inst_size);
  obj_file.streamer->emitInstruction(mc_inst, *obj_file.sub_target);
  std::string inst_str = std::move(obj_file.inst_str);
  obj_file.inst_str.clear();
  llvm::StringRef inst_ref = llvm::StringRef(inst_str).trim();
  auto str_size = inst_ref.size();
  if (*isa_size < str_size) {
    *isa_size = str_size;
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  memcpy(isa_instruction, inst_ref.data(), str_size);
  *isa_size = str_size;
  *isa_memory_size = inst_size;
  return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

rocprofiler_thread_trace_decoder_status_t att_decoder_trace_callback(
    rocprofiler_thread_trace_decoder_record_type_t record_type_id,
    void *trace_events, uint64_t trace_size, void *userdata) {
  auto &ctx = *reinterpret_cast<trace_decoder_context *>(userdata);
  switch (record_type_id) {
  default:
    break;
  case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE:
    auto *wave_events =
        static_cast<rocprofiler_thread_trace_decoder_wave_t *>(trace_events);
    for (size_t wave_n = 0; wave_n < trace_size; wave_n++) {
      const auto &wave = wave_events[wave_n];

      int64_t last_time = wave.begin_time;
      for (size_t j = 0; j < wave.instructions_size; j++) {
        const auto &inst = wave.instructions_array[j];
        if (inst.pc.code_object_id == 0 && inst.pc.address == 0)
          continue;
        assert(inst.pc.code_object_id != 0);
        uint32_t idle;
        if (inst.time >= last_time) {
          idle = inst.time - last_time;
          assert(last_time + idle == inst.time);
        } else {
          idle = 0;
        }
        auto res = ctx.stats.add_inst(inst, idle);
        last_time = inst.time + inst.duration;
        assert(res);
      }
    }
  }
  return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

void dump_states(my_rocperf_tool::Disassembler &disas,
                 my_rocperf_tool::InstStatistics &stats) {
  for (const auto &[obj_id, obj] : disas.get_object_files()) {
    uint64_t text_sec_offset = obj.text_sec_offset;
    uint64_t text_sec_address = obj.text_sec_address;
    uint64_t text_sec_size = obj.text_sec_size;
    uint64_t load_base = obj.load_base;

    auto buffer = obj.memory_buffer->getBuffer();
    auto section_buffer =
        buffer.drop_front(text_sec_offset).take_front(text_sec_size);
    llvm::ArrayRef<unsigned char> bytes{section_buffer.bytes_begin(),
                                        section_buffer.size()};
    uint64_t virt_addr = load_base;
    uint64_t sec_offset = 0;
    while (!bytes.empty()) {
      llvm::MCInst inst;
      uint64_t inst_size;
      auto status = obj.mc_dis_asm->getInstruction(inst, inst_size, bytes,
                                                   virt_addr, llvm::outs());
      if (status == llvm::MCDisassembler::Success) {
        obj.streamer->emitInstruction(inst, *obj.sub_target);
        const_cast<std::string &>(obj.inst_str).clear();
      } else {
        assert(false);
      }
      uint64_t addr = sec_offset + text_sec_address;
      rocprofiler_thread_trace_decoder_pc_t pc{addr, obj_id};
      auto stats_range = stats.get_inst_at(pc);
      for (const auto &stat : stats_range) {
        assert(stat.stall <= stat.duration);
      }
      sec_offset += inst_size;
      virt_addr += inst_size;
      bytes = bytes.drop_front(inst_size);
    }
  }
}

int run_main(const std::string &att_output_dir_path) {
  my_rocperf_tool::AttOutputDir out_dir(att_output_dir_path);
  auto object_load_bases = out_dir.read_load_bases();
  auto [att_file_content, att_file_size] = out_dir.read_att_data();

  trace_decoder_context decoder_ctx{
      true,
      std::move(att_file_content),
      att_file_size,
  };

  for (const auto &obj_file : out_dir.code_objects) {
    decoder_ctx.disas.addCodeObject(obj_file.id, obj_file.path,
                                    object_load_bases[obj_file.id]);
    auto obj_cont = decoder_ctx.disas.get_object_file_by_id(obj_file.id)
                        .memory_buffer->getBuffer();
  }

  auto parse_res = rocprof_trace_decoder_parse_data(
      att_decoder_se_data_callback, att_decoder_trace_callback,
      att_decoder_isa_callback, &decoder_ctx);
  assert(parse_res == ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS);
  dump_states(decoder_ctx.disas, decoder_ctx.stats);
  return 0;
}

int main(int argc, char *argv[]) {
  absl::ParseCommandLine(argc, argv);
  std::optional<std::string> att_output_dir =
      absl::GetFlag(FLAGS_att_output_dir);
  if (!att_output_dir.has_value()) {
    std::cerr << "--att_output_dir not specified\n";
    return 2;
  }
  int LLVMArgc = 1;
  const char **LLVMArgv = const_cast<const char **>(argv);
  my_rocperf_tool::InitLLVM X(LLVMArgc, LLVMArgv);
  return run_main(*att_output_dir);
}
