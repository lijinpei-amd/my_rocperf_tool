#include "my_rocperf_tool/disassembler.h"

#include "gtest/gtest.h"

#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>
#include <type_traits>

namespace my_rocperf_tool {
namespace {

// Every SubTargetContext's MCContext points at its owner's MC options, so a
// relocated Disassembler would leave them dangling.
static_assert(!std::is_copy_constructible_v<Disassembler>);
static_assert(!std::is_copy_assignable_v<Disassembler>);
static_assert(!std::is_move_constructible_v<Disassembler>);
static_assert(!std::is_move_assignable_v<Disassembler>);

// Round-trips s_endpgm through a gfx942 decoder and printer. Building that MC
// stack touches every registered piece of the target: target info, the MC
// layer (printer, code emitter, asm backend, streamer) and the disassembler.
std::string decode_and_print_s_endpgm(Disassembler& disas) {
  const auto& ctx = disas.get_sub_target_context("gfx942", "");
  const uint8_t s_endpgm[] = {0x00, 0x00, 0x81, 0xbf};
  llvm::MCInst inst;
  uint64_t size = 0;
  auto status = ctx.get_mc_disassembler().getInstruction(inst, size, s_endpgm,
                                                         0, llvm::nulls());
  if (status != llvm::MCDisassembler::Success || size != sizeof(s_endpgm))
    return "<undecodable>";
  return ctx.print_inst(inst).str();
}

// gtest's main sets up no InitLLVM, so this is a Disassembler on its own.
TEST(DisassemblerTest, WorksWithoutProcessSetup) {
  Disassembler disas;
  EXPECT_EQ(decode_and_print_s_endpgm(disas), "s_endpgm");
}

// CTest runs each case in a fresh process, so these two constructions are the
// process's first and race to register the target.
TEST(DisassemblerTest, ConstructsConcurrently) {
  std::atomic<bool> go{false};
  auto construct_and_print = [&](std::string& printed) {
    while (!go.load()) std::this_thread::yield();
    Disassembler disas;
    printed = decode_and_print_s_endpgm(disas);
  };
  std::string printed[2];
  std::thread first(construct_and_print, std::ref(printed[0]));
  std::thread second(construct_and_print, std::ref(printed[1]));
  go = true;
  first.join();
  second.join();
  EXPECT_EQ(printed[0], "s_endpgm");
  EXPECT_EQ(printed[1], "s_endpgm");
}

}  // namespace
}  // namespace my_rocperf_tool
