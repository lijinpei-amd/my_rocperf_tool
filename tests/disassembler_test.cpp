#include "my_rocperf_tool/disassembler.h"

#include "gtest/gtest.h"

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/Error.h"

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace my_rocperf_tool {
namespace {

namespace fs = std::filesystem;

// Every SubTargetContext's MCContext points at its owner's MC options, so a
// relocated Disassembler would leave them dangling.
static_assert(!std::is_copy_constructible_v<Disassembler>);
static_assert(!std::is_copy_assignable_v<Disassembler>);
static_assert(!std::is_move_constructible_v<Disassembler>);
static_assert(!std::is_move_assignable_v<Disassembler>);

// Assembled from tests/code_objects/<gfx>.s at build time; the vaddrs below
// are the ones listed there.
std::string code_object_path(const std::string& gfx) {
  return std::string(TEST_CODE_OBJECT_DIR) + "/" + gfx + ".o";
}

// Loads the gfx942 object and prints its s_endpgm. Building the MC stack for
// that touches every registered piece of the target: target info, the MC
// layer (printer, code emitter, asm backend, streamer) and the disassembler.
std::string load_and_print_s_endpgm(Disassembler& disas) {
  if (auto err = disas.add_code_object(1, code_object_path("gfx942")))
    return llvm::toString(std::move(err));
  return disas.find_code_object(1)->text_at(0x18);
}

// gtest's main sets up no InitLLVM, so this is a Disassembler on its own.
TEST(DisassemblerTest, WorksWithoutProcessSetup) {
  Disassembler disas;
  EXPECT_EQ(load_and_print_s_endpgm(disas), "s_endpgm");
}

// CTest runs each case in a fresh process, so these two constructions are the
// process's first and race to register the target.
TEST(DisassemblerTest, ConstructsConcurrently) {
  std::atomic<bool> go{false};
  auto construct_and_print = [&](std::string& printed) {
    while (!go.load()) std::this_thread::yield();
    Disassembler disas;
    printed = load_and_print_s_endpgm(disas);
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

// Both objects in one Disassembler, as a run with two ISAs would load them.
class ObjectFileInfoTest : public ::testing::Test {
 protected:
  static constexpr uint64_t kGfx942Id = 1;
  static constexpr uint64_t kGfx1200Id = 2;

  void SetUp() override {
    for (auto [id, gfx] :
         {std::pair{kGfx942Id, "gfx942"}, std::pair{kGfx1200Id, "gfx1200"}}) {
      auto err = disas_.add_code_object(id, code_object_path(gfx));
      ASSERT_FALSE(static_cast<bool>(err)) << llvm::toString(std::move(err));
    }
    gfx942_ = disas_.find_code_object(kGfx942Id);
    gfx1200_ = disas_.find_code_object(kGfx1200Id);
    ASSERT_NE(gfx942_, nullptr);
    ASSERT_NE(gfx1200_, nullptr);
  }

  Disassembler disas_;
  const ObjectFileInfo* gfx942_ = nullptr;
  const ObjectFileInfo* gfx1200_ = nullptr;
};

TEST_F(ObjectFileInfoTest, AnswersSizeKindAndTextAtEachVaddr) {
  struct Inst {
    uint64_t vaddr;
    uint32_t size;
    InstKind kind;
    std::string text;
  };
  const Inst expected[] = {
      {0x00, 4, InstKind::Other, "s_nop 0"},
      {0x04, 8, InstKind::Mfma,
       "v_mfma_f32_32x32x8_f16 a[0:15], v[0:1], v[2:3], a[0:15]"},
      {0x0c, 4, InstKind::ClassicBarrier, "s_barrier"},
      {0x10, 8, InstKind::Other, "s_mov_b32 s0, 0x12345678"},
      {0x18, 4, InstKind::Other, "s_endpgm"},
  };
  for (const auto& e : expected) {
    SCOPED_TRACE(e.text);
    EXPECT_EQ(gfx942_->size_at(e.vaddr), e.size);
    EXPECT_EQ(gfx942_->kind_at(e.vaddr), e.kind);
    EXPECT_EQ(gfx942_->text_at(e.vaddr), e.text);
  }
}

// gfx12 split the barrier into a signal and a wait; neither is the classic
// s_barrier whose two trace tokens the CLI merges.
TEST_F(ObjectFileInfoTest, Gfx12SplitBarrierIsNotClassic) {
  EXPECT_EQ(gfx1200_->text_at(0x0), "s_barrier_signal -1");
  EXPECT_EQ(gfx1200_->kind_at(0x0), InstKind::Other);
  EXPECT_EQ(gfx1200_->kind_at(0x4), InstKind::Other);
  EXPECT_EQ(gfx1200_->kind_at(0x8), InstKind::Other);
}

TEST_F(ObjectFileInfoTest, CopiesTextOnlyIntoABufferItFits) {
  const std::string text = "s_mov_b32 s0, 0x12345678";
  std::vector<char> buf(text.size() - 1, '#');
  EXPECT_EQ(gfx942_->copy_text_at(0x10, buf), text.size());
  EXPECT_EQ(std::string(buf.begin(), buf.end()),
            std::string(text.size() - 1, '#'));

  buf.assign(text.size(), '#');
  EXPECT_EQ(gfx942_->copy_text_at(0x10, buf), text.size());
  EXPECT_EQ(std::string(buf.begin(), buf.end()), text);
}

TEST_F(ObjectFileInfoTest, WalksARangeByInstructionNotByDword) {
  EXPECT_EQ(gfx942_->inst_vaddrs(0x00, 0x18),
            (std::vector<uint64_t>{0x00, 0x04, 0x0c, 0x10, 0x18}));
  // An instruction counts if it starts in the range, even when `last` falls
  // inside it.
  EXPECT_EQ(gfx942_->inst_vaddrs(0x04, 0x14),
            (std::vector<uint64_t>{0x04, 0x0c, 0x10}));
  EXPECT_EQ(gfx942_->inst_vaddrs(0x0c, 0x0c), (std::vector<uint64_t>{0x0c}));
  EXPECT_EQ(gfx942_->inst_vaddrs(0x10, 0x04), (std::vector<uint64_t>{}));
}

TEST_F(ObjectFileInfoTest, WalksNothingOutsideTheTextSection) {
  // .text is 0x1c bytes long.
  EXPECT_EQ(gfx942_->inst_vaddrs(0x20, 0x40), (std::vector<uint64_t>{}));
  EXPECT_EQ(gfx942_->inst_vaddrs(0x10, 0x1c), (std::vector<uint64_t>{}));
}

TEST_F(ObjectFileInfoTest, UnknownIdFindsNothing) {
  EXPECT_EQ(disas_.find_code_object(0), nullptr);
  EXPECT_EQ(disas_.find_code_object(3), nullptr);
  // The ids come from trace data, so even values a hash map might reserve as
  // sentinels must be ordinary misses.
  EXPECT_EQ(disas_.find_code_object(std::numeric_limits<uint64_t>::max()),
            nullptr);
  EXPECT_EQ(disas_.find_code_object(std::numeric_limits<uint64_t>::max() - 1),
            nullptr);
}

// A linked code object places .text away from vaddr 0, and nothing stops it
// from having more than one. This one has two, at 0x1000 and 0x2000, with the
// higher one first in the file.
class TwoTextSectionsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto err = disas_.add_code_object(1, code_object_path("gfx942_two_texts"));
    ASSERT_FALSE(static_cast<bool>(err)) << llvm::toString(std::move(err));
    obj_ = disas_.find_code_object(1);
    ASSERT_NE(obj_, nullptr);
  }

  Disassembler disas_;
  const ObjectFileInfo* obj_ = nullptr;
};

TEST_F(TwoTextSectionsTest, AnswersAtVaddrsInEitherSection) {
  struct Inst {
    uint64_t vaddr;
    uint32_t size;
    InstKind kind;
    std::string text;
  };
  // Alternating sections, so each query moves to the other one.
  const Inst expected[] = {
      {0x1000, 4, InstKind::Other, "s_nop 0"},
      {0x2000, 8, InstKind::Mfma,
       "v_mfma_f32_32x32x8_f16 a[0:15], v[0:1], v[2:3], a[0:15]"},
      {0x1004, 8, InstKind::Other, "s_mov_b32 s0, 0x12345678"},
      {0x2008, 4, InstKind::ClassicBarrier, "s_barrier"},
      {0x100c, 4, InstKind::Other, "s_endpgm"},
      {0x200c, 4, InstKind::Other, "s_endpgm"},
  };
  for (const auto& e : expected) {
    SCOPED_TRACE(e.vaddr);
    EXPECT_EQ(obj_->size_at(e.vaddr), e.size);
    EXPECT_EQ(obj_->kind_at(e.vaddr), e.kind);
    EXPECT_EQ(obj_->text_at(e.vaddr), e.text);
  }
}

TEST_F(TwoTextSectionsTest, WalksWithinTheSectionHoldingTheRange) {
  EXPECT_EQ(obj_->inst_vaddrs(0x1000, 0x100c),
            (std::vector<uint64_t>{0x1000, 0x1004, 0x100c}));
  EXPECT_EQ(obj_->inst_vaddrs(0x2000, 0x200c),
            (std::vector<uint64_t>{0x2000, 0x2008, 0x200c}));
}

TEST_F(TwoTextSectionsTest, WalksNothingOutsideOrAcrossSections) {
  // Below the low section, partly or wholly.
  EXPECT_EQ(obj_->inst_vaddrs(0x0, 0xc), (std::vector<uint64_t>{}));
  EXPECT_EQ(obj_->inst_vaddrs(0xffc, 0x1004), (std::vector<uint64_t>{}));
  // In the gap between them, and from one into the other.
  EXPECT_EQ(obj_->inst_vaddrs(0x1010, 0x1ffc), (std::vector<uint64_t>{}));
  EXPECT_EQ(obj_->inst_vaddrs(0x1004, 0x2000), (std::vector<uint64_t>{}));
  // Past the high section.
  EXPECT_EQ(obj_->inst_vaddrs(0x2010, 0x2020), (std::vector<uint64_t>{}));
}

// Each test writes its inputs into a fresh temp directory.
class CodeObjectLoadTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::string dir = ::testing::TempDir() + "disassembler_test.XXXXXX";
    ASSERT_NE(mkdtemp(dir.data()), nullptr) << std::strerror(errno);
    root_ = dir;
  }

  void TearDown() override {
    if (!root_.empty()) fs::remove_all(root_);
  }

  std::string path(const std::string& rel) const {
    return (root_ / rel).string();
  }

  fs::path root_;
};

TEST_F(CodeObjectLoadTest, UnknownIsaFailsAtLoad) {
  std::ifstream in(code_object_path("gfx942"), std::ios::binary);
  std::vector<char> bytes{std::istreambuf_iterator<char>(in),
                          std::istreambuf_iterator<char>()};
  ASSERT_GT(bytes.size(), sizeof(llvm::ELF::Elf64_Ehdr));
  // Little-endian, so the low byte of e_flags holds EF_AMDGPU_MACH; NONE (0)
  // names no ISA.
  bytes[offsetof(llvm::ELF::Elf64_Ehdr, e_flags)] = 0;
  const std::string patched = path("no_isa.o");
  std::ofstream(patched, std::ios::binary).write(bytes.data(), bytes.size());

  Disassembler disas;
  auto err = disas.add_code_object(1, patched);
  ASSERT_TRUE(static_cast<bool>(err));
  EXPECT_EQ(llvm::toString(std::move(err)),
            patched + ": unknown AMDGPU ISA (EF_AMDGPU_MACH 0x0)");
  EXPECT_EQ(disas.find_code_object(1), nullptr);
}

TEST_F(CodeObjectLoadTest, UnreadableFileFailsAtLoad) {
  Disassembler disas;
  const std::string missing = path("missing.o");
  auto err = disas.add_code_object(1, missing);
  ASSERT_TRUE(static_cast<bool>(err));
  EXPECT_EQ(
      llvm::toString(std::move(err)),
      missing + ": " +
          std::make_error_code(std::errc::no_such_file_or_directory).message());
  EXPECT_EQ(disas.find_code_object(1), nullptr);
}

TEST_F(CodeObjectLoadTest, NonElfFileFailsAtLoad) {
  const std::string text = path("not_elf.o");
  std::ofstream(text) << "not an ELF file\n";

  Disassembler disas;
  auto err = disas.add_code_object(1, text);
  ASSERT_TRUE(static_cast<bool>(err));
  EXPECT_EQ(llvm::toString(std::move(err)).rfind(text + ": ", 0), 0u);
  EXPECT_EQ(disas.find_code_object(1), nullptr);
}

}  // namespace
}  // namespace my_rocperf_tool
