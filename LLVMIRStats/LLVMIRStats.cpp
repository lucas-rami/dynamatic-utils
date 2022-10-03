#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <unordered_map>

using namespace llvm;

static const std::string MEMORY_OP{"memory"};
static const std::string ARITHMETIC_OP{"arithmetic"};
static const std::string LOGICAL_OP{"logic"};
static const std::string CONTROL_OP{"control"};
static const std::string CAST_OP{"cast"};

namespace {
struct LLVMIRStats : public FunctionPass {
  static char ID;
  LLVMIRStats() : FunctionPass(ID) {}

  bool runOnFunction(Function &f) override {

    if (f.getName() == "main") {
      return false;
    }

    analyzeBasicBlocks(f);
    analyzeInstructions(f);

    return false;
  }

private:
  bool analyzeBasicBlocks(Function &f) {
    // Count the number of basic blocks
    uint bbCnt{0};
    for (Function::iterator bb = f.begin(), bbEnd = f.end(); bb != bbEnd;
         ++bb) {
      bbCnt += 1;
    }

    // Print and return
    std::cerr << "[basic-block]" << std::endl;
    std::cerr << "count = " << bbCnt << std::endl << std::endl;
    return true;
  }

  bool analyzeInstructions(Function &f) {
    // Group instructions by type and count them
    std::unordered_map<std::string, std::set<std::string>> instrTypeToOpcodes{
        {MEMORY_OP, {"load", "store"}},
        {ARITHMETIC_OP,
         {"add", "sub", "icmp", "mul", "fmul", "fadd", "fcmp", "fsub", "fdiv",
          "sdiv", "urem", "shl"}},
        {LOGICAL_OP, {"and", "or", "xor", "select", "ashr"}},
        {CONTROL_OP, {"br", "ret"}},
        {CAST_OP, {"sext", "zext", "trunc", "sitofp"}},
    };
    std::unordered_map<std::string, int> instrTypeToCount{};
    std::set<std::string> unknownInstr{"phi", "getelementptr"};
    for (auto const &[instrType, _] : instrTypeToOpcodes) {
      instrTypeToCount[instrType] = 0;
    }

    // Iterate over all instructions in the function
    uint instrCnt{0};
    for (inst_iterator instr = inst_begin(f), instrEnd = inst_end(f);
         instr != instrEnd; ++instr) {
      instrCnt += 1;

      // Check which category the instruction belongs to
      std::string instrOpcode{instr->getOpcodeName()};
      bool foundType = false;
      for (auto const &[instrType, opcodes] : instrTypeToOpcodes) {
        if (opcodes.find(instrOpcode) != opcodes.end()) {
          instrTypeToCount[instrType] += 1;
          foundType = true;
          break;
        }
      }

      // Store and print all unknown instructions
      if (!foundType && unknownInstr.find(instrOpcode) == unknownInstr.end()) {
        unknownInstr.emplace(instrOpcode);
        std::cout << "Unknown instruction: " << instrOpcode << std::endl;
      }
    }

    // Print and return
    std::cerr << "[instruction]" << std::endl;
    std::cerr << "count = " << instrCnt << std::endl;
    for (auto const &[instrType, cnt] : instrTypeToCount) {
      std::cerr << "type." << instrType << " = " << cnt << std::endl;
    }
    return true;
  }
};
} // namespace

char LLVMIRStats::ID = 0;
static RegisterPass<LLVMIRStats> X("ir-stats", "Print stats about IR", false,
                                   false);
