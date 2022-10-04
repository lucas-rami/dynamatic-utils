#include "IRStats.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <unordered_map>

using namespace llvm;
using namespace std;

namespace {

struct LLVMIRStats : public FunctionPass {
  static char ID;
  LLVMIRStats() : FunctionPass(ID) {}

  bool runOnFunction(Function &f) override {

    if (f.getName() == "main") {
      return false;
    }

    auto bb = analyzeBasicBlocks(f);
    auto instr = analyzeInstructions(f);
    print_stats(bb, instr);
    return false;
  }

private:
  BasicBlockStats analyzeBasicBlocks(Function &f) {
    // Count the number of basic blocks
    uint bbCount{0};
    for (Function::iterator bb = f.begin(), bbEnd = f.end(); bb != bbEnd;
         ++bb) {
      bbCount += 1;
    }
    return BasicBlockStats{bbCount};
  }

  optional<InstructionStats> analyzeInstructions(Function &f) {
    // Group instructions by type and count them
    unordered_map<string, set<string>> instrTypeToOpcodes{
        {MEMORY_OP, {"load", "store"}},
        {ARITHMETIC_OP,
         {"add", "sub", "icmp", "mul", "fmul", "fadd", "fcmp", "fsub", "fdiv",
          "sdiv", "urem", "shl"}},
        {LOGICAL_OP, {"and", "or", "xor", "select", "ashr"}},
        {CONTROL_OP, {"br", "ret"}},
        {CAST_OP, {"sext", "zext", "trunc", "sitofp"}},
    };
    set<string> unknownInstr{"phi", "getelementptr"};
    unordered_map<string, int> instrTypeToCount{};
    for (auto const &instrType : ALL_TYPES) {
      instrTypeToCount[instrType] = 0;
    }

    // Iterate over all instructions in the function
    uint instrCount{0};
    for (inst_iterator instr = inst_begin(f), instrEnd = inst_end(f);
         instr != instrEnd; ++instr) {
      instrCount += 1;

      // Check which category the instruction belongs to
      string instrOpcode{instr->getOpcodeName()};
      bool foundType = false;
      for (auto const &[instrType, opcodes] : instrTypeToOpcodes) {
        if (opcodes.find(instrOpcode) != opcodes.end()) {
          instrTypeToCount[instrType] += 1;
          foundType = true;
          break;
        }
      }

      // Stop on unknown instruction
      if (!foundType && unknownInstr.find(instrOpcode) == unknownInstr.end()) {
        cout << "Unknown instruction: " << instrOpcode << endl;
        return {};
      }
    }

    return InstructionStats{instrCount, instrTypeToCount};
  }
};
} // namespace

char LLVMIRStats::ID = 0;
static RegisterPass<LLVMIRStats> X("ir-stats", "Print stats about IR", false,
                                   false);
