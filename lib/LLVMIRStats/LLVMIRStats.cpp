#include "LLVMIRStats/LLVMIRStats.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <unordered_map>
using namespace llvm;
using namespace std;

LLVMIRStats::LLVMIRStats() : FunctionPass(ID) {}

bool LLVMIRStats::runOnFunction(Function &f) {
  if (f.getName() == "main") {
    return false;
  }

  auto bb = analyzeBasicBlocks(f);
  auto instr = analyzeInstructions(f);
  print_stats(bb, instr);
  return false;
}

BasicBlockStats LLVMIRStats::analyzeBasicBlocks(Function &f) {
  // Count the number of basic blocks
  uint bbCount{0};
  std::vector<uint> predCounts{};
  std::vector<uint> succCounts{};

  for (Function::iterator bbIter = f.begin(), bbEnd = f.end(); bbIter != bbEnd;
       ++bbIter) {
    bbCount += 1;

    BasicBlock *bb = dyn_cast<BasicBlock>(bbIter);

    // Count number of predecessors
    uint n_preds{0};
    for (BasicBlock *p : predecessors(bb)) {
      n_preds += 1;
    }
    predCounts.push_back(n_preds);

    // Count number of successors
    uint n_succs{0};
    for (BasicBlock *p : successors(bb)) {
      n_succs += 1;
    }
    succCounts.push_back(n_succs);
  }
  return BasicBlockStats{bbCount, predCounts, succCounts};
}

Optional<InstructionStats> LLVMIRStats::analyzeInstructions(Function &f) {
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
    for (auto const &typeAndOpcodes : instrTypeToOpcodes) {
      if (typeAndOpcodes.second.find(instrOpcode) !=
          typeAndOpcodes.second.end()) {
        instrTypeToCount[typeAndOpcodes.first] += 1;
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

char LLVMIRStats::ID = 0;
static RegisterPass<LLVMIRStats> X("ir-stats", "Print stats about IR", false,
                                   false);
