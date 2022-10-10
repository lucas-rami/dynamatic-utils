#include "MLIRStats/MLIRStatsAnalysis.h"
#include "mlir/IR/BlockSupport.h"
#include "llvm/ADT/APInt.h"
#include <set>

using namespace std;

MLIRStatsAnalysis::MLIRStatsAnalysis(Operation *op) {}

void MLIRStatsAnalysis::runAnalysis(FuncOp op) {
  auto bb = analyzeBasicBlocks(op);
  auto instr = analyzeInstrutions(op);
  print_stats(bb, instr);
}

BasicBlockStats MLIRStatsAnalysis::analyzeBasicBlocks(FuncOp op) {
  std::vector<uint> predCounts{};
  std::vector<uint> succCounts{};

  auto blocks = &op.getBlocks();
  for (auto bbIter = blocks->begin(), endBBIter = blocks->end();
       bbIter != endBBIter; ++bbIter) {
    Block &bb = *bbIter;

    // Count number of predecessors
    uint n_preds{0};
    for (auto pred : bb.getPredecessors()) {
      n_preds += 1;
    }
    predCounts.push_back(n_preds);

    // Count number of successors
    uint n_succs{0};
    for (auto succ : bb.getSuccessors()) {
      n_succs += 1;
    }
    succCounts.push_back(n_succs);
  }
  return BasicBlockStats{static_cast<uint>((&op.getBlocks())->size()),
                         predCounts, succCounts};
}

llvm::Optional<InstructionStats>
MLIRStatsAnalysis::analyzeInstrutions(FuncOp op) {
  // Group instructions by type and count them
  unordered_map<string, set<string>> instrTypeToOpcodes{
      {MEMORY_OP, {"memref.load", "memref.store"}},
      {ARITHMETIC_OP,
       {"arith.addi", "arith.cmpi", "arith.muli", "arith.mulf", "arith.addf",
        "arith.cmpf", "arith.subf", "arith.divf", "arith.subi", "arith.divsi",
        "arith.subf", "arith.remsi"}},
      {LOGICAL_OP, {"arith.select", "arith.shrsi"}},
      {CONTROL_OP, {"cf.br", "cf.cond_br", "func.return"}},
      {CAST_OP,
       {"arith.index_cast", "arith.truncf", "arith.trunci", "arith.extsi",
        "arith.extui", "arith.sitofp"}},
  };
  set<string> unknownInstr{"arith.constant"};
  unordered_map<string, int> instrTypeToCount{};
  for (auto const &instrType : ALL_TYPES) {
    instrTypeToCount[instrType] = 0;
  }

  uint n_ops{0};
  auto blocks = &op.getBlocks();
  for (auto bbIter = blocks->begin(), endBBIter = blocks->end();
       bbIter != endBBIter; ++bbIter) {
    Block &bb = *bbIter;
    auto operations = &bb.getOperations();
    for (auto opIter = operations->begin(), endOPIter = operations->end();
         opIter != endOPIter; ++opIter) {
      Operation &op = *opIter;
      n_ops++;

      auto dialect = op.getDialect()->getNamespace().str();
      auto opName = op.getName().getStringRef().str();

      bool foundType = false;
      for (auto const &typeAndOpcodes : instrTypeToOpcodes) {
        if (typeAndOpcodes.second.find(opName) != typeAndOpcodes.second.end()) {
          instrTypeToCount[typeAndOpcodes.first] += 1;
          foundType = true;
          break;
        }
      }

      // Store and print all unknown instructions
      // Stop on unknown instruction
      if (!foundType && unknownInstr.find(opName) == unknownInstr.end()) {
        unknownInstr.emplace(opName);
        cout << "Unknown instruction " << opName << endl;
        return {};
      }
    }
  }

  return InstructionStats{n_ops, instrTypeToCount};
}