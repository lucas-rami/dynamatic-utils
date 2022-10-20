#include <iostream>

#include "IRStats.h"
#include "MLIR/MLIRStatisticsPass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace std;
using namespace mlir;

static BasicBlockStats analyzeBasicBlocks(FuncOp op) {
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

static llvm::Optional<InstructionStats> analyzeInstrutions(FuncOp op) {

  // Group instructions by type and count them
  unordered_map<string, set<string>> instrTypeToOpcodes{
      {MEMORY_OP, {"memref.load", "memref.store"}},
      {ARITHMETIC_OP,
       {"arith.addi", "arith.cmpi", "arith.muli", "arith.mulf", "arith.addf",
        "arith.cmpf", "arith.subf", "arith.divf", "arith.subi", "arith.divsi",
        "arith.subf", "arith.remsi", "math.sqrt", "arith.divui"}},
      {LOGICAL_OP, {"arith.select", "arith.shrsi", "arith.andi", "arith.negf"}},
      {CONTROL_OP, {"cf.br", "cf.cond_br", "func.return"}},
      {CAST_OP,
       {"arith.index_cast", "arith.truncf", "arith.trunci", "arith.extsi",
        "arith.extui", "arith.sitofp"}},
  };
  set<string> unknownInstr{"arith.constant", "memref.alloca", "llvm.mlir.undef",
                           "func.call"};
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
        cerr << "Unknown instruction " << opName << endl;
        return {};
      }
    }
  }

  return InstructionStats{n_ops, instrTypeToCount};
}

static GlobalStats analyzeGlobals(FuncOp op,
                                  const std::vector<LLVM::GlobalOp> &globals) {

  unordered_map<string, std::vector<string>> uses{};

  // Look for uses of any global variable in function body
  for (auto g : globals) {
    auto symbolTable = g.getSymbolUses(op);
    if (symbolTable.hasValue()) {
      std::vector<string> globalUses{};
      for (auto const &use : symbolTable.getValue()) {
        globalUses.push_back(use.getUser()->getName().getStringRef().str());
      }
      if (globalUses.size() != 0) {
        uses[g.getName().str()] = globalUses;
      }
    }
  }

  std::vector<string> globalNames{};
  for (auto g : globals) {
    globalNames.push_back(g.getName().str());
  }
  return GlobalStats{globalNames, uses};
}

static bool analyze(FuncOp fun, const std::vector<LLVM::GlobalOp> &globals,
                    const string &filename) {
  auto bb = analyzeBasicBlocks(fun);
  auto instr = analyzeInstrutions(fun);
  if (!instr.hasValue()) {
    return false;
  }
  auto global = analyzeGlobals(fun, globals);
  IRStats stats{bb, instr.getValue(), global};

  stats.dump(filename);
  return true;
}

void MLIRStatisticsPass::runOnOperation() {
  // Get the current operation being operated on (module)
  Operation *module = getOperation();

  if (kernelName.getValue().size() == 0) {
    cerr << "Kernel unspecified" << endl;
    return signalPassFailure();
  }

  // Identify all global variables
  std::vector<LLVM::GlobalOp> globals{};
  module->walk([&](Operation *op) {
    if (isa<LLVM::GlobalOp>(op)) {
      globals.push_back(dyn_cast<LLVM::GlobalOp>(op));
    }
  });

  // Find the kernel function
  bool failure{false}, found{false};
  module->walk([&](Operation *op) {
    if (isa<FuncOp>(op)) {
      FuncOp fun = dyn_cast<FuncOp>(op);

      string fName = fun.getName().str();
      if (fName == kernelName) {
        if (!analyze(fun, globals, filename)) {
          failure = true;
          found = true;
        }
      }
    }
  });

  if (failure || found) {
    return signalPassFailure();
  }
}
