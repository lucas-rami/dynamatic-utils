#include "MLIRStatsAnalysis.h"
#include <iostream>

MLIRStatsAnalysis::MLIRStatsAnalysis(Operation *op) {}

void MLIRStatsAnalysis::runAnalysis(FuncOp op) {
  auto blocks = &op.getBlocks();
  std::cerr << blocks->size() << std::endl;

  uint n_ops{0};
  for (auto bbIter = blocks->begin(), endBBIter = blocks->end();
       bbIter != endBBIter; ++bbIter) {
    Block &bb = *bbIter;
    auto operations = &bb.getOperations();
    for (auto opIter = operations->begin(), endOPIter = operations->end();
         opIter != endOPIter; ++opIter) {
      Operation &op = *opIter;
      n_ops++;
    }
  }

  std::cerr << n_ops << std::endl;
}