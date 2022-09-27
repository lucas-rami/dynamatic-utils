#ifndef _TOOLS_MLIR_STATS_ANALYSIS_H
#define _TOOLS_MLIR_STATS_ANALYSIS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

struct MLIRStatsAnalysis {

  MLIRStatsAnalysis(Operation *op);

  void runAnalysis(func::FuncOp op);
};

#endif //_TOOLS_MLIR_STATS_ANALYSIS_H
