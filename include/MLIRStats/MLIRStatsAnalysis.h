#ifndef _TOOLS_MLIR_STATS_ANALYSIS_H_
#define _TOOLS_MLIR_STATS_ANALYSIS_H_

#include "IRStats.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

struct MLIRStatsAnalysis {

  MLIRStatsAnalysis(Operation *op);

  void runAnalysis(FuncOp op);

private:
  BasicBlockStats analyzeBasicBlocks(FuncOp op);
  llvm::Optional<InstructionStats> analyzeInstrutions(FuncOp op);
};

#endif //_TOOLS_MLIR_STATS_ANALYSIS_H_
