#include "MLIRStatsPass.h"
#include "MLIRStatsAnalysis.h"
#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

void MLIRStatsPass::runOnOperation() {
  // Get the current operation being operated on.
  FuncOp op = getOperation();
  MLIRStatsAnalysis &analysis = getAnalysis<MLIRStatsAnalysis>();
  analysis.runAnalysis(op);
}
