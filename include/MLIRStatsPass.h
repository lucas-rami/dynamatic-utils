#ifndef TOOLS_MLIR_STATS_PASS_H
#define TOOLS_MLIR_STATS_PASS_H

#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

struct MLIRStatsPass
    : public PassWrapper<MLIRStatsPass, OperationPass<FuncOp>> {

  void runOnOperation() override;
  StringRef getArgument() const override { return "ir-stats"; }
  StringRef getDescription() const override {
    return "Some statistics on the IR.";
  }
};

#endif // TOOLS_MLIR_STATS_PASS_H
