#include "MLIRStatsPass.h"
#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include <iostream>

using namespace mlir;

void MLIRStatsPass::runOnOperation() {
  // Get the current operation being operated on.
  Operation *op = getOperation();
  std::cout << "merde" << std::endl;
}
