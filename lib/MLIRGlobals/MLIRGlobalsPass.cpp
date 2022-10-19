#include "MLIRGlobals/MLIRGlobalsPass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include <iostream>

using namespace std;
using namespace mlir;

void MLIRGlobalsPass::runOnOperation() {
  // Check that kernel name was specified
  if (kernelName.getValue().size() == 0) {
    cerr << "Kernel unspecified" << endl;
    return signalPassFailure();
  }

  // Get the current operation being operated on (module)
  Operation *module = getOperation();

  // Identify all global variables
  std::vector<LLVM::GlobalOp> globals{};
  module->walk([&](Operation *op) {
    if (isa<LLVM::GlobalOp>(op)) {
      globals.push_back(dyn_cast<LLVM::GlobalOp>(op));
    }
  });

  // Find the kernel function
  module->walk([&](Operation *op) {
    if (isa<FuncOp>(op)) {
      FuncOp fun = dyn_cast<FuncOp>(op);

      string fName = fun.getName().str();
      if (fName == kernelName) {
        // We have found the kernel function

        // Look for uses of any global variable in function body
        for (auto &g : globals) {
          auto symbolTable = g.getSymbolUses(fun);
          if (symbolTable.hasValue()) {
            std::vector<string> op_uses{};
            for (auto const &use : symbolTable.getValue()) {
              op_uses.push_back(use.getUser()->getName().getStringRef().str());
            }
          }
        }
      }
    }
  });
}
