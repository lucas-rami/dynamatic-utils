#ifndef _TOOLS_MLIR_GLOBALS_PASS_H_
#define _TOOLS_MLIR_GLOBALS_PASS_H_

#include "mlir/Pass/Pass.h"

using namespace mlir;

struct MLIRGlobalsPass : public PassWrapper<MLIRGlobalsPass, OperationPass<>> {

  struct Options : public PassPipelineOptions<Options> {
    Option<std::string> kernelName{*this, "kernel",
                                   llvm::cl::desc("Name of compute kernel")};
  };

  MLIRGlobalsPass() = default;
  MLIRGlobalsPass(const MLIRGlobalsPass &) {}
  MLIRGlobalsPass(const Options &options) { kernelName = options.kernelName; }

  void runOnOperation() override;
  StringRef getArgument() const override { return "ir-globals"; }
  StringRef getDescription() const override {
    return "Detect use of global variables in compute kernels.";
  }

  Option<std::string> kernelName{*this, "kernel",
                                 llvm::cl::desc("Name of compute kernel")};
};

#endif // _TOOLS_MLIR_GLOBALS_PASS_H_
