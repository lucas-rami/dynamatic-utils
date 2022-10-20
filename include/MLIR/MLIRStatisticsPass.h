#ifndef _TOOLS_MLIR_STATISTICS_PASS_H_
#define _TOOLS_MLIR_STATISTICS_PASS_H_

#include "mlir/Pass/Pass.h"

using namespace mlir;

struct MLIRStatisticsPass
    : public PassWrapper<MLIRStatisticsPass, OperationPass<>> {

  struct Options : public PassPipelineOptions<Options> {
    Option<std::string> kernelName{*this, "kernel",
                                   llvm::cl::desc("Name of compute kernel")};
    Option<std::string> filename{
        *this, "filename", llvm::cl::desc("Name of file to write results to")};
  };

  MLIRStatisticsPass() = default;
  MLIRStatisticsPass(const MLIRStatisticsPass &) {}
  MLIRStatisticsPass(const Options &options) {
    kernelName = options.kernelName;
    filename = options.filename;
  }

  void runOnOperation() override;
  StringRef getArgument() const override { return "ir-stats"; }
  StringRef getDescription() const override {
    return "Detect use of global variables in compute kernels.";
  }

  Option<std::string> kernelName{*this, "kernel",
                                 llvm::cl::desc("Name of compute kernel")};
  Option<std::string> filename{
      *this, "filename", llvm::cl::desc("Name of file to write results to")};
};

#endif // _TOOLS_MLIR_STATISTICS_PASS_H_
