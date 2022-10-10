#ifndef _TOOL_LLVM_IR_STATS_H_
#define _TOOL_LLVM_IR_STATS_H_

#include "IRStats.h"
#include "llvm/Pass.h"

struct LLVMIRStats : public llvm::FunctionPass {
  static char ID;
  LLVMIRStats();
  bool runOnFunction(llvm::Function &f) override;

private:
  BasicBlockStats analyzeBasicBlocks(llvm::Function &f);
  llvm::Optional<InstructionStats> analyzeInstructions(llvm::Function &f);
};

#endif //_TOOL_LLVM_IR_STATS_H_
