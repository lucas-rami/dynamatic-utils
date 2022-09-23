#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/InstIterator.h"

using namespace llvm;

namespace {
  struct LLVMIRStats : public FunctionPass {
    static char ID;
    LLVMIRStats() : FunctionPass(ID) {}

    bool runOnFunction(Function &f) override {
      
      if (f.getName() == "main") {
        return false;
      }

      uint bbCnt {0};
      for(Function::iterator bb = f.begin(), bbEnd = f.end(); bb != bbEnd; ++bb) {
        bbCnt += 1;
      }
      errs() << "Number of basic blocks: " << bbCnt << "\n";
      
      uint instrCnt {0};
      for (inst_iterator instr = inst_begin(f), instrEnd = inst_end(f); instr != instrEnd; ++instr){
        instrCnt += 1;
      }
      errs() << "Number of instructions: " << instrCnt << "\n";    
      
      return false;
    }
  };
}

char LLVMIRStats::ID = 0;
static RegisterPass<LLVMIRStats> X("ir-stats", "Print stats about IR", false, false);
