#include "IRStats.h"

bool print_stats(const BasicBlockStats &bb,
                 const llvm::Optional<InstructionStats> &instr) {
  if (!instr.hasValue()) {
    return false;
  }
  bb.print();
  std::cerr << std::endl;
  instr->print();
  return true;
}