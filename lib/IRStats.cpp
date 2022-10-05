#include "IRStats.h"

bool print_stats(const BasicBlockStats &bb,
                 const llvm::Optional<InstructionStats> &instr) {
  if (!instr.hasValue()) {
    return false;
  }
  json blocks = bb.to_json();
  json instructions = instr->to_json();
  json stats = {
      {"basic-blocks", blocks},
      {"instructions", instructions},
  };
  std::cerr << stats.dump() << std::endl;
  return true;
}