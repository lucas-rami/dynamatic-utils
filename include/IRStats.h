#ifndef _TOOLS_IR_STATS_H_
#define _TOOLS_IR_STATS_H_

#include "llvm/ADT/Optional.h"
#include <iostream>
#include <string>
#include <unordered_map>

// Types of instructions
const std::string MEMORY_OP{"memory"};
const std::string ARITHMETIC_OP{"arithmetic"};
const std::string LOGICAL_OP{"logic"};
const std::string CONTROL_OP{"control"};
const std::string CAST_OP{"cast"};

const std::string ALL_TYPES[]{MEMORY_OP, ARITHMETIC_OP, LOGICAL_OP, CONTROL_OP,
                              CAST_OP};

// Information on basic blocks
struct BasicBlockStats {
  uint count;

  void print() const {
    std::cerr << "[basic-block]" << std::endl;
    std::cerr << "count = " << count << std::endl;
  }
};

// Information on instructions
struct InstructionStats {
  uint count;
  std::unordered_map<std::string, int> typeToCount;

  void print() const {
    std::cerr << "[instruction]" << std::endl;
    std::cerr << "count = " << count << std::endl;
    for (auto const &typeAndCount : typeToCount) {
      std::cerr << "type." << typeAndCount.first << " = " << typeAndCount.second
                << std::endl;
    }
  }
};

bool print_stats(const BasicBlockStats &bb,
                 const llvm::Optional<InstructionStats> &instr);

#endif //_TOOLS_IR_STATS_H_
