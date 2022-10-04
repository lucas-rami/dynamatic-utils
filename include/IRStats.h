#ifndef _TOOLS_IR_STATS_H_
#define _TOOLS_IR_STATS_H_

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

  void print() {
    std::cerr << "[basic-block]" << std::endl;
    std::cerr << "count = " << count << std::endl;
  }
};

// Information on instructions
struct InstructionStats {
  uint count;
  std::unordered_map<std::string, int> typeToCount;

  void print() {
    std::cerr << "[instruction]" << std::endl;
    std::cerr << "count = " << count << std::endl;
    for (auto const &[instrType, instrCount] : typeToCount) {
      std::cerr << "type." << instrType << " = " << instrCount << std::endl;
    }
  }
};

bool print_stats(BasicBlockStats &bb, std::optional<InstructionStats> &instr) {
  if (!instr) {
    return false;
  }
  bb.print();
  std::cerr << std::endl;
  instr->print();
  return true;
}

#endif //_TOOLS_IR_STATS_H_
