#ifndef _TOOLS_IR_STATS_H_
#define _TOOLS_IR_STATS_H_

#include "json.hpp"
#include "llvm/ADT/Optional.h"
#include <iostream>
#include <string>
#include <unordered_map>

using json = nlohmann::json;

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
  std::vector<uint> predCounts;
  std::vector<uint> succCounts;

  json to_json() const {
    return {{"count", count},
            {"predCounts", predCounts},
            {"succCounts", succCounts}};
  }
};

// Information on instructions
struct InstructionStats {
  uint count;
  std::unordered_map<std::string, int> typeToCount;

  json to_json() const {
    json data{{"count", count}};
    for (auto const &typeAndCount : typeToCount) {
      data["type"][typeAndCount.first] = typeAndCount.second;
    }
    return data;
  }
};

bool print_stats(const BasicBlockStats &bb,
                 const llvm::Optional<InstructionStats> &instr);

#endif //_TOOLS_IR_STATS_H_
