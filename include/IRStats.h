#ifndef _TOOLS_IR_STATS_H_
#define _TOOLS_IR_STATS_H_

#include "json.hpp"
#include "llvm/ADT/Optional.h"
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

// Types of instructions
const string MEMORY_OP{"memory"};
const string ARITHMETIC_OP{"arithmetic"};
const string LOGICAL_OP{"logic"};
const string CONTROL_OP{"control"};
const string CAST_OP{"cast"};

const string ALL_TYPES[]{MEMORY_OP, ARITHMETIC_OP, LOGICAL_OP, CONTROL_OP,
                         CAST_OP};

// Information on basic blocks
struct BasicBlockStats {
  uint count;
  std::vector<uint> predCounts;
  std::vector<uint> succCounts;

  inline json to_json() const {
    return {{"count", count},
            {"predCounts", predCounts},
            {"succCounts", succCounts}};
  }
};

// Information on instructions
struct InstructionStats {
  uint count;
  unordered_map<string, int> typeToCount;

  inline json to_json() const {
    json data{{"count", count}};
    data["type"] = json::parse("{}");
    for (auto const &typeAndCount : typeToCount) {
      data["type"][typeAndCount.first] = typeAndCount.second;
    }
    return data;
  }
};

struct GlobalStats {
  std::vector<string> names;
  unordered_map<string, std::vector<string>> uses;

  inline json to_json() const {
    json data{{"names", names}};
    data["uses"] = json::parse("{}");
    for (auto const &nameAndUses : uses) {
      data["uses"][nameAndUses.first] = nameAndUses.second;
    }
    return data;
  }
};

struct IRStats {
  BasicBlockStats bb;
  InstructionStats instr;
  GlobalStats global;

  void dump(const string &filename);
};

#endif //_TOOLS_IR_STATS_H_
