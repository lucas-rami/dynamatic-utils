#include <fstream>

#include "IRStats.h"

void IRStats::dump(const string &filename) {
  json stats = {
      {"basic-blocks", bb.to_json()},
      {"instructions", instr.to_json()},
      {"globals", global.to_json()},
  };

  ofstream outputFile{filename};
  outputFile << stats.dump() << endl;
}