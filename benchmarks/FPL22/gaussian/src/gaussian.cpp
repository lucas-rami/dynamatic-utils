
#include "gaussian.h"
#include <stdlib.h>

#define AMOUNT_OF_TEST 1

int gaussian(in_int_t mem0[20], inout_int_t mem1[20][20]) {
  unsigned sum = 0;
  for (unsigned j = 1; j <= 18; j++) {
    for (unsigned i = j + 1; i <= 18; i++) {
      for (unsigned k = 1; k <= 19; k++) {
        mem1[i][k] = mem1[i][k] - mem0[j] * mem1[j][k];
        sum += k;
      }
    }
  }
  return sum;
}

int main(void) {
  in_int_t mem0[AMOUNT_OF_TEST][20];
  in_int_t mem1[AMOUNT_OF_TEST][20][20];

  srand(13);
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    for (int y = 0; y < 20; ++y) {
      mem0[i][y] = 1; // rand()%20;
      for (int x = 0; x < 20; ++x) {
        mem1[i][y][x] = 1; // rand()%20;
      }
    }
  }

  // for(int i = 0; i < AMOUNT_OF_TEST; ++i){
  int i = 0;
  gaussian(mem0[i], mem1[i]);
  //}
}
