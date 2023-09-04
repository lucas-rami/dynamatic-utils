#include "stencil_2d.h"
#include <stdlib.h>

int stencil_2d(in_int_t mem0[900], out_int_t mem1[900], in_int_t mem2[10]) {
  int temp = 0;
  for (unsigned c = 0; c < 28; c++) {
    temp = 0;
    for (unsigned k1 = 0; k1 < 3; k1++)
      for (unsigned k2 = 0; k2 < 3; k2++)
        temp += mem2[k1 * 3 + k2] * mem0[k1 * 30 + c + k2];
    mem1[c] = temp;
  }
  return temp;
}

#define AMOUNT_OF_TEST 1

int main(void) {
  in_int_t mem0[AMOUNT_OF_TEST][900];
  out_int_t mem1[AMOUNT_OF_TEST][900];
  in_int_t mem2[AMOUNT_OF_TEST][10];

  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    for (int j = 0; j < 900; ++j) {
      mem0[i][j] = rand() % 100;
    }
    for (int j = 0; j < 10; ++j) {
      mem2[i][j] = rand() % 100;
    }
  }

  // for(int i = 0; i < AMOUNT_OF_TEST; ++i){
  int i = 0;
  stencil_2d(mem0[i], mem1[i], mem2[i]);
  //}
}
