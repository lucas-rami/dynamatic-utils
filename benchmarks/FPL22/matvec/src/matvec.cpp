#include "matvec.h"
//------------------------------------------------------------------------
// MatVec
//------------------------------------------------------------------------

#include "matvec.h"
#include <stdlib.h>

#define AMOUNT_OF_TEST 1

int matvec(in_int_t mem0[NM][NM], in_int_t mem1[NM], out_int_t mem2[NM]) {
  int i, j;
  int tmp = 0;

  for (i = 0; i < NM; i++) {
    tmp = 0;

    for (j = 0; j < NM; j++) {
      tmp += mem1[j] * mem0[i][j];
    }
    mem2[i] = tmp;
  }

  return tmp;
}

int main(void) {
  in_int_t mem0[AMOUNT_OF_TEST][NM][NM];
  in_int_t mem1[AMOUNT_OF_TEST][NM];
  out_int_t mem2[AMOUNT_OF_TEST][NM];

  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    for (int y = 0; y < NM; ++y) {
      mem1[i][y] = rand() % 100;
      for (int x = 0; x < NM; ++x) {
        mem0[i][y][x] = rand() % 100;
      }
    }
  }

  // for(int i = 0; i < AMOUNT_OF_TEST; ++i){
  int i = 0;
  matvec(mem0[i], mem1[i], mem2[i]);
  //}
}
