#include "fir.h"
//------------------------------------------------------------------------
// FIR
//------------------------------------------------------------------------

// SEPARATOR_FOR_MAIN

#include "fir.h"
#include <stdlib.h>

#define AMOUNT_OF_TEST 1

int fir(in_int_t mem0[1000], in_int_t mem1[1000]) {
  int tmp = 0;
  for (unsigned i = 0; i < 1000; i++) {
    tmp += mem1[i] * mem0[999 - i];
  }
  return tmp;
}

int main(void) {
  in_int_t mem0[AMOUNT_OF_TEST][1000];
  in_int_t mem1[AMOUNT_OF_TEST][1000];
  inout_int_t out[AMOUNT_OF_TEST][1000];

  srand(13);
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    for (int j = 0; j < 1000; ++j) {
      mem0[0][j] = rand() % 100;
      mem1[0][j] = rand() % 100;
    }
  }

  for (int i = 0; i < 1; ++i)
    fir(mem0[0], mem1[0]);
  return 0;
}

// SEPARATOR_FOR_MAIN
