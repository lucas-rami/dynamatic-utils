/**
 * bicg.c: This file is part of the PolyBench/C 3.2 test suite.
 *
 *
 * Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://polybench.sourceforge.net
 */
#include "bicg.h"

#define NX 30
#define NY 30
#define N 30

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int bicg(in_int_t mem0[N][N], inout_int_t mem1[N], inout_int_t mem2[N],
         in_int_t mem3[N], in_int_t mem4[N]) {
  int tmp_q = 0;
  for (unsigned i = 0; i < NX; i++) {
    tmp_q = mem2[i];

    for (unsigned j = 0; j < NY; j++) {
      int tmp = mem0[i][j];
      mem1[j] = mem1[j] + mem4[i] * tmp;
      tmp_q = tmp_q + tmp * mem3[j];
    }

    mem2[i] = tmp_q;
  }
  return tmp_q;
}

#define AMOUNT_OF_TEST 1

int main(void) {
  in_int_t mem0[AMOUNT_OF_TEST][N][N];
  inout_int_t mem1[AMOUNT_OF_TEST][N];
  inout_int_t mem2[AMOUNT_OF_TEST][N];
  in_int_t mem3[AMOUNT_OF_TEST][N];
  in_int_t mem4[AMOUNT_OF_TEST][N];

  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    for (int y = 0; y < N; ++y) {
      mem1[i][y] = rand() % 100;
      mem2[i][y] = rand() % 100;
      mem3[i][y] = rand() % 100;
      mem4[i][y] = rand() % 100;
      for (int x = 0; x < N; ++x) {
        mem0[i][y][x] = rand() % 100;
      }
    }
  }

  // for(int i = 0; i < AMOUNT_OF_TEST; ++i){
  int i = 0;
  bicg(mem0[i], mem1[i], mem2[i], mem3[i], mem4[i]);
  //}
}
