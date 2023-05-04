#include "polynMult.h"
//------------------------------------------------------------------------
// polynMult
//------------------------------------------------------------------------

// SEPARATOR_FOR_MAIN

#include <stdlib.h>

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define AMOUNT_OF_TEST 1

int polynMult(out_int_t mem0[NTRU_N], in_int_t mem1[NTRU_N],
              in_int_t mem2[NTRU_N]) {
  int k, i, p = 0;

  for (k = 0; k < NTRU_N; k++) {
    mem0[k] = 0;
    for (i = 1; i < NTRU_N - k; i++)
      mem0[k] += mem1[k + i] * mem2[NTRU_N - i];
    for (i = 0; i < k + 1; i++)
      mem0[k] += mem1[k - i] * mem2[i];

    p = i + k;
  }

  return p;
}

#define AMOUNT_OF_TEST 1

int main(void) {
  out_int_t mem0[NTRU_N];
  in_int_t mem1[NTRU_N];
  in_int_t mem2[NTRU_N];

  for (int i = 0; i < NTRU_N; i++) {
    mem0[i] = 0;
    mem1[i] = i % 10;
    mem2[i] = (NTRU_N - i) % 10;
  }

  polynMult(mem0, mem1, mem2);
}
