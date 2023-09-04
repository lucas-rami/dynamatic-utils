#include "binary_search.h"
//------------------------------------------------------------------------
// binary_search
//------------------------------------------------------------------------

// SEPARATOR_FOR_MAIN

#include <stdlib.h>

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define AMOUNT_OF_TEST 1

int binary_search(in_int_t in0, in_int_t mem0[N]) {
  int idx_1 = -1;
  int idx_2 = -1;

  for (unsigned i = 0; i < N; i += 2) {
    if (mem0[i] == in0) {
      idx_1 = (int)i;
      break;
    }
  }

  for (unsigned i = 1; i < N; i += 2) {
    if (mem0[i] == in0) {
      idx_2 = (int)i;
      break;
    }
  }

  int done = -1;
  if (idx_1 != -1)
    done = idx_1;
  else if (idx_2 != -1)
    done = idx_2;

  return done;
}

int main(void) {
  in_int_t mem0[N];

  for (int i = 0; i < N; i++) {
    mem0[i] = i;
  }

  binary_search(55, mem0);
}
