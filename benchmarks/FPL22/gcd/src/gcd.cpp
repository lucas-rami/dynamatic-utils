
#include "gcd.h"
#include <stdlib.h>

#define AMOUNT_OF_TEST 1

// https://www.geeksforgeeks.org/steins-algorithm-for-finding-gcd/

int gcd(in_int_t in0, in_int_t in1) {
  // GCD(0, in1) == in1; GCD(in0, 0) == in0, GCD(0, 0) == 0
  // if (in0 == 0)
  //     return in1;
  // if (in1 == 0)
  //    return in0;

  // Finding K, where K is the greatest power of 2 that divides both in0 and
  // in1. for (int k = 0; ((in0 | in1) & 1) == 0; ++k)
  int k = 0;
  while (((in0 | in1) & 1) == 0) {
    in0 >>= 1;
    in1 >>= 1;
    k++;
  }

  // Dividing in0 by 2 until in0 becomes odd
  while ((in0 & 1) == 0)
    in0 >>= 1;

  // From here on, 'in0' is always odd.
  // do
  //{
  // If in1 is even, remove all factor of 2 in in1
  while ((in1 & 1) == 0)
    in1 >>= 1;

  // Now in0 and in1 are both odd. Swap if necessary so in0 <= in1, then set in1
  // = in1 - in0 (which is even).
  while (in1 > 0 && ((in1 & 1) == 0)) {
    in1 = in1 - in0;
  }

  //}while (in1 != 0);

  // restore common factors of 2
  return in0 << k;
}

int main(void) {
  gcd(7966496, 314080416);
  // gcd(24826148, 45296490);
}
