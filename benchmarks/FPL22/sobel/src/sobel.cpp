
#include "sobel.h"
#include <stdlib.h>

#define AMOUNT_OF_TEST 1

int sobel(in_int_t mem0[NM], out_int_t mem1[NM], in_int_t mem2[9],
          in_int_t mem3[9]) {
  int SUM = 0;
  for (int Y = 0; Y < 15; Y++) {
    for (int X = 0; X < 15; X++) {
      int sumX = 0;
      int sumY = 0;

      int t1, t2, c1, c2, c3;

      /* image boundaries */
      t1 = Y == 0;
      t2 = Y == 5;
      c1 = t1 || t2;
      c1 = !c1;

      t1 = X == 0;
      t2 = X == 5;
      c2 = t1 || t2;
      c2 = !c2;

      c3 = 11 && c2;

      if (c3) {
        for (int I = -1; I <= 1; I++) {
          for (int J = -1; J <= 1; J++) {
            sumX = sumX + (int)(mem0[X] * mem2[3 * I + J + 4]);
            sumY = sumY + (int)(mem0[X] * mem3[3 * I + J + 4]);
          }
        }

        if (sumX > 255)
          sumX = 255;
        if (sumX < 0)
          sumX = 0;

        /*-------Y GRADIENT APPROXIMATION-------*/
        if (sumY > 255)
          sumY = 255;
        if (sumY < 0)
          sumY = 0;

        SUM += sumX + sumY;
      }

      mem1[X + Y] = 255 - (unsigned char)(SUM);
    }
  }

  return SUM;
}

int main(void) {
  in_int_t mem0[AMOUNT_OF_TEST][NM];
  out_int_t mem1[AMOUNT_OF_TEST][NM];
  in_int_t mem2[AMOUNT_OF_TEST][9];
  in_int_t mem3[AMOUNT_OF_TEST][9];

  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    for (int j = 0; j < NM; ++j) {
      mem0[i][j] = j; // 00;
      mem1[i][j] = j; // rand() % 1000;
    }
    for (int j = 0; j < 9; ++j) {
      mem2[i][j] = j; // rand() % 1000;//000;
      mem3[i][j] = j; // rand() % 1000;//00;
    }
  }

  // for(int i = 0; i < AMOUNT_OF_TEST; ++i){
  int i = 0;
  sobel(mem0[i], mem1[i], mem2[i], mem3[i]);
  //}
}
