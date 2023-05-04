#pragma once

#include <stddef.h>
#include <stdint.h>

#define NTRU_N 100

typedef uint32_t out_int_t;
typedef uint32_t in_int_t;

int polynMult(out_int_t mem0[NTRU_N], in_int_t mem1[NTRU_N],
              in_int_t mem2[NTRU_N]);
