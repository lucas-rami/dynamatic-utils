typedef int in_int_t;
typedef int out_int_t;

#define NM 256

int sobel(in_int_t mem0[NM], out_int_t mem1[NM], in_int_t mem2[9],
          in_int_t mem3[9]);
