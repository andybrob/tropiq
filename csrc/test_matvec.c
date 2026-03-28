#include <stdio.h>     /* printf */
#include <stdlib.h>    /* malloc, free */
#include <float.h>     /* DBL_MAX */
#include "tropiq_core.h"

int main() {
    //double A[6] = {1, 4, 2, 5, 1, 0};
    //double x[3] = {0,1,3};
    //double y[2] = {5, 5};
    double *A = malloc(6*sizeof(double));
    double *x = malloc(3*sizeof(double));
    double *y = malloc(2*sizeof(double));
    A[0] = 1; A[1] = 4; A[2] = 2; A[3] = 5; A[4] = 1; A[5] = 2;
    x[0] = 0; x[1] = 1; x[2] = 3;
    maxplus_matvec_f64(A, x, y, 2, 3);
    printf("y = [%f, %f]\n", y[0], y[1]);
    free(A);
    free(x);
    free(y);
    return 0;
}