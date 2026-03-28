#include <float.h>      /* DBL_MAX */
#include "tropiq_core.h"

void maxplus_matvec_f64(const double *A, const double *x, double *y, int M, int K)
{
    for (int i = 0; i < M; i++) {
        double maximum = -DBL_MAX;
        for (int k = 0; k < K; k++) {
            if (A[i * K + k] + x[k] > maximum) {
                maximum = A[i * K + k] + x[k];
            }
        }
        y[i] = maximum;
    }
}