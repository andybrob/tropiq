#ifndef TROPIQ_CORE_H
#define TROPIQ_CORE_H

/*
 * Max-plus matrix-vector product.
 * y[i] = max over k of (A[i,k] + x[k])
 *
 * A is stored row-major: element at row i, col k is A[i*K + k]
 * M: number of rows in A (= length of output y)
 * K: number of columns in A (= length of input x)
 */
void maxplus_matvec_f64(
    const double *A, const double *x, double *y,
    int M, int K
);

#endif /* TROPIQ_CORE_H */
