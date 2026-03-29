#include <Python.h>
#include <stdlib.h>
#include "tropiq_core.h"

static PyObject *py_maxplus_matvec(PyObject *self, PyObject *args)
{
    PyObject *A_list, *x_list;
    int M, K;

    if (!PyArg_ParseTuple(args, "OOii", &A_list, &x_list, &M, &K)) {
        return NULL;
    }

    double *A = malloc(M * K * sizeof(double));
    double *x = malloc(K * sizeof(double));
    double *y = malloc(M * sizeof(double));

    for (int i = 0; i < M * K; i++) {
        A[i] = PyFloat_AsDouble(PyList_GetItem(A_list, i));
    }
    for (int i = 0; i < K; i++) {
        x[i] = PyFloat_AsDouble(PyList_GetItem(x_list, i));
    }

    maxplus_matvec_f64(A, x, y, M, K);

    PyObject *result = PyList_New(M);
    for (int i = 0; i < M; i++) {
        PyList_SetItem(result, i, PyFloat_FromDouble(y[i]));
    }

    free(A);
    free(x);
    free(y);

    return result;
}

/* --- Boilerplate below: do not edit --- */

static PyMethodDef TropiqMethods[] = {
    {"maxplus_matvec", py_maxplus_matvec, METH_VARARGS,
     "Max-plus matrix-vector product.\n"
     "Args: A (flat list, row-major), x (list), M (int), K (int)\n"
     "Returns: y (list of length M)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef tropiqmodule = {
    PyModuleDef_HEAD_INIT,
    "_core",
    "TropiQ C extension: tropical algebra kernels",
    -1,
    TropiqMethods
};

PyMODINIT_FUNC PyInit__core(void)
{
    return PyModule_Create(&tropiqmodule);
}
