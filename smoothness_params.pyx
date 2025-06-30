# smoothness_params.pyx
# cython: boundscheck=False, wraparound=False, language_level=3

cimport numpy as np
import numpy as np
ctypedef np.double_t DTYPE_t

cpdef dict smoothness_metrics(np.ndarray[DTYPE_t, ndim=1] x):
    """
    Returns {
      'first_diff_sq_sum': float,   # Σ (x[i+1]-x[i])^2
      'second_diff_sq_sum': float,  # Σ (x[i+2]-2x[i+1]+x[i])^2
      'suggested_windows': [int,…],
      'penalty_weights': [double,…]
    }
    """
    cdef Py_ssize_t n = x.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 points to measure smoothness")

    cdef double fsum=0.0, ssum=0.0
    cdef Py_ssize_t i
    # first differences
    for i in range(n-1):
        fsum += (x[i+1] - x[i])**2
    # second differences
    for i in range(n-2):
        ssum += (x[i+2] - 2*x[i+1] + x[i])**2

    # heuristics for smoothing window sizes
    cdef list wins = [3,5,7,9]
    # penalty weights ~ normalized to roughness magnitudes
    cdef list wts = [fsum/n, ssum/(n-1)]

    return {
      'first_diff_sq_sum': fsum,
      'second_diff_sq_sum': ssum,
      'suggested_windows': wins,
      'penalty_weights': wts
    }
