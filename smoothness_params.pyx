# smoothness_params.pyx
# cython: boundscheck=False, wraparound=False, language_level=3

import numpy as np
cimport numpy as np
ctypedef np.double_t DTYPE_t

cpdef dict smoothness_metrics(np.ndarray[DTYPE_t, ndim=1] x,
                              double scale_pct):
    """
    Measures roughness on data *after* scaling by scale_pct.
    Returns first‐ & second‐diff sums, plus window/penalty suggestions.
    """
    cdef Py_ssize_t n = x.shape[0]
    if n < 3:
        raise ValueError("Need ≥3 points")

    cdef double fsum=0.0, ssum=0.0
    cdef double factor = scale_pct/100.0
    cdef Py_ssize_t i
    # first diff
    for i in range(n-1):
        fsum += (factor*(x[i+1] - x[i]))**2
    # second diff
    for i in range(n-2):
        ssum += (factor*(x[i+2] - 2*x[i+1] + x[i]))**2

    # window sizes scale‐invariant, but penalty weights grow ∝ scale²
    cdef list wins = [3,5,7,9]
    cdef list wts = [fsum/n, ssum/(n-1)]

    return {
      'first_diff_sq_sum': fsum,
      'second_diff_sq_sum': ssum,
      'suggested_windows': wins,
      'penalty_weights': wts
    }
