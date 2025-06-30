# other_params.pyx
# cython: boundscheck=False, wraparound=False, language_level=3

import numpy as np
from libc.math cimport log
cimport numpy as np
ctypedef np.double_t DTYPE_t

cpdef dict extra_descriptors(np.ndarray[DTYPE_t, ndim=1] data,
                             double scale_pct):
    """
    Returns variance, skewness, kurtosis, entropy.
    Variance scales ∝ scale²; skewness/kurtosis/entropy are scale‐invariant.
    """
    cdef Py_ssize_t n = data.shape[0]
    if n < 2:
        raise ValueError("Need ≥2 points")

    cdef double factor = scale_pct/100.0
    cdef double sum0=0.0, sum1=0.0, sum2=0.0, sum3=0.0
    cdef double mu, var, skew, kurt
    cdef Py_ssize_t i

    # mean on scaled data
    for i in range(n):
        sum0 += factor * data[i]
    mu = sum0/n

    # central moments
    for i in range(n):
        cdef double d = factor*data[i] - mu
        sum1 += d*d
        sum2 += d*d*d
        sum3 += d*d*d*d

    var  = sum1 / n           # ∝ scale²
    skew = (sum2/n) / (var**1.5) if var>0 else 0.0
    kurt = (sum3/n) / (var**2)   if var>0 else 0.0

    # Shannon entropy (scale‐invariant)
    cdef int bins = 10
    cdef np.ndarray[np.int_t, ndim=1] hist = np.zeros(bins, dtype=np.int32)
    cdef double dmin = (data.min()*factor), dmax = (data.max()*factor)
    cdef double width = (dmax-dmin)/bins
    for i in range(n):
        cdef int idx = <int>(((data[i]*factor)-dmin)/width)
        if idx==bins: idx = bins-1
        hist[idx] += 1
    cdef double ent=0.0
    for i in range(bins):
        if hist[i]:
            cdef double p = hist[i]/n
            ent -= p * log(p)

    return {
      'variance': var,
      'skewness': skew,
      'kurtosis': kurt,
      'shannon_entropy': ent
    }
