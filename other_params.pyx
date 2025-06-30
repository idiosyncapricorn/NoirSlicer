# other_params.pyx
# cython: boundscheck=False, wraparound=False, language_level=3

cimport numpy as np
import numpy as np
from libc.math cimport log, pow
ctypedef np.double_t DTYPE_t

cpdef dict extra_descriptors(np.ndarray[DTYPE_t, ndim=1] data):
    """
    Returns {
      'variance': float,
      'skewness': float,
      'kurtosis': float,
      'shannon_entropy': float
    }
    """
    cdef Py_ssize_t n = data.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 points")

    # compute mean & var
    cdef double sum0=0.0, sum1=0.0, sum2=0.0, sum3=0.0, sum4=0.0
    cdef Py_ssize_t i
    for i in range(n):
        sum0 += data[i]
    cdef double mu = sum0 / n
    for i in range(n):
        cdef double d = data[i] - mu
        sum1 += d*d
        sum2 += d*d*d
        sum3 += d*d*d*d
    cdef double var = sum1 / n
    # skewness & kurtosis
    cdef double skew = (sum2/n) / (var**1.5) if var>0 else 0.0
    cdef double kurt = (sum3/n) / (var**2)    if var>0 else 0.0

    # approximate Shannon entropy via histogram
    cdef int bins = 10
    cdef np.ndarray[np.int_t, ndim=1] hist = np.zeros(bins, dtype=np.int32)
    cdef double dmin = data.min(), dmax = data.max(), width = (dmax-dmin)/bins
    for i in range(n):
        cdef int idx = <int>((data[i]-dmin)/width)
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
