# sizing_scale.pyx
# cython: boundscheck=False, wraparound=False, language_level=3

from cython.parallel cimport prange
cimport numpy as np
import numpy as np
ctypedef np.double_t DTYPE_t

cpdef dict analyze_scale(np.ndarray[DTYPE_t, ndim=1] data, double target_min, double target_max):
    """
    Returns {
      'min': float, 'max': float, 'mean': float, 'std': float,
      'linear_scale': lambda x: …,
      'zscore_scale': lambda x: …,
      'log_scale': lambda x: …,
      'suggested_factors': [float,…]
    }
    """
    cdef Py_ssize_t n = data.shape[0]
    cdef double smin = data[0], smax = data[0], ssum = 0.0, ssum2 = 0.0
    cdef Py_ssize_t i
    # 1. one-pass stats
    for i in range(n):
        ssum  += data[i]
        ssum2 += data[i]*data[i]
        if data[i] < smin: smin = data[i]
        elif data[i] > smax: smax = data[i]
    cdef double mean = ssum / n
    cdef double var  = ssum2/n - mean*mean
    cdef double std  = var > 0.0 and var**0.5 or 0.0

    # 2. linear factors to map [smin,smax]→[target_min,target_max]
    cdef double scale = (target_max - target_min) / (smax - smin)
    cdef double shift = target_min - smin * scale

    # 3. collect a few “suggested_factors” around that scale
    cdef list factors = [scale * f for f in (0.5, 1.0, 1.5, 2.0)]

    return {
      'min': smin, 'max': smax, 'mean': mean, 'std': std,
      'linear': (scale, shift),
      'zscore': (mean, std),
      'log': None if smin<=0 else True,
      'suggested_factors': factors
    }
