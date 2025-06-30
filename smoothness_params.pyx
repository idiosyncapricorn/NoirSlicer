# sizing_scale.pyx
# cython: boundscheck=False, wraparound=False, language_level=3

import numpy as np
cimport numpy as np
ctypedef np.double_t DTYPE_t

cpdef dict analyze_scale(np.ndarray[DTYPE_t, ndim=1] data,
                         double scale_pct,
                         double target_min,
                         double target_max):
    """
    data            : raw sizes
    scale_pct       : user‐chosen scaling in percent (e.g. 150 → 1.5×)
    target_min,max  : desired output range
    """
    cdef Py_ssize_t n = data.shape[0]
    cdef double factor = scale_pct / 100.0

    # pre‐scale data copy
    cdef np.ndarray[DTYPE_t, ndim=1] scaled = data * factor

    # compute stats on scaled[]
    cdef double smin = scaled[0], smax = scaled[0], ssum = 0.0, ssum2 = 0.0
    cdef Py_ssize_t i
    for i in range(n):
        ssum  += scaled[i]
        ssum2 += scaled[i]*scaled[i]
        if scaled[i] < smin: smin = scaled[i]
        elif scaled[i] > smax: smax = scaled[i]
    cdef double mean = ssum / n
    cdef double var  = ssum2/n - mean*mean
    cdef double std  = var > 0.0 and var**0.5 or 0.0

    # linear map to [target_min, target_max]
    cdef double scale = (target_max - target_min) / (smax - smin)
    cdef double shift = target_min - smin * scale

    # suggestions around that map
    cdef list factors = [scale * f for f in (0.5, 1.0, 1.5, 2.0)]

    return {
      'user_factor': factor,
      'min': smin, 'max': smax, 'mean': mean, 'std': std,
      'linear_map': (scale, shift),
      'zscore': (mean, std),
      'log_ok': smin > 0,
      'suggested_factors': factors
    }
