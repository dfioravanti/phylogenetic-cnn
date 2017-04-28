# 2014-01-09: [MC] addressed norm_l2 (added "if m is None" checks)
# 2013-11-21: [AZ] added scaling method norm L2
import numpy as np

def minmax_scaling(x, m=None, r=None):
    """
    m: midrange
    r: range
    """
    
    xmin = np.min(x, axis=0)
    xmax = np.max(x, axis=0)
    
    if m is None:
        m = (xmin + xmax) / 2
    if r is None:
        r = xmax - xmin

    #with np.errstate(divide='ignore'):
    np.seterr(divide='ignore')
    ret = (x-m)/(r/2)
        
    ret[np.where(np.isnan(ret))] = 0.0  
    return ret, m, r


def standardize(x, m=None, r=None):
    """
    m: midrange
    r: range
    """

    if m is None:
        m = np.mean(x, axis=0)
    if r is None:
        r = np.std(x, axis=0, ddof=1)

    #with np.errstate(divide='ignore'):
    np.seterr(divide='ignore', invalid='ignore')
    ret = (x-m)/r
     
    ret[np.where(np.isnan(ret))] = 0.0  
    return ret, m, r


def norm_l2(x, m = None, r = None):
    """
    m: midrange
    r: range
    """

    if m is None:
        m = np.mean(x) # not axis = 0 because result = (x-E(x))/normL2(X)

    # centering
    x = x-m

    if r is None:
        r = np.sqrt(np.sum((x)**2, axis = 0))

    #with np.errstate(divide='ignore'):
    np.seterr(divide='ignore')
    ret = x/r 
	 
    ret[np.where(np.isnan(ret))] = 0.0

    return ret, m, r

