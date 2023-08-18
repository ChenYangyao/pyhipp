import numpy as np

class Num:
    @staticmethod
    def bound(x, lo = None, hi = None, copy = True):
        '''
        Depeneding on whether or not x is a scalar, call bound_{scalar|array}().
        For array, controlled by ``copy``.
        '''
        if np.isscalar(x):
            x = Num.bound_scalar(x, lo = lo, hi = hi)
        else:
            x = Num.bound_array(x, lo = lo, hi = hi, copy=copy)
        return x
    
    @staticmethod
    def bound_scalar(x, lo = None, hi = None):
        '''
        Return a new instance of x bounded by lo and hi.
        '''
        if lo is not None:
            x = max(x, lo)
        if hi is not None:
            x = min(x, hi)
        return x
    
    @staticmethod
    def bound_array(x: np.ndarray, lo = None, hi = None, copy = True):
        '''
        Inplace bound x within [lo, hi] and return a reference to it.
        If ``copy`` is True, copy ``x`` before bounding.
        '''
        x = np.asarray(x)
        if copy:
            x = x.copy() 
        if lo is not None:
            sel = x < lo
            x[sel] = lo
        if hi is not None:
            sel = x > hi
            x[sel] = hi
        return x
    
    @staticmethod
    def safe_div(x: np.ndarray, y: np.ndarray, lo=1.0e-10):
        return x / Num.bound(y, lo)
    
    @staticmethod
    def lg(*args):
        if len(args) == 1:
            return np.log10(args[0])
        return tuple(np.log10(arg) for arg in args)
    
    @staticmethod
    def safe_lg(*args: np.ndarray, lo=1.0e-10):
        '''
        Pad each of args with lo, then take np.log10.
        Return a tuple if len(args) > 1.
        '''
        if len(args) == 1:
            x = Num.bound(args[0], lo)
            return Num.lg(x)
        return tuple(Num.safe_lg(arg, lo=lo) for arg in args)
    
    @staticmethod
    def ln(*args):
        if len(args) == 1:
            return np.log(args[0])
        return tuple(np.log(arg) for arg in args)
    
    @staticmethod
    def safe_ln(*args: np.ndarray, lo=1.0e-10):
        '''
        Pad each of args with lo, then take np.log.
        Return a tuple if len(args) > 1.
        '''
        if len(args) == 1:
            x = Num.bound(args[0], lo)
            return Num.ln(x)
        return tuple(Num.safe_ln(arg, lo=lo) for arg in args)
    
    @staticmethod
    def norm(x, axis=-1, keepdims=False):
        x = np.asarray(x)
        return np.linalg.norm(x, axis=axis, keepdims=keepdims)
    
    @staticmethod
    def normalize(x, axis=-1, lo=1.0e-10, inplace=False):
        x = np.asarray(x)
        n = Num.norm(x, axis=axis, keepdims=True)
        n = np.clip(n, lo, None, out=n) 
        if inplace:
            x /= n
        else:
            x = x / n
        return x