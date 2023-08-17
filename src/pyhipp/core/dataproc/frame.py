import numpy as np

class Polor:
    @staticmethod
    def unit_vec_to_cart(theta: np.ndarray, phi: np.ndarray, stack=True):
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        cos_p, sin_p = np.cos(phi), np.sin(phi)
        z = cos_t
        x = sin_t * cos_p
        y = sin_t * sin_p
        out = x,y,z
        if stack:
            out = np.stack(out, axis=-1)
        return out
    
class Cartesian:
    pass
    
class PeriodicBox:
    @staticmethod
    def shift_in(x: np.ndarray, l_box: float, where=None, copy=True):
        if copy:
            x = x.copy()
            
        mask = PeriodicBox.__combine_mask(x < 0.0, where)
        x[mask] += l_box
            
        mask = PeriodicBox.__combine_mask(x >= l_box, where)
        x[mask] -= l_box
            
        return x
    
    @staticmethod
    def shift_to(x: np.ndarray, x_ref: np.ndarray, l_box: float, 
                 where=None, copy=True):
        if copy:
            x = x.copy()
        l_half = .5 * l_box
        dx = x - x_ref
            
        mask = PeriodicBox.__combine_mask(dx < -l_half, where)
        x[mask] += l_box
            
        mask = PeriodicBox.__combine_mask(dx >= l_half, where)
        x[mask] -= l_box
            
        return x
    
    @staticmethod
    def test_bound(x: np.ndarray, l_box: float, where=None):
        if where is not None:
            x = x[where]
            
        return ((x >= 0.0) & (x < l_box)).all()
    
    @staticmethod
    def __combine_mask(mask, where):
        return mask if where is None else mask & where
    