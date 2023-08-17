import numpy as np

class Filter:
    def __init__(self, **kw) -> None:
        super().__init__(**kw)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self._map_array(x)
        
    def _map_array(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

class BooleanBasedFilter(Filter):
    def _map_array(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        return x[self._get_boolean_mask(x)]
        
    def _get_boolean_mask(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

class Range(BooleanBasedFilter):
    def __init__(self, min = None, max = None, **kw) -> None:
        super().__init__(**kw)
        
        self.min = min
        self.max = max
        
    def _get_boolean_mask(self, x: np.ndarray) -> np.ndarray:
        min, max = self.min, self.max
        
        mask = np.ones_like(x, dtype=bool)
        if min is not None:
            mask &= ( x >= min )
        if max is not None:
            mask &= ( x < max )
        
        return mask