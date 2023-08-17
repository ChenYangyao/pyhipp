import numpy as np
from ...core.abc import HasName, HasValue, HasSimpleRepr
from typing import Sequence, Union, Iterator, Tuple

class Param(HasName, HasValue, HasSimpleRepr):
    def __init__(self, value: np.ndarray, name: str = None) -> None:
        super().__init__(name=name, value=value)
    
    def to_simple_repr(self):
        return {
            'name': self.name,
            'value': self.value.tolist(),
        }
        
class ParamList(HasName, HasSimpleRepr):
    def __init__(self, params: Sequence[Param]) -> None:
        super().__init__()
        
        _params = list(p for p in params)
        _name2idx = {p.name: i for i, p in enumerate(_params)}
        
        self._params = _params
        self._name2idx = _name2idx
        
    def __getitem__(self, key: Union[str, int]) -> Param:
        if not isinstance(key, int):
            key = self._name2idx[key]
        return self._params[key]
    
    def __setitem__(self, idx: int, param: Param) -> None:
        self._params[idx] = param
        self._name2idx[param.name] = idx
    
    def __len__(self) -> int:
        return len(self._params)
    
    @property
    def values(self) -> Tuple[np.ndarray]:
        return tuple(p.value for p in self._params)
    
    def set_values(self, values: Sequence[np.ndarray]):
        assert len(values) == len(self)
        for i, p in enumerate(self._params):
            p.set_value(values[i])
    
    def __iter__(self) -> Iterator[Param]:
        return iter(self._params)
        
    def to_simple_repr(self):
        return [p.to_simple_repr() for p in self._params]