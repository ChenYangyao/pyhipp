from ..stats.random import Rng
from ..core import abc
from dataclasses import dataclass
from .comm import Comm

@dataclass
class Policy:
    dist_seed    = True

class DistributedRng(abc.HasDictRepr):
    def __init__(self, comm: Comm, pl: Policy = None, **kw) -> None:
        
        super().__init__(**kw)        
        
        if pl is None:
            pl = Policy()
        
        assert pl.dist_seed
        
        self.seed = comm.rank
        
    def get_sequential(self) -> Rng:
        return Rng(seed = self.seed)
        

