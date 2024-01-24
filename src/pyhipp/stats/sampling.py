from typing import Callable, Iterable, Mapping, Optional, Any, Union, List, Dict
from .random import Rng
from ..core import DataDict, dataproc as dp
from .stacking import Stack
import numpy as np


class Bootstrap:

    @staticmethod
    def resampled_call(
            stats_fn: Callable,
            dsets_in: Iterable[Mapping[str, np.ndarray]],
            keys_out: Iterable[str],
            stats_kw: Optional[Mapping[str, Any]] = None,
            n_resample: int = 10,
            rng: Rng.Initializer = 0,
            keep_samples: bool = False) -> DataDict:

        rng = Rng(rng)

        dsets_out: List[Mapping] = []
        for _ in range(n_resample):
            _dset_in = Bootstrap.__resample(dsets_in, rng)
            if stats_kw is not None:
                _dset_in |= stats_kw
            _dset_out = stats_fn(**_dset_in)
            dsets_out.append(_dset_out)

        dset_out = DataDict()
        for key in keys_out:
            vals = np.array([d[key] for d in dsets_out])
            mean, sd = Stack.mean_and_sd(vals)
            dset_out |= {
                f'{key}': mean, f'{key}_sd': sd,
            }
            if keep_samples:
                dset_out[f'{key}_samples'] = vals
        for k, v in dsets_out[0].items():
            dset_out.setdefault(k, v)

        return dset_out

    @staticmethod
    def __resample(dsets: List[Dict[str, np.ndarray]], rng: Rng):
        dset_out = DataDict()
        for dset in dsets:
            ids = None
            for k, v in dset.items():
                if ids is None:
                    n = len(v)
                    ids = rng.choice(n, size=n)
                dset_out[k] = v[ids]
        return dset_out


class RandomNoise:

    def __init__(self, sigma: float, noise_dist='normal',
                 noise_in_lg=False, rng: Rng = 0) -> None:

        assert noise_dist == 'normal'

        self.sigma = float(sigma)
        self.noise_dist = str(noise_dist)
        self.noise_in_lg = bool(noise_in_lg)
        self.rng = Rng(rng)

    def add_to(self, x: np.ndarray, n_repeats=1) -> np.ndarray:
        rng, sigma, in_lg = self.rng, self.sigma, self.noise_in_lg

        x = np.asarray(x)
        n_xs = len(x)
        dx = rng.standard_normal((n_repeats, n_xs)) * sigma
        if in_lg:
            out = x * 10**dx
        else:
            out = x + dx

        if n_repeats == 1:
            out = out[0]

        return out
