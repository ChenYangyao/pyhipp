from pyhipp.core import DataDict
from pyhipp.stats import binning
import numpy as np

class TestSummary:

    def get_bin(self, n_bins):
        return binning._BinnedData(n_bins)

    def get_rind_seq(self, rn_min, rn_max, n_numbers):
        return np.random.randint(rn_min, rn_max, n_numbers)
    
    def get_rval_seq(self, n_numbers):
        return np.random.uniform(0, 100., n_numbers)

    def count(self, bin: binning._BinnedData, rinds: np.ndarray):
        _count_exp = ((rinds >= 0) & (rinds < bin.n_bins())).sum()
        count_exp = 0.
        for i in range(10):
            count_exp += _count_exp
            bin.cnt_n_chked(rinds)
            count = bin.data.sum()
            assert np.abs(count_exp - count) < .5, \
                f"Expected {count_exp}, but got {count}"
    
    def add(self, bin: binning._BinnedData, rinds: np.ndarray,
            rvals: np.ndarray):
        sel = (rinds >= 0) & (rinds < bin.n_bins())
        _add_exp = (rvals[sel]).sum()
        add_exp = 0.
        for i in range(10):
            add_exp += _add_exp
            bin.add_n_chked(rinds, rvals)
            add = bin.data.sum()
            assert np.abs(add_exp - add) < .5, \
                f"Expected {add_exp}, but got {add}"

    def test_cnt_n_chked(self):
        for _n in 0, 1, 2, 3, 5, 10, 100:
            rinds = self.get_rind_seq(-_n, 2*_n+1, 1024)
            bin = self.get_bin(_n)
            self.count(bin, rinds)

    def test_add_n_chked(self):
        for _n in 0, 1, 2, 3, 5, 10, 100:
            rinds = self.get_rind_seq(-_n, 2*_n+1, 1024)
            rvals = self.get_rval_seq(1024)
            bin = self.get_bin(_n)
            self.add(bin, rinds, rvals)