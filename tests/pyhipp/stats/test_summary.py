from pyhipp.core import DataDict
from pyhipp.stats.summary import Summary
import numpy as np


class TestBinnedData:

    def get_data(self, n) -> np.ndarray:
        return np.random.uniform(0, 100., n)

    def approx_equal(self, a, b, tol=1e-5):
        return np.all(np.abs(a - b) < tol)

    def summary_approx_equal(self, s: Summary.FullResult, x: np.ndarray, 
                             ind: tuple, tol=1e-5):
        x = x[:, *ind]
        assert self.approx_equal(s.mean[ind], x.mean())
        assert self.approx_equal(s.sd[ind], x.std())
        assert self.approx_equal(s.median[ind], np.median(x))
        assert self.approx_equal(s.min[ind], x.min())
        assert self.approx_equal(s.max[ind], x.max())
        assert self.approx_equal(s.sigma_1[0,*ind],
                                np.quantile(x, Summary.p_sigma_1[0]))
        assert self.approx_equal(s.sigma_1[1,*ind],
                                np.quantile(x, Summary.p_sigma_1[1]))
        assert self.approx_equal(s.sigma_2[0,*ind],
                                np.quantile(x, Summary.p_sigma_2[0]))
        assert self.approx_equal(s.sigma_2[1,*ind],
                                np.quantile(x, Summary.p_sigma_2[1]))
        assert self.approx_equal(s.sigma_3[0,*ind],
                                np.quantile(x, Summary.p_sigma_3[0]))
        assert self.approx_equal(s.sigma_3[1,*ind],
                                np.quantile(x, Summary.p_sigma_3[1]))
        
    def test_summary_1d(self):
        x = self.get_data(1024)
        s = Summary.on(x)

        assert np.isscalar(s.mean)
        assert np.isscalar(s.sd)
        assert np.isscalar(s.median)
        assert np.isscalar(s.min)
        assert np.isscalar(s.max)
        assert s.sigma_1.shape == (2,)
        assert s.sigma_2.shape == (2,)
        assert s.sigma_3.shape == (2,)
        self.summary_approx_equal(s, x, ())

    def test_summary_2d(self):
        for n in 1, 2, 3, 4, 8:
            x = self.get_data((1024, n))
            s = Summary.on(x)

            assert s.mean.shape == (n,)
            assert s.sd.shape == (n,)
            assert s.median.shape == (n,)
            assert s.min.shape == (n,)
            assert s.max.shape == (n,)
            assert s.sigma_1.shape == (2, n)
            assert s.sigma_2.shape == (2, n)
            assert s.sigma_3.shape == (2, n)
            
            for i in range(n):
                self.summary_approx_equal(s, x, (i,))

    def test_summary_3d(self):
        for n1 in 1, 2, 3, 4, 8:
            for n2 in 1, 2, 3, 4, 8:
                shape = (n1, n2)
                x = self.get_data((128,) + shape)
                s = Summary.on(x)
                
                assert s.mean.shape == shape
                assert s.sd.shape == shape
                assert s.median.shape == shape
                assert s.min.shape == shape
                assert s.max.shape == shape
                assert s.sigma_1.shape == (2,) + shape
                assert s.sigma_2.shape == (2,) + shape
                assert s.sigma_3.shape == (2,) + shape
                
                for i in range(n1):
                    for j in range(n2):
                        self.summary_approx_equal(s, x, (i,j))
    
    def test_as_dict(self):
        shapes = 1024, (1024, 2), (128, 128, 2, 3)
        for shape in shapes:
            x = self.get_data(shape)
            s = Summary.on(x)
            d = s.as_dict()
            
            assert np.all(d['mean'] == s.mean)
            assert np.all(d['sd'] == s.sd)
            assert np.all(d['median'] == s.median)
            assert np.all(d['min'] == s.min)
            assert np.all(d['max'] == s.max)
            assert np.all(d['sigma_1'] == s.sigma_1)
            assert np.all(d['sigma_2'] == s.sigma_2)
            assert np.all(d['sigma_3'] == s.sigma_3)
            
    def test_from_dict(self):
        shapes = 1024, (1024, 2), (128, 128, 2, 3)
        for shape in shapes:
            x = self.get_data(shape)
            s = Summary.on(x)
            d = s.as_dict()
            s2 = Summary.FullResult.from_dict(d)
            assert np.all(s.mean == s2.mean)
            assert np.all(s.sd == s2.sd)
            assert np.all(s.median == s2.median)
            assert np.all(s.min == s2.min)
            assert np.all(s.max == s2.max)
            assert np.all(s.sigma_1 == s2.sigma_1)
            assert np.all(s.sigma_2 == s2.sigma_2)
            assert np.all(s.sigma_3 == s2.sigma_3)