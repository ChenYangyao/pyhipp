# Copyright (C) 2026 Yangyao Chen (yangyaochen.astro@foxmail.com) - All Rights
# Reserved
#
# You may use, distribute and modify this code under the MIT license. We kindly
# request you to give credit to the original author(s) of this code, and cite
# the following paper(s) if you use this code in your research:
# - Chen, Y. and Wang K. 2023, ASCL:2301.030, NASA/ADS entry:
#   https://ui.adsabs.harvard.edu/abs/2023ascl.soft01030C/abstract
#   (this package, pyhipp).
# - Zhang Z. et al. 2025. Nature 642, 47-52 (for definition of 2PCCF and
#   relative bias).
# - Wang H. et al. 2016. ApJ, 831, 164 (for the reconstructed density field
#   and halo catalogs).

from __future__ import annotations
import numpy as np
from pyhipp.core import DataDict, abc, DataTable, Num
from pyhipp.stats.summary import Summary
from pyhipp.stats import Rng


class BinUtils:
    def rs2rs_c(rs: np.ndarray):
        '''
        Compute the bin centers of the radial bins defined by the edges `rs`.
        '''
        rs_c = .5 * (rs[1:] + rs[:-1])
        lg_rs = Num.safe_lg(rs)
        lg_rs_c = .5 * (lg_rs[1:] + lg_rs[:-1])
        return DataDict({
            'rs': rs.copy(), 'rs_c': rs_c,
            'lg_rs': lg_rs, 'lg_rs_c': lg_rs_c
        })


class RelativeBiasUtils:
    @staticmethod
    def relative_bias_1(wp_dst: DataDict, wp_ref: DataDict):
        '''
        Find the relative bias between two projected correlation functions 
        (2PCCF), `wp_dst` and `wp_ref`.
        Both 2PCCF should be evaluated in a single radial range, e.g.,
        in r_min <= r < r_max, with n times of bootstrapping.
        '''
        wps = wp_dst['wp']
        wps_ref = wp_ref['wp']
        assert wps.ndim == 2
        assert wps.shape[1] == 1
        assert wps.shape == wps_ref.shape
        rel_biass = wps[:, 0] / wps_ref[:, 0]
        return DataDict(Summary.on(rel_biass).as_dict())

    @staticmethod
    def relative_bias_curve(
            wp_dsts: list[DataDict],
            wp_ref: DataDict | None = None):
        '''
        Find the relative bias curve between a list of 2PCCFs `wp_dsts` and a 
        reference 2PCCF `wp_ref` (defaults to the last element of `wp_dsts`).
        '''
        if wp_ref is None:
            wp_ref = wp_dsts[-1]
        rbs = [
            RelativeBiasUtils.relative_bias_1(wp_dst, wp_ref)
            for wp_dst in wp_dsts]
        return DataDict({k: np.array([rb[k] for rb in rbs])
                         for k in rbs[0].keys()})


class CCFPeriodicProjectedUtils:
    @staticmethod
    def pair_count(
            x1: np.ndarray, x2: np.ndarray | None, l_box: float, rs: np.ndarray,
            n_threads=1, pi_max=10.0):

        from Corrfunc.theory import DDrppi
        rs = np.asarray(rs)

        kw = {'nthreads': n_threads, 'pimax': pi_max, 'binfile': rs,
              'periodic': True, 'boxsize': l_box}
        n_bin = len(rs) - 1
        kw['X1'], kw['Y1'], kw['Z1'] = x1.T
        n1 = len(x1)
        if x2 is not None:
            kw['autocorr'] = False
            kw['X2'], kw['Y2'], kw['Z2'] = x2.T
            n2 = len(x2)
        else:
            kw['autocorr'] = True
            n2 = n1

        n_pairs = DDrppi(**kw)['npairs'].reshape(n_bin, -1).astype(np.float64)
        out = BinUtils.rs2rs_c(rs) | {
            'n_pairs': n_pairs,
            'n1': n1, 'n2': n2,
            'pi_max': pi_max,
            'l_box': l_box
        }

        return DataDict(out)

    @staticmethod
    def n_pairs2wp(pair_data: DataDict):
        rs, n1, n2, l_box, n_pairs = pair_data['rs',
                                               'n1', 'n2', 'l_box', 'n_pairs']
        vol = np.pi * rs**2 * 2.0
        dvol = np.diff(vol)
        exp_n = dvol * n2 / l_box**3
        xi = n_pairs / n1 / exp_n[:, None] - 1.
        wp = np.sum(xi * 2., axis=1)
        return DataDict({
            'exp_n': exp_n, 'dvol': dvol, 'xi': xi, 'wp': wp
        })


class SimSample(abc.HasDictRepr):

    repr_attr_keys = ('l_box', 'n_objs')

    def __init__(self, l_box: float, data: DataTable):

        assert (data['x'] >= 0).all()
        assert (data['x'] < l_box).all()

        n_objs = len(data['x'])

        self.l_box = l_box
        self.n_objs = n_objs
        self.data = DataTable(data)

    def bootstrapped(
            self, n: int | None = None, rng: Rng | int = 10086, replace=True):
        rng = Rng(rng)
        n_in = self.n_objs
        if n is None:
            n = n_in
        inds = rng.choice(n_in, n, replace=replace)
        return self.subset(inds)

    def subset(self, args: np.ndarray | slice):
        return SimSample(self.l_box, self.data.subset(args))


class SimCCFProjected(abc.HasDictRepr):

    repr_attr_keys = ('s_ref', 'n_threads',
                      'pi_max', 'n_bootstrap', 'n_max_rand')

    def __init__(self, s_ref: SimSample, rng: Rng | int = 10086,
                 n_threads=1, pi_max=10.0, n_bootstrap=10,
                 n_max_rand=None
                 ):

        self.s_ref = s_ref
        self.rng = Rng(rng)
        self.n_threads = n_threads
        self.pi_max = pi_max
        self.n_bootstrap = n_bootstrap
        self.n_max_rand = n_max_rand

    def wp(self, s_dst: SimSample, rs: np.ndarray):
        rng, n_bootstrap, n_max_rand, s_ref = (
            self.rng, self.n_bootstrap, self.n_max_rand, self.s_ref)
        x_ref, l_box = s_ref.data['x'], s_ref.l_box
        pc_kw = {
            'l_box': l_box, 'n_threads': self.n_threads,
            'pi_max': self.pi_max, 'x2': x_ref, 'rs': rs
        }
        details = []
        Utils = CCFPeriodicProjectedUtils
        for _ in range(n_bootstrap):
            x1 = s_dst.bootstrapped(n=n_max_rand, rng=rng).data['x']
            pc = Utils.pair_count(x1, **pc_kw)
            wp = Utils.n_pairs2wp(pc)
            details.append(wp | pc)
        out = DataDict({
            'bootstrap_details': details
        })
        for key in 'n_pairs', 'xi', 'wp':
            out[key] = np.array([detail[key] for detail in details])
        return out

    def relative_bias_curve(self, s_dst: SimSample, bin_by_key: str,
                            bin_edges: np.ndarray, ref_bin=-1,
                            r_min=1., r_max=10.):
        val = s_dst.data[bin_by_key]
        rs = np.array([r_min, r_max])
        x_subs = []
        wp_subs = []
        for (lo, hi) in zip(bin_edges[:-1], bin_edges[1:]):
            sel = (val >= lo) & (val < hi)
            wp = self.wp(s_dst.subset(sel), rs=rs)
            wp_subs.append(wp)
            x_subs.append(Summary.on(val[sel]).as_dict())

        x = DataDict({key: np.array([x_sub[key] for x_sub in x_subs])
                      for key in x_subs[0].keys()})
        wp_ref = wp_subs[ref_bin]
        y = RelativeBiasUtils.relative_bias_curve(wp_subs, wp_ref)

        return DataDict({'x': x, 'y': y})
