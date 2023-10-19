from dataclasses import dataclass
from ...core import DataDict, dataproc as dp
from ...stats import Bootstrap, Bin, Hist1D
import numpy as np

class BinnedVolumeDensity:
    '''
    Used for, e.g., GSMF (set volume to the actual volume), CGSMF 
    (set volume to n_halos).
    '''
    @dataclass
    class Policy:
        lg_y_pad: float = 1.0e-8
    
    @staticmethod
    def from_raw(x_to_bin, volume, 
                 bins=15, sub_bins=3, range=None, p_range=None, 
                 weights=None, 
                 n_bootstrap = 10,
                 policy: Policy=None,
                 **resample_kw) -> DataDict:
        '''
        Returned: 
            x, dx, sub_e, h, y, lg_y,
            _sd for h, y and lg_y
        '''
        if policy is None:
            policy = BinnedVolumeDensity.Policy()
            
        x = np.asarray(x_to_bin)
        if np.isscalar(bins):
            range = Bin.parse_p_range_spec(x, range, p_range)
        bins, sub_bins, (x_min, x_max) = Bin.parse_sub_bin_spec(
            bins, sub_bins, range)
        sel = (x>=x_min)&(x<x_max)
        dset_in = {'x_to_bin': x[sel]}
        stats_kw = {'bins': bins, 'sub_bins': sub_bins, 'range': range,
                    'volume': volume, 'lg_y_pad': policy.lg_y_pad}
        if weights is not None:
            dset_in['weights'] = weights[sel] 
        else:
            stats_kw['weights'] = None
        
        d_out = Bootstrap.resampled_call(
            BinnedVolumeDensity.__hist,
            dsets_in = (dset_in,),
            keys_out = ('h', 'y', 'lg_y'),
            stats_kw = stats_kw,
            n_resample = n_bootstrap,
            **resample_kw,
            keep_samples = True,
        )
        return d_out
    
    @staticmethod
    def __hist(x_to_bin, bins, sub_bins, range, weights, volume, lg_y_pad):
        out = Hist1D.from_overlapped_bins(x_to_bin, bins=bins, 
            sub_bins=sub_bins, range=range, weights=weights)
        x, dx, sub_e, h = out['x', 'dx', 'sub_e', 'h']
        y = h / (volume * dx)
        lg_y = dp.Num.safe_lg(y, lo=lg_y_pad)
        
        return DataDict({
            'x': x, 'dx': dx, 'sub_e': sub_e, 
            'h': h, 'y': y, 'lg_y': lg_y
        })
        
        