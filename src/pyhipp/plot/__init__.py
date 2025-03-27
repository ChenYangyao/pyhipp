from .runtime_config import RuntimeConfig
from .abc import MplObj
from .color import Color, ColorSeq
from .axes import Axes, AxesArray
from .figure import Figure
from .shortcut import figure, subplots, subfigures, savefig, show, close
from .two_dim import DensityEstimator2D, Histogram2D, KNearestNeighbor2D, Scatter2D

runtime_config = RuntimeConfig()
runtime_config.set_global()