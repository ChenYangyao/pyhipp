from .runtime_config import RuntimeConfig
from .abc import MplObj
from .color import Color
from .axes import Axes, AxesArray
from .figure import Figure
from .shortcut import figure, subplots, subfigures, savefig, show, close
from .scatter_2d import DensityEstimator2D, Histogram2D, KNearestNeighbor2D, Scatter2D

RuntimeConfig().set_global()