from typing import Protocol
from matplotlib.pylab import Axes, Figure


class IAxesMaker(Protocol):
    def get_axes(self)->Axes:
        ...
    def get_figure(self)->Figure:
        ...
