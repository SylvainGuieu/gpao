from typing import Protocol
from matplotlib.pylab import Axes


class IAxesMaker(Protocol):
    def get_axes(self)->Axes:
        ...

