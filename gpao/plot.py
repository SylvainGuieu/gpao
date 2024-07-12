
from __future__ import annotations
from dataclasses import dataclass, field
from matplotlib.pylab import plt , Axes
import functools 
@dataclass
class AxesMaker:
    figure: int|str|None = None
    show: bool = True 

    @functools.cached_property
    def axes(self)->Axes:
        fig = plt.figure(self.figure)
        ax = fig.add_subplot(1,1,1)
        if self.show: fig.show()
        return ax

    def get_axes(self)->Axes:
        return self.axes
            
