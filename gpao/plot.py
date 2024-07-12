
from __future__ import annotations
from dataclasses import dataclass, field
from matplotlib.pylab import Figure, plt , Axes
import functools 
@dataclass
class AxesMaker:
    figure: int|str|None = None
    show: bool = True 

    @functools.cached_property
    def axes(self)->Axes:
        fig = self.get_figure()
        ax = fig.add_subplot(1,1,1)
        if self.show: fig.show()
        return ax
    
    @functools.cached_property
    def _fig(self)->Figure:
        return plt.figure(self.figure)
    
    def get_figure(self)->Figure:
        return self._fig

    def get_axes(self)->Axes:
        return self.axes
            
