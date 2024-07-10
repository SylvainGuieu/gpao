
from __future__ import annotations
from dataclasses import dataclass, field
from matplotlib.pylab import plt , Axes
import functools 
@dataclass
class AxesMaker:
    figure: int|str|None = None
    
    @functools.cached_property
    def axes(self):
        return plt.figure(self.figure).add_subplot(1,1,1)

    def get_axes(self):
        return self.axes
            
