from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
from gpao.core.imodel import IRegion


@dataclass
class DmPattern:
    x: np.ndarray
    y: np.ndarray
    mask: np.ndarray
    name: str = ""
    def as_screen(self, values: ArrayLike, fill:float=np.nan )->np.ndarray:
        values = np.asarray(values)
        img = np.ones( self.mask.shape, values.dtype)*fill 
        img[self.mask] = values
        return img

@dataclass
class RegionDisk:
    """ A Region defined as a disk """
    x: float 
    y: float 
    radius: float
    resolution: int = 100 
    """ resolution of the polynome """
    def contains(self, x: ArrayLike, y:ArrayLike):
        x, y = np.asarray(x), np.asarray(y)
        return  ( (x-self.x)**2 + (y-self.y)**2 ) < self.radius**2

    def polynoms(self)->list[ ArrayLike ]:
        a = np.linspace( 0, 2*np.pi-2*np.pi/self.resolution, self.resolution )
        pol = np.ndarray( (2,self.resolution), dtype=float)
        pol[0,:] = self.radius * np.cos(a)
        pol[1,:] = self.radius * np.sin(a) 
        return [pol]
        
class Regions:
    """ Compose regions (or operator) """
    regions: list[IRegion]
    def __init__(self, *regions):
        self.regions = list(regions) 

    def contains(self, x: ArrayLike, y:ArrayLike)->np.ndarray:
        x, y = np.asarray(x), np.asarray(y)
        test = np.zeros( x.shape, dtype=bool)
        for region in self.regions:
            test[ region.contains(x,y) ] = True 
        return test 

    def polynoms(self)->list[ArrayLike]:
        return sum( (r.polynoms() for r in self.regions), [] )

def build_pattern(
      n:int,
      w:float,
      region: IRegion|None= None,  
      exclude: IRegion|None=None, 
      name:str = ""
    )->DmPattern:
    """ Build a DM Pattern 
    
    Args:
        n: max number of actuator at each side 
        w: spearation between actuators 
        ri: internal radius  
    """
    x0 = np.arange(n , dtype= np.float32)
    x0 -= x0.mean()
    x,y = np.meshgrid( w*x0, w*x0 )
    ok = np.ones( x.shape, bool)
    if region:
        ok[~region.contains(x,y)] = False
    if exclude is not None:
        ok[ exclude.contains(x,y) ] = False
            
    return DmPattern(x[ok], y[ok], ok, name=name) 
   

