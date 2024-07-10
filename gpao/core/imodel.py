from typing import Protocol
import numpy as np 
from numpy.typing import ArrayLike

class IDmPattern(Protocol):
    x: np.ndarray 
    y: np.ndarray 
    mask: np.ndarray 
    name: str
    def as_screen(self, values: ArrayLike, fill:float=... )->np.ndarray:
        ... 

class IRegion(Protocol):
    def contains(self, x: ArrayLike, y:ArrayLike)->np.ndarray:
        ...

    def polynoms(self)->list[ArrayLike]:
        ...
