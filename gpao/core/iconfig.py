from __future__ import annotations
from typing import Protocol
import numpy as np

class IConfigIo(Protocol):
    def read_flat(self)->np.ndarray:
        ...
    
    def read_lut(self)->list[int]:
        ...
