from __future__ import annotations
from dataclasses import dataclass
import time

from numpy.typing import ArrayLike


@dataclass
class DmComSim:
    serial_name: str 
    sleep: float = 0.1 
    def send(self,  cmd:ArrayLike, sleep:float|None=None):
        time.sleep(self.sleep if sleep is None else sleep)


