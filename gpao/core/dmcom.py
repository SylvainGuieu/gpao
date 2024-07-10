from dataclasses import dataclass
import time
from Lib64.asdk import DM as _DM
import numpy as np

@dataclass
class DmCom:
    """ Wrapper to ALPAO SKD """
    serial_name: str 
    sleep: float = 0.1 

    def __post_init__(self):
        self._dm = _DM(self.serial_name) 
         
    def send(self, cmd:np.ndarray, sleep=None):
        self._dm.Send( cmd )
        time.sleep(self.sleep if sleep is None else sleep)
        
    
    
