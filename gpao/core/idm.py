

from __future__ import annotations
import abc
from typing import Protocol
import numpy as np
from numpy.typing import ArrayLike
from gpao.core.imodel import IDmPattern 

class IDmProperty(Protocol):
    nact: int 
    center: int
    serial_name: str 
    pitch: float 
    pixel_scale: float 
    act_box: tuple[int,int]
    pattern: IDmPattern

class IDmCalibration(Protocol):
    def get_flat(self)->np.ndarray:
        ... 
    def get_kl(self)->np.ndarray:
        ...
    def get_channel_number(self, actnum:int)->int:
        ... 
    def get_cable_number(self, actnum:int)->int:
        ...

class IBaseGpaoDm(Protocol):
    def send(self,cmd:ArrayLike, sleep:float=...)->None:
        ...

class IDmCommand(Protocol):
    
    def get_property(self)->IDmProperty:
        ...
    
    def get_calibration(self)->IDmCalibration:
        ... 

    def actuator(self, act:int, amplitude:float)->np.ndarray:
        ...

    def group(self, act:int, size:float, amplitude:float)->np.ndarray:
        ...

    def flat(self)->np.ndarray:
        ... 

    def rest(self)->np.ndarray:
        ... 

    def kl(self, mode:int, amplitude:float)->np.ndarray:
        ...



