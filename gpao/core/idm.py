

from __future__ import annotations
import abc
from typing import Protocol
import numpy as np
from numpy.typing import ArrayLike
from gpao.core.ihk import HkData
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



class IDm(Protocol):
    def get_property(self)->IDmProperty:
        ...
    
    def get_calibration(self)->IDmCalibration:
        ... 
    
    def send_actuator(self,  act:int, amplitude:float)->None:
        ... 
        
    def send_group(self, act:int, size:float, amplitude:float)->None:
        ... 

    def send_flat(self)->None:
        ...

    def send_rest(self)->None:
        ...
    
    def reset(self)->None:
        ...

    def send_kl(self, mode:int, amplitude:float)->None:
        ...
    
    def get_hk_data(self)->tuple[HkData, ...]:
        ...

    def get_current(self)->tuple[float,...]:
        ...
