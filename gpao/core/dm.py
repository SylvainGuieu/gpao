from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from gpao.core.idm import IDmCalibration, IDmProperty
from gpao.core.imodel import IDmPattern

@dataclass
class DmProperty:
    """ Hold DM Property information """
    serial_name: str
    nact: int 
    center: int
    pitch: float 
    pixel_scale: float 
    act_box: tuple[int, int] 
    pattern: IDmPattern

@dataclass
class DmCalib:
    """ Hold DM Calibration for command """
    flat: np.ndarray 
    kl: np.ndarray 
    lut: list[int]
    
    def get_flat(self)->np.ndarray:
        return self.flat 

    def get_kl(self)->np.ndarray:
        return self.kl

    def get_channel_number(self, actnum:int)->int:
        return self.lut[actnum]

    def get_cable_number(self, actnum:int)->int:
        return self.get_channel_number(actnum)//64 + 1

@dataclass
class DmCommand:
    prop: IDmProperty 
    calib: IDmCalibration
    
    def get_property(self)->IDmProperty:
        return self.prop 
    
    def get_calibration(self)->IDmCalibration:
        return self.calib 

    def actuator(self, act:int, amplitude:float)->np.ndarray:
        cmd = self.calib.get_flat().copy()
        cmd[act] += amplitude
        return cmd 

    def group(self, act:int, size: float, amplitude:float)->np.ndarray:
        amap = self.prop.pattern.as_screen( 
                 np.arange(self.prop.nact)
            ).astype(int)
        na = self.prop.pattern.mask.shape[0]
        x,y  = np.array(np.where(amap==act)).T[0]
        group = amap[max(x-size//2,0):min(x+size//2+1,na),
                     max(y-size//2,0):min(y+size//2+1,na)
                ].flatten()
        group = group[group>=0]
        cmd = self.calib.get_flat().copy()
        cmd[group] += amplitude
        return cmd
    
    def flat(self)->np.ndarray:
        return self.calib.get_flat().copy()

    def rest(self)->np.ndarray:
        return self.calib.get_flat() * 0.0 

    def center(self, amplitude:float)->np.ndarray:
        return self.actuator( self.prop.center, amplitude )

    def kl(self, mode:int, amplitude:float)->np.ndarray:
        return  self.calib.get_flat() + amplitude*self.calib.get_kl()[mode]

