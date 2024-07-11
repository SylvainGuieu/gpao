from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from gpao.core.idm import IDmCalibration, IDmCommand, IDmProperty
from gpao.core.idmcom import IDmCom
from gpao.core.ihk import HkData, IDmDeHks
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


@dataclass
class Dm:
    cmd: IDmCommand
    com: IDmCom
    hks: IDmDeHks| None = None

    def get_property(self)->IDmProperty:
        return self.cmd.get_property()

    def get_calibration(self)->IDmCalibration:
        return self.cmd.get_calibration()

    def send_actuator(self,  act:int, amplitude:float)->None:
        self.com.send( self.cmd.actuator(act, amplitude))

    def send_group(self, act:int, size:float, amplitude:float)->None:
        self.com.send( self.cmd.group(act, size, amplitude) ) 

    def send_flat(self)->None:
        self.com.send( self.cmd.flat() ) 

    def send_rest(self)->None:
        self.com.send( self.cmd.rest() ) 
    
    def reset(self)->None:
        self.send_rest()

    def send_kl(self, mode:int, amplitude:float)->None:
        self.com.send( self.cmd.kl( mode, amplitude) )
    
    def get_hk_data(self)->tuple[HkData, ...]:
        if self.hks is None:
            raise ValueError("This instance has no DM House keeping connection")
        return self.hks.get_hk_data()

    def get_current(self)->tuple[float,...]:
        if self.hks is None:
            raise ValueError("This instance has no DM House keeping connection")
        return self.hks.get_current() 
        
