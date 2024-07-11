from __future__ import annotations
from dataclasses import dataclass
import json
import os
from typing import Callable
import scipy.io 
from gpao.core.iconfig import IConfigIo
import numpy as np 

_alpao_sdk_config = "C:/Program Files/Alpao/SDK/Config"
def get_config_root():
    return _alpao_sdk_config

def set_config_root(root:str):
    global _alpao_sdk_config
    _alpao_sdk_config = root

config_io_loockup: dict[str,IConfigIo] = {

}
config_ips_loockup: dict[str,tuple[str,...]] = {

}
def register_io(
        serial_name: str, 
        io: IConfigIo 
    ):
    config_io_loockup[serial_name] = io

def get_io(serial_name:str)->IConfigIo:
    try:
        return config_io_loockup[serial_name]
    except KeyError:
        raise ValueError(f"Cannot found mat file for {serial_name}")

def register_ips(serial_name: str , *ips):
    config_ips_loockup[serial_name] = ips 

def get_ips(serial_name: str)->tuple[str,...]:
    try:
        return config_ips_loockup[serial_name]
    except KeyError:
        raise ValueError(f"Cannot found ip addresses for {serial_name}")



def read_lut(path:str)->list[int]:
    with open( path ) as f:
        cfg = json.load( f) 
    return cfg['LUT']

@dataclass
class DmConfigIo:
    mat_file: str 
    lut_file: str
    root_getter: Callable[[],str] =  get_config_root
    
    def extract_flat(self, mat: dict)->np.ndarray:
        return mat['dataFlat'][0][0][1].flatten()

    def read_flat(self)->np.ndarray:
        file_path = os.path.join(self.root_getter(), self.mat_file)
        mat = scipy.io.loadmat( file_path )
        return self.extract_flat(mat)
    
    def read_lut(self)->list[int]:
        file_path = os.path.join(self.root_getter(), self.lut_file)
        return read_lut( file_path )
