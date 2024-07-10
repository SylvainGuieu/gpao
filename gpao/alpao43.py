from __future__ import annotations
from gpao.core import core 
from gpao.jbmodel import compute_KL_GPAO
import numpy as np
import importlib_resources


class BAX651IO(core.DmConfigIo): # TODO: Why ?
    def extract_flat(self, mat: dict) -> np.ndarray:
        cmd = mat['dataFlat20'][0][0][1].T.flatten()
        pattern = alpao43_dm_pattern()
        return cmd[pattern.mask.flat]

core.register_io('BAX651', 
                 BAX651IO('BAX651_Flat_T20.mat', 'BAX651.json')
                 )
core.register_io('BAX652', 
                 core.DmConfigIo('BAX652_Flat_T20.mat', 'BAX652.json')
                 )
core.register_io('BAX653', 
                 core.DmConfigIo('BAX653_Flat_T20.mat', 'BAX653.json')
                 )
core.register_io('BAX654', 
                 core.DmConfigIo('BAX654_Flat_T20.mat', 'BAX654.json')
                 )
core.register_io('BAX655', 
                 core.DmConfigIo('BAX655_Flat_T20.mat', 'BAX655.json')
                 )

def alpao43_dm_pattern(pitch:float=2.62)->core.DmPattern:
    pattern = core.build_pattern(
            43, pitch,  
            core.RegionDisk(0.0,0.0, 55.8*pitch/2.62), 
            exclude=core.RegionDisk(-55.0,2.6,pitch/4.0) # one actuator missing
        ) 
    pattern.name = "Alpao43"
    return pattern

def alpao43_property(serial_name:str)->core.IDmProperty:
    """ Build and return a DmProperty object for ALPAO 43 """
    return core.DmProperty(serial_name,
                      nact= 1432, 
                      center= 716,
                      pitch= 2.62e-3 , 
                      pixel_scale= 0.139, 
                      act_box= (43,43),  
                      pattern = alpao43_dm_pattern()
            ) 

def alpao43_default_calib(dm_property: core.IDmProperty)->core.DmCalib:
    flat = np.zeros( (dm_property.nact,) , np.float32)
    kl: np.ndarray = compute_KL_GPAO(nact=dm_property.act_box[0], pup_fraction=1.0) # type:ignore # mixed return 
    
    lutpath = importlib_resources.files('gpao').joinpath('resources/DM43.json')
    lut: list[int] = core.read_lut(str(lutpath))
    return core.DmCalib(flat, kl.T, lut=lut) 

def alpao43_calib(dm_property: core.IDmProperty)->core.DmCalib:
    io = core.get_io( dm_property.serial_name )
    flat = io.read_flat()
    kl : np.ndarray= compute_KL_GPAO(nact=dm_property.act_box[0], pup_fraction=1.0) # type:ignore #mixed return 
    return core.DmCalib(flat=flat, kl=kl.T, lut=io.read_lut())

def alpao43_command(serial_name:str, use_flat:bool = True)->core.DmCommand:
    prop = alpao43_property(serial_name)
    calib = alpao43_calib(prop)
    if not use_flat:
        calib.flat *= 0.0
    return core.DmCommand(prop, calib)

if __name__ == "__main__":
    print( alpao43_default_calib(alpao43_property('B')))
