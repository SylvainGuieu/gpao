
from gpao.core.iconfig import ( 
    IConfigIo as IConfigIo
)

from gpao.core.config import (
    DmConfigIo as DmConfigIo, 
    register_io as register_io, 
    get_io as get_io, 
    get_config_root as get_config_root, 
    set_config_root as set_config_root, 
    read_lut as read_lut
)
from gpao.core.imodel import ( 
    IDmPattern as IDmPattern
)
from gpao.core.model import( 
    DmPattern as DmPattern, 
    RegionDisk as RegionDisk, 
    Regions as Regions, 
    build_pattern as build_pattern, 
)
from gpao.core.idm import (
    IDmProperty as IDmProperty, 
    IDmCommand as IDmCommand, 
    IDmCalibration as IDmCalibration, 
)

from gpao.core.dm import ( 
    DmProperty as DmProperty, 
    DmCalib as DmCalib, 
    DmCommand as DmCommand, 
)

from gpao.core.idmcom import (
    IDmCom as IDmCom
)

from gpao.core.ihk import (
    IDmDeHk as IDmDeHk, 
    IDmDeHks as IDmDeHks, 
    HkData as HkData,
)

from gpao.core.hk import( 
    DmDeHk as DmDeHk, 
    DmDeHks as DmDeHks, 
    DmDeHkSim as DmDeHkSim,
)

from gpao.core.dmcomsim import (
    DmComSim as DmComSim
)

try:
    from gpao.core.dmcom import (
        DmCom as DmCom
    )
except ModuleNotFoundError:
    _AS_SDK: bool = False 
else:
    _AS_SDK = True 

def new_com(serial_name:str, simulated:bool = False)->IDmCom:
    if simulated:
        return DmComSim(serial_name) 
    else:
        if not _AS_SDK:
            raise ModuleNotFoundError("ALPAO SDK package is not found. please install or use simulate=True")
        return DmCom(serial_name) # type: ignore # see check above
