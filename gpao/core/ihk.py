from dataclasses import dataclass, field
from typing import Protocol

@dataclass
class Hsdl:
    count: int = 0
    crc: int = 0 
    error: str = "NO"

@dataclass
class ErrorFlag:
    addr_dac: str = 'OK'
    amp: str = 'OK'
    deTemp: str = 'OK'
    dmTemp: str = 'OK'
    global_: str = 'OK'
    high_speed: str = 'OK'
    load_dac: str = 'OK'
    negCurrent: str = 'OK'
    posCurrent: str = 'OK'
    power: str = 'OK'
    watchdog:str = 'OK'
    
@dataclass
class Status:
    crc:str = 'OK'
    error:str = 'NO'
    error_flag: ErrorFlag = field(default_factory=ErrorFlag)
    power: str = 'OFF'
    version: str = 'OK'
    wd: str = 'ALIVE'

@dataclass
class HkData:
    boxTemp: int = 32 
    deTemp: float = 0.0 
    dmTemp: float = 0.0 
    hsdl: Hsdl = field(default_factory=Hsdl)
    outputs: int = 12
    power: float = 0.0 
    status: Status = field(default_factory=Status)
    version: str = ''


class IDmDeHk(Protocol):
    def connect(self)->None:
        ...

    def disconnect(self)->None:
        ...
    
    def get_current(self)->float:
        ... 

    def get_hk_data(self)->HkData:
        ... 

class IDmDeHks(Protocol):
    def connect(self)->None:
        ...

    def disconnect(self)->None:
        ...
   
    def get_current(self)->tuple[float, ...]:    
        ...
    
    def get_hk_data(self)->tuple[HkData, ...]:
        ...
   
