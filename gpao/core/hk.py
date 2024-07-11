from dataclasses import dataclass, field
import json
import socket
from typing import NamedTuple
import requests
import math 
from gpao.core.ihk import IDmDeHk, HkData

SSH_PORT = 1665
HTTP_PORT = 1666
BUFFER_SIZE = 256*2
CURENT_COMMAND = b"CURRENT\0\0\0\0\0\0\0\0\0"


class HkResponse(NamedTuple):
    msg: str 
    response: str 

def hkl_cmd(sockid: socket.socket, msg:bytes)->HkResponse:    
    sockid.send(msg) 
    data = sockid.recv(BUFFER_SIZE) 
    # Decode response (simple UTF-8 conversion as response is human readable)   
    #  Further  and  more  specific  decoding  may  be  implemented  depending  on message   
    response = data.decode('utf-8');   
    response = response.replace('\x00', '') 
    return HkResponse(msg.decode('utf-8').replace('\x00', ''), response)

def _extract_current(s: str):
    return float( s.replace('CURRENT', '').replace(' A','') )

@dataclass
class Connector:
    hk: IDmDeHk
    was_connected: bool = False 
    def __enter__(self):
        if self.hk.is_connected() :
            self.was_connected = True 
        else:
            self.was_connected = False 
            self.hk.connect()
        return self.hk 
    
    def __exit__(self, *args)->bool:
        if not self.was_connected:
            self.hk.disconnect()
        return False
        
        


@dataclass
class DmDeHk:
    dmdeip: str
    socket_instance: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _connected: bool = False
    
    def is_connected(self)->bool:
        return self._connected 

    def connect(self)->None:
        self.socket_instance.close()
        self.socket_instance  = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_instance.connect( (self.dmdeip, SSH_PORT) )
        self._connected = True
    
    def disconnect(self)->None:
        self.socket_instance.close()
        self._connected = False

    def get_current(self)->float:
        with Connector(self):
            r = hkl_cmd(self.socket_instance, CURENT_COMMAND)
            return _extract_current(r.response)
        return math.nan

    def get_hk_data(self)->HkData:
        f = requests.get(f"http://{self.dmdeip}:{HTTP_PORT}")
        data = json.loads(f.text.replace('global', 'global_'))
        return HkData(**data)
    
    def __enter__(self):
        self.connect()
        return self 

    def __exit__(self,*args):
        self.disconnect()

@dataclass
class DmDeHkSim:

    def is_connected(self)->bool:
        return True 

    def connect(self)->None:
        pass
    
    def disconnect(self)->None:
        pass 

    def get_current(self)->float:
        return 0.0

    def get_hk_data(self)->HkData:
        return HkData()
    
    def __enter__(self):
        self.connect()
        return self 
    def __exit__(self,*args):
        self.disconnect()


@dataclass(init=False)
class DmDeHks:
    dehks: tuple[IDmDeHk,...] 
    def __init__(self, *dmdehks:IDmDeHk):
        self.dehks = tuple(dmdehks)
        
    def connect(self):
        for dehk in self.dehks:
            dehk.connect( )
    
    def disconnect(self):
        for dehk in self.dehks:
            dehk.disconnect( )

    def get_current(self)->tuple[float, ...]:
        return tuple( dehk.get_current() for dehk in self.dehks)
        
    def get_hk_data(self)->tuple[HkData, ...]:
        return tuple( dehk.get_hk_data() for dehk in self.dehks)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self,*args):
        self.disconnect()

def house_keepings(*ips:str, simulated:bool= False)->DmDeHks:
    if simulated:
        return  DmDeHks( *(DmDeHkSim() for _ in ips) )
    else:
        return DmDeHks( *(DmDeHk(ip) for ip in ips) )
