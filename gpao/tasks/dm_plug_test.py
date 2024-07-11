# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 16:41:12 2024

@author: Instru
"""
from __future__ import annotations
from typing import Any, Callable
from gpao.alpao43 import alpao43_command
from gpao.core import core 
import numpy as np
from dataclasses import dataclass, field
import logging
from matplotlib.pylab import plt

from gpao.iplot import IAxesMaker
from gpao.plot import AxesMaker

ip_loockup = {
   "BAX655": ('134.171.240.148', "134.171.240.149"),  
   "BAX652": ('134.171.240.146', "134.171.240.147"), 
   "BAX651": ('134.171.240.144', "134.171.240.145"),
}


def get_ips(dmid:str)->tuple[str,str]:
    try:
        return ip_loockup[dmid]
    except KeyError:
        raise ValueError(f"Ip Adddress of {dmid} no found")

@dataclass 
class RunnerSetting:
    act_amp: float = 1.0
    pushlag: float = 0.00001
    baselag: float = 0.00001

@dataclass 
class Measurement:
    actnum: int = 0
    amplitude: float = 0.0
    current1: float = 0.0 
    current2: float = 0.0 
    cable_number: int = 0 
    @property 
    def current(self)->float:
        return self.current1+self.current2

@dataclass
class PlugTestRunner:
    dmcmd: core.IDmCommand
    hks: core.DmDeHks
    dmcom: core.IDmCom|None = None
    setting: RunnerSetting = RunnerSetting()
    
    log: logging.Logger = logging.getLogger("dm_plug_test")
    log.setLevel(logging.INFO)
    
    def run(self, callback:Callable[[Measurement], None] = lambda m:None): 
        if self.dmcom:
            dmcom= self.dmcom
        else:
            dmcom = core.new_com( self.dmcmd.get_property().serial_name )
        
        self.hks.connect()
        get_cable_number = self.dmcmd.get_calibration().get_cable_number
        try:
            self.hks.get_current()
            for act in range(self.dmcmd.get_property().nact):
                dmcom.send ( self.dmcmd.rest() )
                cb1, cb2 = self.hks.get_current()
                
                dmcom.send( self.dmcmd.actuator(act, self.setting.act_amp ) )
                c1, c2 = self.hks.get_current()
                    
                m = Measurement(act, self.setting.act_amp, 
                                current1 = c1-cb1, 
                                current2 = c2-cb2,
                                cable_number= get_cable_number(act)
                            )
                callback(m)
        finally:
            dmcom.send ( self.dmcmd.rest() )
            self.hks.disconnect()
    
@dataclass 
class Checker:
    warning_treshold: float = 0.004
    def check(self, measurement: Measurement)->None:
        if measurement.current< self.warning_treshold:
            print( "WARNING ", f"Weak current for {measurement.actnum}={measurement.current} A ; cable {measurement.cable_number}")
        print (f"{measurement.actnum}", end=" ")

@dataclass
class PlugTestScope:
    setting: RunnerSetting = RunnerSetting()
    warning_treshold: float = 0.004
    
    data: list[Measurement] = field(default_factory=list)
    axes_maker = IAxesMaker = AxesMaker()
    
    def init_figure(self):
        self.fig, self.ax = plt.subplots(1,1)
    
    def add(self, measurement:Measurement)->None:
        Checker(self.warning_treshold).check(measurement)
        self.data.append( measurement )
    
    def plot_new(self,  measurement:Measurement)->None:
        self.add( measurement )
        self.plot()
        plt.pause(0.001)
        
    def plot(self, nlast=100)->None:
        ax = self.axes_maker.get_axes() 

        ax.clear()
        data = self.data[-nlast:None]
        acton = [m.actnum  for m in data]
        c1 =  [m.current1 for m in data] 
        
        ax.plot(acton,  c1, 'k*' )       
        c2 =  [m.current2 for m in data]     
        ax.plot(acton,  c2, 'k+' )
        ax.set_xlabel("act")
        ax.set_ylabel("current [A]")

def plug_test_runner(serial_name:str, simulated:bool = False)->PlugTestRunner:
    dmcmd = alpao43_command(serial_name)
    if simulated:
        hks = core.DmDeHks( core.DmDeHkSim(), core.DmDeHkSim())
    else:
        ip1, ip2 = ip_loockup[serial_name]
        hks =  core.DmDeHks(core.DmDeHk(ip1), core.DmDeHk(ip2))

    return  PlugTestRunner(
            dmcmd=dmcmd, hks=hks, 
            dmcom=core.new_com(serial_name, simulated)
        )


if __name__ == "__main__":


    runner = plug_test_runner('BAX651', True)
    plotter = PlugTestScope()
    runner.run( plotter.plot_new )
    


    




    
