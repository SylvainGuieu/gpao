from gpao.alpao43 import alpao43 
from gpao.tasks.dm_plug_test import PlugTestRunner, PlugTestScope
import sys
ip_loockup = {

   "BAX651": ('134.171.240.144', "134.171.240.145"),
   "BAX652": ('134.171.240.146', "134.171.240.147"), 
   "BAX653": ('134.171.240.148', "134.171.240.149"), 
   "BAX654": ('134.171.240.150', "134.171.240.151"), 
}


def main():
    serial = sys.argv[1]
    run(serial)

def run(serial, threshold:float = 0.004):
    dm = alpao43(serial, ips=ip_loockup[serial], use_flat=False)
    runner = PlugTestRunner(dm)
    plotter = PlugTestScope(warning_treshold=threshold)

    runner.run( plotter.plot_new )

if __name__ == '__main__':
    main() 
