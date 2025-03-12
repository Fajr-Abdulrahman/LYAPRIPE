import Emulator
from astropy.table import QTable, Table, Column
from astropy import units as u
import argparse
import numpy as np

emu_func = Emulator.flags_to_right_order


show_pred = None
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emulation between the cosmological parameter sets")
    parser.add_argument("--root_name",type = str, help = 'Here, you put the path together with the root name of the chain you want to emulate. This chain should have the conventional form of a chain (that can be read by getdist for example)')

    parser.add_argument("--in_set",type = str,help = 'Write the parameter set for the input data. Choose between "primordial" and "lya"')
    parser.add_argument("--z_in", type = float,nargs='?', const=None, default=None,  help = 'redshift of your input parameter if they are lya parameters')
    parser.add_argument("--unit_in", type=str,nargs='?', const=None, default=None,  help = 'unit of the wave number of your input parameter if they are lya parameters. The possible units are "kms" and "Mpc".')
    parser.add_argument("--k_in", type=float,nargs='?', const=None, default=None,  help = 'value of the wave number of your input parameter if they are lya parameters')
    
    parser.add_argument("--out_set",type= str, help = 'Write the parameter set for the output data. Choose between "primordial" and "lya"')
    parser.add_argument("--z_out", type=float, nargs='?', const=None, default=None, help = 'redshift of your output parameter if they are lya parameters')
    parser.add_argument("--unit_out", type=str,nargs='?', const=None, default=None,  help = 'unit of the wave number of your output parameter if they are lya parameters. The possible units are "kms" and "Mpc".')
    parser.add_argument("--k_out", type= float, nargs='?', const=None, default=None, help = 'value of the wave number of your output parameter if they are lya parameters')
    parser.add_argument("--compute", action="store_true", help = 'Using this flag starts the computation of new combinations for k_piv and z_piv if the chosen z_piv is supported. Otherwise write the wished z_piv value in the json-file and run mPk_computing_file.py then you can you can come back here and activate this flag.')
    
    args = parser.parse_args()
    print(args)
    
    show_pred=emu_func(root_name=args.root_name, in_set=args.in_set, z_in=args.z_in, 
                            unit_in=args.unit_in,k_in=args.k_in,out_set=args.out_set, z_out=args.z_out, unit_out=args.unit_out, k_out=args.k_out, compute=args.compute)
    

