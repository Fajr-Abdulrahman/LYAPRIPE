import numpy as np 

from lym1d.emulator_Nyx_george_backend import create_emulator
import json
import h5py
import argparse




# import the converted data for the training
file_path = 'Planck_10_sigma_essentials.hdf5'
with h5py.File(file_path, 'r') as hdf:
    loaded_data = {key: hdf[key][:] for key in hdf.keys()}
allparams_essentials = loaded_data

file_path = 'Planck_10_sigma_extras.hdf5'
with h5py.File(file_path, 'r') as hdf:
    loaded_data = {key: hdf[key][:] for key in hdf.keys()}
allparams_extras = loaded_data


with open("supported_combinations.json", "r") as file:
    supported_combinations = json.load(file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Here you can train different emulation functions for the supported k_piv and z_piv combinations.")
    parser.add_argument("--training_samples", type = int, default=supported_combinations["number_of_training_samples"],  help = 'number of the samples used to train the emulator.')
    args = parser.parse_args()
    


print("Starting the training...")

k_piv = allparams_extras["k_pivots"]
units = np.array([str(allparams_extras["units"][i],'utf-8') for i in range(len(allparams_extras["units"]))])
z_piv = allparams_extras["z_pivots"]

k_z_combination_list = np.array([k_piv, units,z_piv], dtype=object).T

# Empty list to be filled with data after the training
emu_list_to_alphaDn_lya = np.zeros(len(k_z_combination_list), dtype= object)
emupars_list_to_alphaDn_lya = np.zeros(len(k_z_combination_list), dtype=object)
k_piv_list = np.zeros(len(k_z_combination_list), dtype=object)
units_list = np.zeros(len(k_z_combination_list), dtype=object)
z_piv_list = np.zeros(len(k_z_combination_list), dtype=object)

emu_list_to_omega_mlnAn_s =  np.zeros(len(k_z_combination_list), dtype=object)
emupars_list_to_omega_mlnAn_s =  np.zeros(len(k_z_combination_list), dtype=object)

# Import number of training samples if not specified
nsamples = args.training_samples


# Emulation direction from primordial to lya
inpars = np.array([allparams_essentials["H0"][:nsamples], allparams_essentials["omega_m"][:nsamples],allparams_essentials["ln_A_s_1e10"][:nsamples], allparams_essentials["n_s"][:nsamples]])
for i in range(len(k_z_combination_list)):
    outpars = np.array([ allparams_extras["alpha_lya"][:nsamples,i], allparams_essentials["D(k_pivot,z_pivot)"][:nsamples,i], allparams_essentials["n(k_pivot,z_pivot)"][:nsamples,i]])
    smooth_lengths = np.array(5*np.std(inpars, axis=1))
    emu, update_emu, emupars, emuparnames = create_emulator(  
        inpars.T,
        outpars.T,
        smooth_lengths,
        noise=(1e-3 if not False else None), 
        npc=70, 
        optimize=False,
        output_cov=False,
            
        sigma_0=np.sqrt(
            1
        ),
        #sigma_l=np.sqrt(0.1),
        noPCA=True, 
        kerneltype="SE" if not (False) else "M52", 
    )
    emu_list_to_alphaDn_lya[i]=emu 
    emupars_list_to_alphaDn_lya[i] = emupars
    k_piv_list[i] = k_z_combination_list[i,0]
    units_list[i] = k_z_combination_list[i ,1]
    z_piv_list[i] = k_z_combination_list[i,2]
# save trained data
np.savez_compressed("from_primordial_to_lya.npz", emu_to_alphaDn_lya=emu_list_to_alphaDn_lya, emupars_to_alphaDn_lya = emupars_list_to_alphaDn_lya, parnames=emuparnames, k_piv = k_piv_list, units = units_list, z_piv = z_piv_list, allow_pickle=True)




# Emulation direction from lya to primordial
outpars = np.array([allparams_essentials["omega_m"][:nsamples],allparams_essentials["ln_A_s_1e10"][:nsamples], allparams_essentials["n_s"][:nsamples]])
for i in range(len(k_z_combination_list)):
    inpars = np.array([ allparams_essentials["H0"][:nsamples], allparams_extras["alpha_lya"][:nsamples,i], allparams_essentials["D(k_pivot,z_pivot)"][:nsamples,i], allparams_essentials["n(k_pivot,z_pivot)"][:nsamples,i]])
    smooth_lengths = np.array(5*np.std(inpars, axis=1))
    emu, update_emu, emupars, emuparnames = create_emulator(  
        inpars.T,
        outpars.T,
        smooth_lengths,
        noise=(1e-3 if not False else None), 
        npc=70, 
        optimize=False,
        output_cov=False,
            
        sigma_0=np.sqrt(
            1
        ),
        #sigma_l=np.sqrt(0.1),
        noPCA=True, 
        kerneltype="SE" if not (False) else "M52", 
    )
    emu_list_to_omega_mlnAn_s[i]=emu 
    emupars_list_to_omega_mlnAn_s[i] = emupars
    k_piv_list[i] = k_z_combination_list[i,0]
    units_list[i] = k_z_combination_list[i ,1]
    z_piv_list[i] = k_z_combination_list[i,2]
# save trained data
np.savez_compressed("from_lya_to_primordial.npz", emu_to_omega_mlnAn_s=emu_list_to_omega_mlnAn_s, emupars_to_omega_mlnAn_s = emupars_list_to_omega_mlnAn_s, parnames=emuparnames, k_piv = k_piv_list, units = units_list, z_piv = z_piv_list, allow_pickle=True)

