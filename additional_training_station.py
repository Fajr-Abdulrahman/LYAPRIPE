import numpy as np 
import time
from lym1d.emulator_Nyx_george_backend import create_emulator
import h5py
import json
with open("supported_combinations.json", "r") as file:
    supported_combinations = json.load(file)




def additional_training(k_z_combination_list_to_train):

    # import the converted data for the training
    file_path = 'Planck_10_sigma_essentials.hdf5'
    with h5py.File(file_path, 'r') as hdf:
        loaded_data = {key: hdf[key][:] for key in hdf.keys()}
    allparams_essentials = loaded_data
    
    file_path = 'Planck_10_sigma_extras.hdf5'
    with h5py.File(file_path, 'r') as hdf:
        loaded_data = {key: hdf[key][:] for key in hdf.keys()}
    allparams_extras = loaded_data

    
    k_z_combination_list = np.array(k_z_combination_list_to_train, dtype=object) 
    avilable_k_piv = allparams_extras["k_pivots"]
    avilable_units =  np.array([str(allparams_extras["units"][i],'utf-8') for i in range(len(allparams_extras["units"]))])
    avilable_z_piv = allparams_extras["z_pivots"]
    
    k_z_combination_list_avilable = np.array([avilable_k_piv,avilable_units,avilable_z_piv],dtype=object).T
    
    
    
    # Empty list to be filled with data after the training
    
    emu_list_to_alphaDn_lya = np.zeros(len(k_z_combination_list), dtype= object)
    emupars_list_to_alphaDn_lya = np.zeros(len(k_z_combination_list), dtype=object)
    k_piv_list = np.zeros(len(k_z_combination_list), dtype=object)
    units_list = np.zeros(len(k_z_combination_list), dtype=object)
    z_piv_list = np.zeros(len(k_z_combination_list), dtype=object)
    
    emu_list_to_omega_mlnAn_s =  np.zeros(len(k_z_combination_list), dtype=object)
    emupars_list_to_omega_mlnAn_s =  np.zeros(len(k_z_combination_list), dtype=object)
    
    # Import number of training samples. It takes automatically the number given in the json-file.
    nsamples = supported_combinations["number_of_training_samples"]
    
    # Check if one of the two given combinations by the user is already avilable and ignore it. If both are not avilable, both will be emulated
    list_of_index = []
    for i,combination in enumerate(k_z_combination_list_avilable):
        for combination_to_train in k_z_combination_list:
            if np.array_equal(combination, combination_to_train):
                list_of_index.append(i)
    if not list_of_index:
        time.sleep(2)
        return additional_training(k_z_combination_list_to_train=k_z_combination_list_to_train)

    true_alpha_lya = np.zeros((nsamples,len(k_z_combination_list)))
    true_D_lya = np.zeros((nsamples,len(k_z_combination_list)))
    true_n_lya = np.zeros((nsamples,len(k_z_combination_list)))
    
    for i,j in enumerate(list_of_index):
        true_alpha_lya[:,i] = allparams_extras["alpha_lya"][:,j]
        true_D_lya[:,i] = allparams_essentials["D(k_pivot,z_pivot)"][:,j]
        true_n_lya[:,i] = allparams_essentials["n(k_pivot,z_pivot)"][:,j]
    
    
    
    # Emulation direction from primordial to lya
    inpars = np.array([allparams_essentials["H0"][:nsamples], allparams_essentials["omega_m"][:nsamples],allparams_essentials["ln_A_s_1e10"][:nsamples], allparams_essentials["n_s"][:nsamples]])
    for i in range(len(k_z_combination_list)):
        outpars = np.array([ true_alpha_lya[:,i], true_D_lya[:,i], true_n_lya[:,i]])
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
    # load old data and append the new ones to it
    data_to_lya= np.load("from_primordial_to_lya.npz", allow_pickle=True)
    
    emu_list_to_lya = np.append(data_to_lya["emu_to_alphaDn_lya"], emu_list_to_alphaDn_lya)
    emupars_to_lya = np.append(data_to_lya["emupars_to_alphaDn_lya"], emupars_list_to_alphaDn_lya)
    k_piv_list = np.append(data_to_lya["k_piv"], k_piv_list)
    units_list = np.append(data_to_lya["units"], units_list)
    z_piv_list = np.append(data_to_lya["z_piv"], z_piv_list)

    # save the data
    np.savez_compressed("from_primordial_to_lya.npz", emu_to_alphaDn_lya=emu_list_to_lya, emupars_to_alphaDn_lya = emupars_to_lya, parnames=emuparnames, k_piv = k_piv_list, units = units_list, z_piv = z_piv_list, allow_pickle=True)
    
    
    
    # Emulation direction from lya to primordial
    outpars = np.array([allparams_essentials["omega_m"][:nsamples],allparams_essentials["ln_A_s_1e10"][:nsamples], allparams_essentials["n_s"][:nsamples]])
    for i in range(len(k_z_combination_list)):
        inpars = np.array([ allparams_essentials["H0"][:nsamples], true_alpha_lya[:,i], true_D_lya[:,i], true_n_lya[:,i]])
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
    # load old data and append the new ones to it
    data_to_primordial= np.load("from_lya_to_primordial.npz", allow_pickle=True)
    emu_list_to_primodial = np.append(data_to_primordial["emu_to_omega_mlnAn_s"], emu_list_to_omega_mlnAn_s)
    emupars_to_primodial = np.append(data_to_primordial["emupars_to_omega_mlnAn_s"], emupars_list_to_omega_mlnAn_s)

    # save the data
    np.savez_compressed("from_lya_to_primordial.npz", emu_to_omega_mlnAn_s=emu_list_to_primodial, emupars_to_omega_mlnAn_s = emupars_to_primodial, parnames=emuparnames, k_piv = k_piv_list, units = units_list, z_piv = z_piv_list, allow_pickle=True)
    

