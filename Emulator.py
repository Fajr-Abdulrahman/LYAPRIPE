from getdist import loadMCSamples
import glob
import h5py
import os
import numpy as np
from astropy.table import QTable, Table, Column
from astropy import units as u
import json
import additional_training_station
import additional_converter
with open("supported_combinations.json", "r") as file:
    supported_combinations = json.load(file)
supported_combinations=supported_combinations


def conversion_load(file_path  ):
    # Load the converted combinations in order to check their avilability
    with h5py.File(file_path, 'r') as hdf:
        loaded_data = {key: hdf[key][:] for key in hdf.keys()}
    convert_load = loaded_data
    convert_k_piv= convert_load["k_pivots"]
    convert_units= np.array([str(convert_load["units"][i],'utf-8') for i in range(len(convert_load["units"]))])
    convert_z_piv= convert_load["z_pivots"]
    convert_k_z_combination_list = np.array([convert_k_piv, convert_units, convert_z_piv], dtype=object).T
    return { 'k_piv':convert_k_piv, 'units': convert_units, 'z_piv': convert_z_piv, 'k_z_combination_list': convert_k_z_combination_list}

def training_load(file_path ):
    # Load the training combinations to check their avilability
    training_load = np.load(file_path, allow_pickle=True)
    k_piv = training_load["k_piv"]
    units = training_load["units"]
    z_piv = training_load["z_piv"]
    
    k_z_combination_list = np.array([k_piv, units, z_piv], dtype = object).T
    return {'k_piv':k_piv, 'units': units, 'z_piv':z_piv, 'k_z_combination_list':k_z_combination_list}


def validate_lya_parameters(set_type, k, unit, z, valid_k, valid_units, valid_z):
    # Validate if the chosen combinations of k_piv and z_piv are new or already avilable
    if set_type == "lya":
        if k is None or unit is None or z is None:
            raise ValueError(f"Missing parameters for {set_type} set.")
        if k not in valid_k:
            print(f"New k value detected for {set_type} set: {k}.")
        if unit not in valid_units:
            print(f"New unit detected for {set_type} set: {unit}.")
        if z not in valid_z:
            print(f"New z value detected for {set_type} set: {z}.")
    
    elif set_type == "primordial" and any(param is not None for param in [k, unit, z]):
        raise ValueError(f"No additional parameters allowed for {set_type} set.")


def combination_check(set_type, k, unit, z, allowed_combination_list_1, allowed_combination_list_2):
    # Check if the chosen combinations are new and give them a status 
    combination = np.array([k,unit,z], dtype=object)
    status= False
    if set_type == "lya":
        if not any(np.array_equal(combination, allowed_combination_list_1[i, :]) for i in range(len(allowed_combination_list_1))) or not any(np.array_equal(combination, allowed_combination_list_2[i, :]) for i in range(len(allowed_combination_list_2))):
            status= True
        else:
            status= False
    return status

def data_order(data, in_set):
    # Put the parameters in the right order to be computed
    if in_set == "primordial":
        if 'omega_m' in data.colnames:
            omega_m_list = data['omega_m']
        elif 'omega_b' in data.colnames and 'omega_cdm' in data.colnames:
            omega_m_list = data['omega_cdm'] + data['omega_b']
        else:
            omega_m_list = data['Omega_m'] *(data['H0']/100)**2
        
        data_result = np.vstack([data['H0'], omega_m_list, data['ln10^{10}A_s'], data['n_s']]).T
    elif in_set == "lya":
        data_result = np.vstack([data['H0'], data['alpha_lya'], data['Delta^2'], data['n_lya']]).T
    return data_result


def save_names(data_path = None, nametab = None, in_set=None, z_in=None, unit_in=None, k_in=None,
                out_set=None, z_out=None, unit_out=None, k_out=None):
    # Save the name of parameters in a .paramnames file
    paramnames = list(nametab['name'])
    paramnames_latex = list(nametab['texname'])
    def add_param(name, latex):
        if name not in paramnames:
            paramnames.append(name)
            paramnames_latex.append(latex)
    
    if in_set == "primordial" and out_set =="lya":
        add_param(f"alpha_lya(k={k_out} {unit_out}, z={z_out})",rf"\alpha_{{\text{{Ly}}\alpha}}(k={k_out} {unit_out}, z={z_out})")
        add_param(f"Delta^2(k={k_out} {unit_out}, z={z_out})",rf"\Delta^2(k={k_out} {unit_out}, z={z_out})")
        add_param(f"n_lya(k={k_out} {unit_out}, z={z_out})",rf"n_{{\text{{Ly}}\alpha}}(k={k_out} {unit_out}, z={z_out})")
                
    if in_set == "lya" and out_set == "lya":
        add_param(f"alpha_lya(k={k_out} {unit_out}, z={z_out})",rf"\alpha_{{\text{{Ly}}\alpha}}(k={k_out} {unit_out}, z={z_out})")
        add_param(f"Delta^2(k={k_out} {unit_out}, z={z_out})",rf"\Delta^2(k={k_out} {unit_out}, z={z_out})")
        add_param(f"n_lya(k={k_out} {unit_out}, z={z_out})",rf"n_{{\text{{Ly}}\alpha}}(k={k_out} {unit_out}, z={z_out})")
    if in_set =="lya" and out_set =="primordial":
        add_param("omega_m", r"\omega_{m}")
        add_param("ln10^{10}A_s", r"ln(10^{10} A_{s})")
        add_param("n_s", r"n_{s}")
    paramnames_combined = Table([paramnames, paramnames_latex], names=["name", "texname"])
    paramnames_combined.write(f"{data_path}.paramnames", format="ascii.no_header", delimiter="\t", overwrite=True)


def save_func(file_name = None, data_chain=None, emulation = None, in_set=None, z_in=None, unit_in=None, k_in=None,
                out_set=None, z_out=None, unit_out=None, k_out=None):
    # Save the data of the chain in .txt files
    chain = data_chain
    if in_set == "primordial" and out_set =="lya":

        chain[f"alpha_lya(k={k_out} {unit_out}, z={z_out})"] = emulation[:,0]
        chain[f"Delta^2(k={k_out} {unit_out}, z={z_out})"] = emulation[:,1]
        chain[f"n_lya(k={k_out} {unit_out}, z={z_out})"] = emulation[:,2]
                
    if in_set == "lya" and out_set == "lya":

        chain[f"alpha_lya(k={k_out} {unit_out}, z={z_out})"] = emulation[:,0]
        chain[f"Delta^2(k={k_out} {unit_out}, z={z_out})"] = emulation[:,1]
        chain[f"n_lya(k={k_out} {unit_out}, z={z_out})"] = emulation[:,2]

    if in_set =="lya" and out_set =="primordial":
        chain["omega_m"] = emulation[:,0]
        chain["ln10^{10}A_s"] = emulation[:,1]
        chain["n_s"] = emulation[:,2]

    chain.write(f'{file_name}', format="ascii.no_header", delimiter="\t", overwrite=True)


def flags_to_right_order(
        root_name=None, in_set=None, z_in=None, unit_in=None, k_in=None,
        out_set=None, z_out=None, unit_out=None, k_out=None, compute=False):
    
    # This function does the main job. It activate new training if needed, import emulation functions and emulate the data.
    
    # Load avilable combinations
    training_data = training_load(file_path = "from_primordial_to_lya.npz")
    conversion_data = conversion_load( file_path = 'Planck_10_sigma_extras.hdf5')
    k_z_combination_list = training_data["k_z_combination_list"]
    convert_k_z_combination_list = conversion_data["k_z_combination_list"]

    # Default values
    defaults = {
        "in_set": "primordial",
        "out_set": "lya",
        "z_out": 3.0,
        "unit_out": "kms",
        "k_out": 0.009
    }


    
        # Assign defaults if no inputs except are provided
    if all(param is None for param in [root_name, in_set, z_in, unit_in, k_in, out_set, z_out, unit_out, k_out]):
        raise ValueError("No constraints were given. Give at least root_name for default emulation.")

    # for the case that root_name isn't given but constraints are given
    if root_name is None and any(param is not None for param in [ in_set, z_in, unit_in, k_in, out_set, z_out, unit_out, k_out]):
        raise ValueError("No root_name was given. Give root_name and further constraints.")
        
    # Assign defaults if no inputs except for the root_name are provided
    if all(param is None for param in [ in_set, z_in, unit_in, k_in, out_set, z_out, unit_out, k_out]):
        print("No constraints were given. Using default emulation direction...")
        in_set, out_set = defaults["in_set"], defaults["out_set"]
        z_out, unit_out, k_out = defaults["z_out"], defaults["unit_out"], defaults["k_out"]

    # Validate input sets
    if in_set is None:
        raise ValueError("No in_set provided.")
    if out_set is None:
        raise ValueError("No out_set provided.")

    # Validate lya parameters
    validate_lya_parameters(in_set, k_in, unit_in, z_in, training_data["k_piv"], training_data["units"], training_data["z_piv"])
    validate_lya_parameters(out_set, k_out, unit_out, z_out, training_data["k_piv"], training_data["units"], training_data["z_piv"])



    # combination check
    check_in = combination_check(in_set, k_in, unit_in, z_in, k_z_combination_list, convert_k_z_combination_list)
    check_out = combination_check(out_set, k_out, unit_out, z_out, k_z_combination_list, convert_k_z_combination_list)

    # Here we check if there combination of the lya parameters given is supported
    if any(anomaly  for anomaly in [check_in,check_out]):
        if not check_in and check_out: 
            compute_k_piv,compute_units,compute_z_piv= k_out, unit_out, z_out
        elif check_in and not check_out:
            compute_k_piv,compute_units,compute_z_piv= k_in, unit_in, z_in
        elif check_in  and check_out:
            compute_k_piv,compute_units,compute_z_piv= np.array([k_in,k_out],dtype=object), np.array([unit_in,unit_out],dtype=object), np.array([z_in,z_out],dtype=object)


        if compute==False:
            raise ValueError(
                "The chosen combination for lya parameters (for in_set or out_set) is not supported. "
                "Please choose another combination. If you want to compute this combination, please activate the --compute flag "
                "and take into attention that the computation may take long."
            )
        # If the flage --compute was activated the computation for the chosen combination starts.
        elif compute==True:
            print("Proceeding with the computation for the chosen combination...")
            
            k_z_combination_list_to_add = np.vstack([compute_k_piv, compute_units,compute_z_piv],dtype=object).T 
            list_to_convert = []
            list_to_train = []   
            
            # Check if the combination was already converted
            for i in range(len(k_z_combination_list_to_add)):
                combination = k_z_combination_list_to_add[i,:]
                if not any(np.array_equal(combination, k_z_combination_list[j,:]) for j in range(len(k_z_combination_list))):
                    if any(np.array_equal(combination, convert_k_z_combination_list[j,:]) for j in range(len(convert_k_z_combination_list))):
                        list_to_train.append(combination)
                    else:
                        list_to_convert.append(combination)
            
            # If the combination is not converted yet, the conversion starts
            if list_to_convert:
                list_to_convert = np.array(list_to_convert, dtype=object)
                print("Starting the conversion process...")
                additional_converter.additional_converter(list_to_convert)

            # Train converted data
            list_to_convert = np.array(list_to_convert,dtype=object)
            list_to_train = np.array(list_to_train,dtype=object)
            all_combinations_to_train = None
            if list_to_convert.size == 0:  
                all_combinations_to_train = np.array(list_to_train, dtype=object)
            elif list_to_train.size == 0:  
                all_combinations_to_train = np.array(list_to_convert, dtype=object)
            else:  
                all_combinations_to_train = np.vstack([list_to_convert, list_to_train], dtype=object)

            print("Starting the training...")
            additional_training_station.additional_training(all_combinations_to_train)

            # appending new combination to old ones and saving them       
            supported_combinations_array= np.array([supported_combinations["k_piv"], supported_combinations["units"], supported_combinations["z_piv"]], dtype=object).T
            for i in range(len(all_combinations_to_train)): 
                combination_to_train = all_combinations_to_train[i,:]
                if not any (np.array_equal(combination_to_train, supported_combinations_array[j,:]) for j in range(len(supported_combinations_array))):
                    supported_combinations["k_piv"].append(combination_to_train[0])
                    supported_combinations["units"].append(combination_to_train[1])
                    supported_combinations["z_piv"].append(combination_to_train[2])
            
            new_dict = {"number_of_training_samples": supported_combinations["number_of_training_samples"],"k_piv": supported_combinations["k_piv"], 
                        "units": supported_combinations["units"],"z_piv": supported_combinations["z_piv"], 
                        "z_piv_for_computing_mPk": supported_combinations["z_piv_for_computing_mPk"]}
            with open("supported_combinations.json", "w") as file:
                json.dump(new_dict, file, indent=4)
            # Run the function again 
            return flags_to_right_order(
                root_name=root_name, in_set=in_set, z_in=z_in, unit_in=unit_in, k_in=k_in,
                out_set=out_set, z_out=z_out, unit_out=unit_out, k_out=k_out, compute=False)


                           
    
    # Determine emulation logic      
    combination_order_in = combination_order_out = None
    for i, (k, unit, z) in enumerate(k_z_combination_list):
        if in_set == "lya" and k_in == k and z_in == z and unit_in == unit:
            combination_order_in = i
        if out_set == "lya" and k_out == k and z_out == z and unit_out == unit:
            combination_order_out = i

    # Read the chain
    nametab=Table.read(f'{root_name}.paramnames',format='ascii.no_header',delimiter='\t',names=['name','texname'])

    filenames=glob.glob(f'{root_name}_*.txt')

    for j in filenames:
        data_chain = Table.read(j,names=['firstcol', *nametab['name']],format='ascii.no_header',delimiter='\t')
        data_to_emu = data_order(data_chain, in_set)

        
        if data_to_emu.shape != (len(data_to_emu),len(data_to_emu[0])): 
            raise ValueError("root_name doesn't have the right shape.")
        # Emulate from lya to primordial
        if in_set == "lya" and out_set == "primordial":
            load = np.load("from_lya_to_primordial.npz", allow_pickle=True)
            emu = load["emu_to_omega_mlnAn_s"][combination_order_in]
            prediction = emu(data_to_emu)[0] 

        # Emulate from lya to primordial
        elif in_set == "lya" and out_set == "lya":
            # Step 1: Emulation from lya to primordial
            load = np.load("from_lya_to_primordial.npz", allow_pickle=True)
            emu_to_omegalnAns = load["emu_to_omega_mlnAn_s"][combination_order_in]
            prediction_to_omegalnAns = emu_to_omegalnAns(data_to_emu)[0]
            
            # Step 2: Emulation from primordial to the other lya 
            load = np.load("from_primordial_to_lya.npz", allow_pickle=True)
            emu_to_alphaDnlya = load["emu_to_alphaDn_lya"][combination_order_out]
            prediction = emu_to_alphaDnlya(np.vstack([data_to_emu[:, 0], prediction_to_omegalnAns[:, 0], prediction_to_omegalnAns[:, 1], prediction_to_omegalnAns[:, 2]]).T)[0]

        # Emulate from primordial to lya
        elif in_set == "primordial" and out_set == "lya":
            load = np.load("from_primordial_to_lya.npz", allow_pickle=True)
            emu = load["emu_to_alphaDn_lya"][combination_order_out]
            prediction = emu(data_to_emu)[0]
        elif in_set =="primordial" and out_set == "primordial":
            raise ValueError("Emulation from primordial to primordial is not supported.")
        else:
            raise ValueError("Your chosen parameter set is either not writtten correctly or not supported.")
        # save the data    
        save_func(file_name = j, data_chain=data_chain, emulation = prediction, in_set=in_set, 
              z_in=z_in, unit_in=unit_in, k_in=k_in,out_set=out_set, z_out=z_out, unit_out=unit_out, k_out=k_out)
        # save the parameter name
        save_names(data_path = root_name, nametab = nametab, in_set=in_set, z_in=z_in, unit_in=unit_in, k_in=k_in,
                    out_set=out_set, z_out=z_out, unit_out=unit_out, k_out=k_out)        
        
    
