import classy
import numpy as np
import h5py
import json
with open("supported_combinations.json", "r") as file:
    supported_combinations = json.load(file)

file_path = 'Planck_10_sigma_essentials.hdf5'
with h5py.File(file_path, 'r') as hdf:
    loaded_data = {key: hdf[key][:] for key in hdf.keys()}
allparams_essentials = loaded_data

file_path = 'Planck_10_sigma_extras.hdf5'
with h5py.File(file_path, 'r') as hdf:
    loaded_data = {key: hdf[key][:] for key in hdf.keys()}
allparams_extras = loaded_data


rng=np.random.default_rng(seed=12345)
ks = np.geomspace(1e-5,5,num=10000)



def postprocessing_A_and_n_lya(cosmo, z_p = 2.6, pks = None, k_p = 0.017, units = "kms", normalize = True, cdmbar = False):

    # With this function, the conversion of primordial data to lya data can be done
    
    pks = pks
    if units == "Mpc" or units == "MPC" or units == "mpc":
      unit = 1
      unit_kms = 1/(cosmo.Hubble(z_p)/cosmo.Hubble(0)*cosmo.h()*100./(1.+z_p))
    elif units == "skm" or units == "SKM" or units == "kms" or units == "KMS":
      unit = cosmo.Hubble(z_p)/cosmo.Hubble(0)*cosmo.h()*100./(1.+z_p)
      unit_kms = 1
    elif "h" in units or "H" in units:
      unit = cosmo.h()
      unit_kms = cosmo.Hubble(0)*(1+z_p)/(cosmo.Hubble(z_p)*100)
    else:
      raise ValueError("Your input of units='{}' could not be interpreted".format(units))
    x,y = np.log(ks),np.log(pks)
    k_p_Mpc = k_p*unit
    k_p_kms = k_p*unit_kms
    x0 = np.log(k_p_Mpc)
    scale = 0.1
    w = np.exp(-0.5*(x-x0)*(x-x0)/scale/scale) #Unit = 1
    dw = (x-x0)/scale/scale*np.exp(-0.5*(x-x0)*(x-x0)/scale/scale) #Unit = 1/scale
    ddw = (-1./scale/scale + ((x-x0)/scale/scale)**2)*np.exp(-0.5*(x-x0)*(x-x0)/scale/scale) #Unit = 1/scale^2
    s = np.trapz(w,x)
    r = np.trapz(y*w,x)/s
    dr = np.trapz(y*dw,x)/s
    ddr = np.trapz(y*ddw,x)/s
    A_lya_Mpc = np.exp(r)
    n_lya = dr
    alpha_lya = ddr
    # Unit conversion
    if not normalize:
      A_lya = A_lya_Mpc/unit**3
    else:
      A_lya = A_lya_Mpc*k_p_Mpc**3
    return {'D_lya': A_lya, 'n_lya': n_lya, 'mPk': pks, 'alpha_lya': alpha_lya}

# call CLASS
cosmo = classy.Class()
cosmo.set({ 'output':''})

# Generate random data of primordial parameters taken from Planck results and combine them together

""""""
nsamples = supported_combinations["number_of_training_samples"]

""""""
omega_b = 0.02233
omega_c = 0.1200
omega_sum = omega_b + omega_c
single_nsamples = nsamples
uncertenties = 10 * np.array([0.0012, 0.54, 0.014, 0.0042])
omega_m = rng.normal(omega_sum , uncertenties[0], single_nsamples)
H0 = rng.normal(67.36 , uncertenties[1], single_nsamples)
log_A_s = rng.normal(3.044 , uncertenties[2] , single_nsamples) 
n_s = rng.normal(0.9649, uncertenties[3] , single_nsamples)
parameter_combinations = np.vstack([omega_m, H0, log_A_s, n_s]).T

n_combinations = len(parameter_combinations)

omega_m_combination_list = np.zeros(n_combinations, dtype=float)
H0_combination_list = np.zeros(n_combinations, dtype=float)
log_A_s_combination_list = np.zeros(n_combinations, dtype=float)
n_s_combination_list = np.zeros(n_combinations, dtype=float)

def additional_converter(k_z_combination_list_to_convert):
    
    k_z_combination_list = k_z_combination_list_to_convert
    z_piv = np.array(k_z_combination_list[:,2], dtype = float)
    
    
    # importing the supported mPk
    
    supported_z_piv = np.load("mPk_of_z_piv.npz", allow_pickle=True)["z_piv_list"]
    supported_mPk = np.load("mPk_of_z_piv.npz", allow_pickle=True)["mPk_list"]
    z_piv = np.array([2.4,2.2,2.8])
    mPk_list = np.zeros((len(supported_mPk[:,0,0]), len(z_piv), len(supported_mPk[0,0,:])))
    
    for i, z_p in enumerate(supported_z_piv):
        for j in range(len(z_piv)):
            if np.equal(z_p, z_piv[j]):  
                mPk_list[:,j,:] = np.load("mPk_of_z_piv.npz", allow_pickle=True)["mPk_list"][:,i,:]  
    
    
    
    n_types = len(k_z_combination_list)
    
    D_lya_results = np.zeros((n_combinations, n_types), dtype=float)
    n_lya_results = np.zeros((n_combinations, n_types), dtype=float)

    alpha_lya_results = np.zeros((n_combinations, n_types), dtype=float)
    power_spectrum_results = np.zeros((n_combinations, n_types, len(ks)), dtype=float)
    
    # this copy makes sure that the type is the right one to safe.
    k_piv_for_save = np.array(k_z_combination_list[:,0], dtype = float)
    units_for_save = np.array(k_z_combination_list[:,1], dtype = "S")
    z_piv_for_save = np.array(k_z_combination_list[:,2], dtype = float)
    
    
    # initialize the primordial parameters in CLASS and convert them to parameters to the lya parameters.
    for i, params in enumerate(parameter_combinations):
        omega_m_val, H0_val, log_A_s_val, n_s_val = params
        
        omega_m_combination_list[i] = omega_m_val
        H0_combination_list[i] = H0_val
        log_A_s_combination_list[i] = log_A_s_val
        n_s_combination_list[i] = n_s_val
    
        parms = {'omega_m': omega_m_val, 'H0': H0_val, 'ln_A_s_1e10': log_A_s_val, 'n_s': n_s_val}
        cosmo.set(parms)
        cosmo.compute()
        


        for j,(kp,unit,zp) in enumerate(k_z_combination_list):
            ppsing = postprocessing_A_and_n_lya(cosmo, pks =mPk_list[i,j,:],k_p=kp,z_p = zp,units=unit)
            D_lya_results[i,j] = ppsing['D_lya']
            n_lya_results[i,j] = ppsing['n_lya']
            alpha_lya_results[i,j] = ppsing['alpha_lya']
            power_spectrum_results[ i, j, :] = ppsing['mPk'][:]
            k_convert_Mpc[i,j] = ppsing['k_p_Mpc']
            k_convert_kms[i,j] = ppsing['k_p_kms']
            
            
    # the converted dictonaries
    
    results_list_essentials = {'omega_m':omega_m_combination_list, 'H0': H0_combination_list, 'ln_A_s_1e10': log_A_s_combination_list, 
                    'n_s' : n_s_combination_list,'D(k_pivot,z_pivot)':D_lya_results,
                     'n(k_pivot,z_pivot)': n_lya_results,
                    'mPk(z_pivot)' : power_spectrum_results}
    results_list_extras = { 'alpha_lya' : alpha_lya_results,'z_pivots':z_piv_for_save,
                           'units':units_for_save,'k_pivots': k_piv_for_save}
    
    
    
    # overwriting the data of the old conversion-dictionary
    
    results_list_essentials["D(k_pivot,z_pivot)"] = np.hstack([allparams_essentials["D(k_pivot,z_pivot)"], D_lya_results])
    results_list_essentials["n(k_pivot,z_pivot)"] = np.hstack([allparams_essentials["n(k_pivot,z_pivot)"], n_lya_results])
    
    if results_list_essentials["mPk(z_pivot)"].ndim == 2: 
        mPk_reshaped = results_list_essentials["mPk(z_pivot)"][:,np.newaxis,:]
    else:
        mPk_reshaped = results_list_essentials["mPk(z_pivot)"]
    
    results_list_essentials["mPk(z_pivot)"] = np.hstack([allparams_essentials["mPk(z_pivot)"],mPk_reshaped])
    results_list_extras["alpha_lya"] = np.hstack([allparams_extras["alpha_lya"], alpha_lya_results])
    results_list_extras["z_pivots"] = np.append(allparams_extras["z_pivots"], z_piv_for_save)
    results_list_extras["units"] = np.append(allparams_extras["units"], units_for_save)
    results_list_extras["k_pivots"] = np.append(allparams_extras["k_pivots"], k_piv_for_save)

    
    
    
    # saving the overwrited data in the old dictionary: 
    
    with h5py.File('Planck_10_sigma_essentials.hdf5', 'w') as hdf:
        for k,v in results_list_essentials.items():
            hdf[k]=v
    
    with h5py.File('Planck_10_sigma_extras.hdf5', 'w') as hdf:
        for k,v in results_list_extras.items():
            hdf[k]=v
