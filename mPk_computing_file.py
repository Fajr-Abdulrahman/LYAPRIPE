import classy
import numpy as np
import json
import json
with open("supported_combinations.json", "r") as file:
    supported_combinations = json.load(file)

print("Starting computing the mPk...")
# call CLASS
cosmo = classy.Class()
cosmo.set({'z_max_pk':20, 'P_k_max_h/Mpc':100.0, 'omega_b': 0.02233, 'output':'mPk'})

rng=np.random.default_rng(seed=12345)
ks = np.geomspace(1e-5,5,num=10000)

# import number of training samples from the json-file
""""""
nsamples = supported_combinations["number_of_training_samples"]

""""""
# Generate random data of primordial parameters taken from Planck results and combine them together

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

z_piv = np.array(supported_combinations["z_piv_for_computing_mPk"])


n_types = len(z_piv)

power_spectrum_results = np.zeros((n_combinations, n_types, len(ks)), dtype=float)

# compute the mPk using CLASS 
for i, params in enumerate(parameter_combinations):
    omega_m_val, H0_val, log_A_s_val, n_s_val = params

    parms = {'omega_m': omega_m_val, 'H0': H0_val, 'ln_A_s_1e10': log_A_s_val, 'n_s': n_s_val}
    cosmo.set(parms)
    cosmo.compute()
    
    for j, z_p in enumerate(z_piv):
        power_spectrum_results[ i, j, :] =cosmo.get_pk_all(ks, z=z_p, nonlinear = False, cdmbar = False)
        
# save the mPk data
np.savez_compressed("mPk_of_z_piv.npz", mPk_list = power_spectrum_results, z_piv_list=z_piv, allow_pickle=True)