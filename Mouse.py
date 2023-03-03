import level3
import jax.numpy as np
import os
import pickle

# Reference data
with open(os.path.join('Data', 'data_save.pkl'), 'rb') as f:
    data = pickle.load(f)

# Network size
N_E = 80
N_I = 20
N = N_E + N_I

pref_E = np.linspace(0, 180, N_E, False)
pref_I = np.linspace(0, 180, N_I, False)
pref = np.concatenate([pref_E, pref_I])

# Contrast and orientation ranges
orientations = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165])
contrasts = np.array([0., 0.0432773, 0.103411, 0.186966, 0.303066, 0.464386, 0.68854, 1.])

# Parameters for input stage
g_E = 1
g_I = 1
w_ff_E = 30
w_ff_I = 30
sig_ext = 5

g = np.concatenate([np.ones(N_E) * g_E,
                    np.ones(N_I) * g_I])
        
w_ff = np.concatenate([np.ones(N_E) * w_ff_E,
                        np.ones(N_I) * w_ff_I])

# Auxiliary time constants for excitatory and inhibitory
T_alpha = 0.5
T_E = 0.01
T_I = 0.01 * T_alpha
# Auxiliary time constant vector for all cells
T = np.concatenate([T_E * np.ones(N_E), T_I * np.ones(N_I)])
T_inv = np.reciprocal(T)
        
# Membrane time constants for excitatory and inhibitory
tau_alpha = 1
tau_E = 0.01
tau_I = 0.01 * tau_alpha
 # Membrane time constant vector for all cells
tau = np.concatenate([tau_E * np.ones(N_E), tau_I * np.ones(N_I)])
        
# Refractory periods for exitatory and inhibitory
tau_ref_E = 0.005
tau_ref_I = 0.001
tau_ref = np.concatenate([
    tau_ref_E * np.ones(N_E),
    tau_ref_I * np.ones(N_I)])



# First layer
J = np.log(np.array([[0.63, 0.6],
                    [0.32, 0.25]]))

P = np.log(np.array([[0.11, 0.11], 
                    [0.45, 0.45]]))

w = np.log(32) * np.ones([2,2])


if __name__ == "__main__":
    L = level3.loss_from_parameters(data, 1, 40, N_E, N_I, contrasts, orientations, J*10, P, w, T_inv, tau, tau_ref, pref_E, pref_I, g, w_ff, sig_ext)

    print(L)

