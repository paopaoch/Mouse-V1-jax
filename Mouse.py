import level4, level3, level2, level1, sim_utils
import jax.numpy as np
import jax
import numpy
import os
import pickle
import matplotlib.pyplot as plt
import time

#print("jax backend {}".format(xla_bridge.get_backend().platform))
print(jax.devices())

# Reference data
with open(os.path.join('Data', 'data_save.pkl'), 'rb') as f:
    data = pickle.load(f)

# Network size
N = 4000

N_E = int(.8 * N)
N_I = N - N_E
print(N_E, N_I)
N = N_E + N_I

pref_E = np.linspace(0, 180, N_E, False)
pref_I = np.linspace(0, 180, N_I, False)
pref = np.concatenate([pref_E, pref_I])

# Contrast and orientation ranges
orientations = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165])
contrasts = np.array([0., 0.0432773, 0.103411, 0.186966, 0.303066, 0.464386, 0.68854, 1.])


step_size_effect = 0.01
n_subsamples = 100

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
J = np.array([[0.63, 0.6], [0.32, 0.25]]) * np.sqrt(10)

P = np.array([[0.11, 0.11], [0.45, 0.45]])

w = 32 * np.ones([2,2])

example_random = level3.random_matrix(N)


def test1():
    
    pref = np.concatenate([pref_E, pref_I])
    prob = level2.generate_prob_matrix(pref_E, pref_I, P, w)
    
    c = contrasts[0]
    theta = orientations[0]

    C = level2.generate_C_matrix(prob, example_random)
    W, W2 = level2.generate_network(C, J, N_E, N_I)

    t0 = time.process_time()
    level1.network_to_state(N, W, W2, c, theta, T_inv, tau, tau_ref, pref, g, w_ff, sig_ext)
    print(time.process_time() - t0)


def test2():
    t0 = time.process_time()
    TC, avg_step, balance = level3.generate_tuning_curves(example_random, N_E, N_I, contrasts, orientations, J, P, w, T_inv, tau, tau_ref, pref_E, pref_I, g, w_ff, sig_ext)
    print(time.process_time() - t0)
    plt.show()

    for i, tc in enumerate(TC):
        plt.imshow(tc)
        plt.savefig(os.path.join("plots", "p"+str(i)+".png"))


def test3():
    t0 = time.process_time()
    L = level3.loss_from_parameters(data, step_size_effect, n_subsamples, example_random, N_E, N_I, contrasts, orientations, J, P, w, T_inv, tau, tau_ref, pref_E, pref_I, g, w_ff, sig_ext)
    print(time.process_time() - t0)
    print("Loss: "+str(L))


def test4():
    level4.optimise_JPw(data, step_size_effect, n_subsamples, N_E, N_I, contrasts, orientations, J, P, w, T_inv, tau, tau_ref, pref_E, pref_I, g, w_ff, sig_ext)


if __name__ == "__main__":
    jax.TF_CPP_MIN_LOG_LEVEL=0

    x = np.linspace(-10, 10, 101)

    #plt.plot(x, sim_utils.f_ricci(x))
    #plt.savefig(os.path.join("plots", "plot.png"))
    
    

    test4()
