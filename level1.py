import sim_utils
import jax.numpy as np
from jax import jit



def get_mu_sigma(W, W2, r, h, xi, tau):
    # Find net input mean and variance given inputs
    mu = tau * (W @ r) + h
    sigma = np.sqrt(tau * (W2 @ r) + xi**2)
    
    return mu, sigma


def stim_to_inputs(N, c, theta, pref, g, w_ff, sig_ext):
    '''Set the inputs based on the contrast and orientation of the stimulus'''
        
    # Distribute parameters over all neurons based on type
    h = c * 20 * g * sim_utils.circ_gauss(theta - pref, w_ff)
    xi = np.ones(N) * sig_ext
        
    return h, xi


def solve_fixed_point(N, W, W2, h, xi, T_inv, tau, tau_ref):
    r_init = np.zeros(N)

    # Define the function to be solved for
    def drdt_func(r):
        return T_inv * (sim_utils.Phi(*get_mu_sigma(W, W2, r, h, xi, tau), tau, tau_ref=tau_ref) - r)
        
    # Solve using Euler
    r_fp, avg_step = sim_utils.Euler2fixedpt(jit(drdt_func), r_init)
    return r_fp, avg_step


def network_to_state(N, W, W2, c, theta, T_inv, tau, tau_ref, pref, g, w_ff, sig_ext):
    h, xi = stim_to_inputs(N, c, theta, pref, g, w_ff, sig_ext)
    return solve_fixed_point(N, W, W2, h, xi, T_inv, tau, tau_ref)
