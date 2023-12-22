import level2
import level1
import jax
import jax.numpy as np
import jax.random as jrand
from jax.lax import map as jmap


prng = jrand.PRNGKey(1)


def random_matrix(N):
    return jrand.uniform(prng, (N,N))


def kernel(x, y, w=1, axes=[-2,-1]):
    '''Problem: This operation is now comparing two tuning curves instead of two data points??'''
    return np.exp( - np.sum((x - y)**2, axis=axes) / (2 * w**2))


# def MMD(X, Y):
#     # Maximum Mean Discrepancy
#     N = len(X)
#     M = len(Y)
    
#     XX = np.mean(kernel(X[None, :, :, :], X[:, None, :, :]))
#     XY = np.mean(kernel(X[None, :, :, :], Y[:, None, :, :]))
#     YY = np.mean(kernel(Y[None, :, :, :], Y[:, None, :, :]))
    
#     return XX - 2*XY + YY


def individual_terms_vmap(x, y):
    N = x.shape[0]
    M = y.shape[0]

    def inner_loop(i):
        x_repeated = np.expand_dims(x[i, :, :], 0).repeat(M, axis=0)
        return np.mean(kernel(y, x_repeated))

    accum_output = jax.vmap(inner_loop)(np.arange(N))
    
    return np.sum(accum_output) / N

def MMD(x, y):
    XX  = individual_terms_vmap(x, x)
    XY  = individual_terms_vmap(x, y)
    YY  = individual_terms_vmap(y, y)
    return XX + YY - 2 * XY


def loss_function(sim_tuning_curves, avg_step, data, step_size_effect):
    loss = MMD(sim_tuning_curves, data) + step_size_effect * (np.maximum(1, avg_step) - 1)
    print("Calculated loss")
    return loss


def generate_tuning_curves(rand_mat, N_E, N_I, contrasts, orientations, J, P, w, T_inv, tau, tau_ref, pref_E, pref_I, g, w_ff, sig_ext):
    N = N_E + N_I
    pref = np.concatenate([pref_E, pref_I])
    prob = level2.generate_prob_matrix(pref_E, pref_I, P, w)
    C = level2.generate_C_matrix(prob, rand_mat)

    def solve_for(inputs):
        c = inputs[0]
        theta = inputs[1]

        W, W2 = level2.generate_network(C, J, N_E, N_I)

        r_fp, avg_step = level1.network_to_state(N, W, W2, c, theta, T_inv, tau, tau_ref, pref, g, w_ff, sig_ext)

        Wr = W * r_fp[None, :]
        B = np.sum(np.absolute(Wr), axis=1) / np.absolute(np.sum(Wr, axis=1))

        #return r_fp, avg_step
        return np.concatenate([r_fp, np.array([avg_step, np.mean(B), np.var(B)])])

    
    inputs_list = np.array(np.meshgrid(contrasts, orientations)).T.reshape([-1,2])

    # solves = np.array(jmap(solve_for, inputs_list)).reshape([len(contrasts), len(orientations), N+3])   
    solves = np.array(jax.vmap(solve_for)(inputs_list)).reshape([len(contrasts), len(orientations), N+3])             
    result = np.moveaxis(solves, 2, 0)   
    print("Ran all orientation and contrast")
    
    balance = result[-2:]  # Q: Last two elem. Why?
    avg_step = np.mean(result[-3])
    tuning_curves = result[:-3]
    '''

    tuning_curves = np.zeros((N, len(contrasts), len(orientations)))
    avg_tot = 0

    for i, c in enumerate(contrasts):
        for j, theta in enumerate(orientations):
            r_fp, avg_step = solve_for((c, theta))
            tuning_curves[:, i, j] = r_fp
            avg_tot += avg_step
    '''
    return tuning_curves, avg_step, balance

    
def loss_from_parameters(data, step_size_effect, n_subsamples, rand_mat, N_E, N_I, contrasts, orientations, J, P, w, T_inv, tau, tau_ref, pref_E, pref_I, g, w_ff, sig_ext):
    tuning_curves, avg_step, balance = generate_tuning_curves(rand_mat, N_E, N_I, contrasts, orientations, J, P, w, T_inv, tau, tau_ref, pref_E, pref_I, g, w_ff, sig_ext)

    sub_tuning_curves = jrand.choice(prng, tuning_curves, (n_subsamples,1))
    sub_data = jrand.choice(prng, data, (n_subsamples,1))

    return loss_function(sub_tuning_curves, avg_step, sub_data, step_size_effect)


def get_balance(rand_mat, N_E, N_I, contrasts, orientations, J, P, w, T_inv, tau, tau_ref, pref_E, pref_I, g, w_ff, sig_ext):
    """Q: What does this function do?"""
    tuning_curves, avg_step, balance = generate_tuning_curves(rand_mat, N_E, N_I, contrasts, orientations, J, P, w, T_inv, tau, tau_ref, pref_E, pref_I, g, w_ff, sig_ext)

    balance_mean = np.mean(balance[0])
    balance_var =  np.var(balance[0]) #+ np.mean(balance[1])

    print(balance)

    return balance_mean, np.sqrt(balance_var)


# def get_balance(rand_mat, N_E, N_I, contrasts, orientations, J, P, w, T_inv, tau, tau_ref, pref_E, pref_I, g, w_ff, sig_ext):
#     tuning_curves, avg_step, balance = generate_tuning_curves(rand_mat, N_E, N_I, contrasts, orientations, J, P, w, T_inv, tau, tau_ref, pref_E, pref_I, g, w_ff, sig_ext)

#     balance_mean = np.mean(balance[0])
#     balance_var =  np.var(balance[0]) #+ np.mean(balance[1])

#     #print(balance)

#     return balance_mean, np.sqrt(balance_var)


def get_K(rand_mat, N_E, N_I, J, P, w, pref_E, pref_I):
    prob = level2.generate_prob_matrix(pref_E, pref_I, P, w)
    C = level2.generate_C_matrix(prob, rand_mat)

    K_E = np.mean(np.sum(C[:N_E, : ] > 0.5, axis = 1))
    K_I = np.mean(np.sum(C[N_E:, : ] > 0.5, axis = 1))

    return K_E, K_I