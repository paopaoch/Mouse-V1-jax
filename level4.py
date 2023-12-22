import level3
import jax.numpy as np
from jax import grad, jit
import matplotlib.pyplot as plt
import time
import os

def optimise_JPw_2(data, step_size_effect, n_subsamples, N_E, N_I, contrasts, orientations, J_init, P_init, w_init, T_inv, tau, tau_ref, pref_E, pref_I, g, w_ff, sig_ext):
    params = np.array(J_init[0,0])
    rand_mat = level3.random_matrix(N_E + N_I)  # Put inside gradient descent loop

    def optimising_func(J1):

        J = np.array([[J1, J_init[0,1]], [J_init[1,0], J_init[1,1]]])
        return level3.loss_from_parameters(data, step_size_effect, n_subsamples, rand_mat, N_E, N_I, contrasts, orientations, J, P_init, w_init, T_inv, tau, tau_ref, pref_E, pref_I, g, w_ff, sig_ext)

    # gradient_func = jit(grad(optimising_func))
    # jit_func = jit(optimising_func)

    gradient_func = grad(optimising_func)
    jit_func = optimising_func

    for i in range(3):
        print("params: " + str(params))
        t0 = time.process_time()

        loss = jit_func(params)

        t1 = time.process_time()
        print("loss: " + str(loss))
        print("cpu time: " + str(t1 - t0))
        
        gradient = gradient_func(params)
        
        t2 = time.process_time()
        print("gradient: " + str(gradient))
        print("cpu time: " + str(t2 - t1))

    print(loss)


def optimise_JPw(data, step_size_effect, n_subsamples, N_E, N_I, contrasts, orientations, J_init, P_init, w_init, T_inv, tau, tau_ref, pref_E, pref_I, g, w_ff, sig_ext, n_iter=12):
    init_params = np.array([J_init, P_init, w_init])
    log_params = np.log(init_params)
    rand_mat = level3.random_matrix(N_E + N_I)

    def optimising_func(log_params):
        params = np.exp(log_params)
        return level3.loss_from_parameters(data, step_size_effect, n_subsamples, rand_mat, N_E, N_I, contrasts, orientations, *params, T_inv, tau, tau_ref, pref_E, pref_I, g, w_ff, sig_ext)

    gradient_func = grad(optimising_func)

    tracker = np.zeros((3, 4, n_iter))
    loss_track = np.zeros((n_iter, 1))


    for i in range(n_iter):
        print("params: " + str(np.exp(log_params)))
        t0 = time.process_time()

        loss = optimising_func(log_params)
        t1 = time.process_time()
        print("loss: " + str(loss))
        print("cpu time: " + str(t1 - t0))
        
        gradient = gradient_func(log_params)
        
        t2 = time.process_time()
        print("gradient: " + str(gradient))
        print("cpu time: " + str(t2 - t1))
        print()

        log_params = log_params - 0.1 * gradient

        tracker = tracker.at[:, :2, i].set(np.exp(log_params[:, 0, :]))
        tracker = tracker.at[:, 2:, i].set(np.exp(log_params[:, 1, :]))
        loss_track = loss_track.at[i].set(loss)

    params = np.exp(log_params)
    print("Final loss: " + str(loss))
    print("Initial params: " + str(init_params))
    print("Final params: " + str(params))

    balance_mean, balance_std = level3.get_balance(rand_mat, N_E, N_I, contrasts, orientations, *params, T_inv, tau, tau_ref, pref_E, pref_I, g, w_ff, sig_ext)
    print("Balance: " + str(balance_mean) + " std: " + str(balance_std))

    K_E, K_I = level3.get_K(rand_mat, N_E, N_I, *params, pref_E, pref_I)
    print("K_E: " + str(K_E) + " K_I: " + str(K_I))
    '''
    for i in range(3):
        plt.plot(range(n_iter), tracker[i].T)
        plt.savefig(os.path.join("plots", ["J","P","w"][i] + "-opt.png"))
        plt.show()

    plt.plot(range(n_iter), loss_track)
    plt.savefig(os.path.join("plots", "loss-opt.png"))
    plt.show()
    '''

def optimise_JPw_no_grad(data, step_size_effect, n_subsamples, N_E, N_I, contrasts, orientations, J_init, P_init, w_init, T_inv, tau, tau_ref, pref_E, pref_I, g, w_ff, sig_ext, n_iter=12):
    init_params = np.array([J_init, P_init, w_init])
    log_params = np.log(init_params)
    rand_mat = level3.random_matrix(N_E + N_I)

    def optimising_func(log_params):
        params = np.exp(log_params)
        return level3.loss_from_parameters(data, step_size_effect, n_subsamples, rand_mat, N_E, N_I, contrasts, orientations, *params, T_inv, tau, tau_ref, pref_E, pref_I, g, w_ff, sig_ext)
    
    tracker = np.zeros((3, 4, n_iter))
    loss_track = np.zeros((n_iter, 1))

    optimising_func = jit(optimising_func)
    print("DONE JIT")

    for i in range(n_iter):
        print("params: " + str(np.exp(log_params)))
        t0 = time.process_time()

        loss = optimising_func(log_params)
        t1 = time.process_time()
        print("loss: " + str(loss))
        print("cpu time: " + str(t1 - t0))
        

        log_params = log_params - 0.1 * 0

        tracker = tracker.at[:, :2, i].set(np.exp(log_params[:, 0, :]))
        tracker = tracker.at[:, 2:, i].set(np.exp(log_params[:, 1, :]))
        loss_track = loss_track.at[i].set(loss)

    params = np.exp(log_params)
    print("Final loss: " + str(loss))
    print("Initial params: " + str(init_params))
    print("Final params: " + str(params))

    balance_mean, balance_std = level3.get_balance(rand_mat, N_E, N_I, contrasts, orientations, *params, T_inv, tau, tau_ref, pref_E, pref_I, g, w_ff, sig_ext)
    print("Balance: " + str(balance_mean) + " std: " + str(balance_std))

    K_E, K_I = level3.get_K(rand_mat, N_E, N_I, *params, pref_E, pref_I)
    print("K_E: " + str(K_E) + " K_I: " + str(K_I))
    '''
    for i in range(3):
        plt.plot(range(n_iter), tracker[i].T)
        plt.savefig(os.path.join("plots", ["J","P","w"][i] + "-opt.png"))
        plt.show()

    plt.plot(range(n_iter), loss_track)
    plt.savefig(os.path.join("plots", "loss-opt.png"))
    plt.show()
    '''