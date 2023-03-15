import level1
import sim_utils
import jax.numpy as np


def sigmoid_matrix(probabilities, rand_mat):
    # Produce continuous Bernoulli substitute
    return sim_utils.sigmoid(32 * (probabilities - rand_mat))  # Factor 32 can change


def pref_diff(pref_a, pref_b):
    # Create matrix of differences between preferred orientations
    return pref_a[:, None] - pref_b[None, :]


def generate_prob_matrix(pref_E, pref_I, P, w):
    prob_EE = P[0,0] * sim_utils.circ_gauss(pref_diff(pref_E, pref_E), w[0,0])
    prob_EI = P[0,1] * sim_utils.circ_gauss(pref_diff(pref_E, pref_I), w[0,1])
    prob_IE = P[1,0] * sim_utils.circ_gauss(pref_diff(pref_I, pref_E), w[1,0])
    prob_II = P[1,1] * sim_utils.circ_gauss(pref_diff(pref_I, pref_I), w[1,1])

    return np.block([[prob_EE, prob_EI], [prob_IE, prob_II]])


def generate_C_matrix(prob, rand_mat):
    C = sigmoid_matrix(prob, rand_mat) * (1-np.eye(prob.shape[0]))
    return C


def generate_network(C, J, N_E, N_I):
    '''Randomly generate network'''
    J_full = sim_utils.block_matrix(J, [N_E, N_I])

    # Weight matrix and squared weight matrix
    W = J_full * C
    W2 = np.square(W)

    return W, W2
