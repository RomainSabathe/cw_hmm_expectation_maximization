import numpy as np
from expected_counts import *
from initialisation import *
from tools import *
from compute_quantities import *

def expectation_maximization(data, K, N, steps=None, model='multinomial'):
    """
    Performs the EM algorithm on a HMM modeled dataset (this algorithm is also
    named Baum-Welch.
    data: of size [T, S] where T is the length of the Markov Chains and S is the
          number of samples we derived from the chain.
          For now, the values of 'data' can only be discrete.
    K: an estimate of the number of states the latent variables can take
    S: the number of states the visible variables can take
    steps: the number of iteration to perform until stopping the algorithm.
           If set to None, then this number is adjusted automatically to perform
           until convergence.
    """
    T, S = data.shape
    A, B, pi = initialise_parameters(K, N, model)

    for _ in range(steps):
        all_gammas = compute_all_gammas(data, A, B, pi, model)
        all_xis = compute_all_xis(data, A, B, pi, model)

        A = update_A(data, all_gammas, all_xis)
        pi = update_pi(data, all_gammas, all_xis)

        if model == 'multinomial':
            B = update_B(data, all_gammas, all_xis)
        else:
            B['mu'] = update_mu(data, all_gammas, all_xis)
            B['sigma2'] = update_sigma2(data, all_gammas, all_xis, B)


    return [A, B, pi]


def update_A(data, all_gammas, all_xis):
    """
    In the case of a HMM with discrete visible variables, performs EM to update
    the value of the transition matrix A between the hidden states.
    N: the number of states the visible variables can take | unused here
    data: of size [S, T] where T is the length of the Markov Chains and S is the
          number of samples we derived from the chain.
          For now, the values of 'data' can only be discrete.
    """
    _, _, K, _ = all_gammas.shape # number of hidden states
    updated_A = np.matrix(np.zeros([K, K]))
    for j in range(K):
        normalisation_factor = np.sum([expected_count_j_k(j, k, all_xis) for k in range(K)])
        for k in range(K):
            expected_count = expected_count_j_k(j, k, all_xis)
            updated_A[j,k] = expected_count / normalisation_factor

    return updated_A


def update_pi(data, all_gammas, all_xis):
    """
    In the case of a HMM with discrete visible variables, performs EM to update
    the value of the initial distribution between the hidden states.
    N: the number of states the visible variables can take | unused here
    data: of size [S, T] where T is the length of the Markov Chains and S is the
          number of samples we derived from the chain.
          For now, the values of 'data' can only be discrete.
    """
    _, _, K, _ = all_gammas.shape # number of hidden states
    updated_pi = np.matrix(np.zeros([K, 1]))

    normalisation_factor = np.sum([expected_count_1_k(k, all_gammas) for k in range(K)])
    for k in range(K):
        expected_count = expected_count_1_k(k, all_gammas)
        updated_pi[k,0] = expected_count / normalisation_factor

    return updated_pi


def update_B(data, all_gammas, all_xis):
    """
    In the case of a HMM with discrete visible variables, performs EM to update
    the value of evidence vectors.
    data: of size [T, S] where T is the length of the Markov Chains and S is the
          number of samples we derived from the chain.
          For now, the values of 'data' can only be discrete.
    """
    _, _, K, _ = all_gammas.shape # number of hidden states
    N = len(np.unique(np.array(data))) # number of visible states
    updated_B = np.matrix(np.zeros([K, N]))
    for j in range(K):
        normalisation_factor = expected_count_j(j, all_gammas)
        for l in range(N):
            expected_count = expected_count_m_j_l(j, l, all_gammas, data)
            updated_B[j,l] = expected_count / normalisation_factor

    return updated_B


def update_mu(data, all_gammas, all_xis):
    _, _, K, _ = all_gammas.shape # number of hidden states
    updated_mu = np.matrix(np.zeros([K, 1]))
    for k in range(K):
        num = expected_count_x_k(k, all_gammas, data)
        dem = expected_count_j(k, all_gammas)
        updated_mu[k,0] = num / dem

    return updated_mu


def update_sigma2(data, all_gammas, all_xis, E):
    _, _, K, _ = all_gammas.shape # number of hidden states
    updated_sigma2 = np.matrix(np.zeros([K, 1]))
    for k in range(K):
        ec_x_x_k = expected_count_x_x_k(k, all_gammas, data)
        ec_k = expected_count_j(k, all_gammas)

        num = ec_x_x_k - ec_k * E['mu'][k]**2
        dem = ec_k
        updated_sigma2[k,0] = num / dem

    return updated_sigma2
