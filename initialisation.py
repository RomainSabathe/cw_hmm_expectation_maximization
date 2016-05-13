import numpy as np
from tools import *

def initialise_parameters(K, N, model='multinomial'):
    """
    Initialise parameters (A, B, pi) for a EM of HMM.
    K: the number of states the hidden variables can take
    N: the number of states the visible variables can take.
    model: whether 'multinomial' or 'normal'
    """
    A = initialise_A(K)
    pi = initialise_pi(K)

    if model == 'multinomial':
        B = initialise_B(K, N)
        return A, B, pi
    elif model == 'normal':
        E = {}
        E['mu'] = np.random.rand(K)
        E['sigma2'] = np.random.rand(K)
        return A, E, pi
    else:
        raise Exception('Model unknown')


def initialise_A(K):
    """
    See 'initialise_parameters' for usage.
    Remainder: A is the transition matrix for the latent variables.
               A[i,j] = p(z_t = j | z_t-1 = i)
    K: the number of states the hidden variables can take
    """
    A = np.matrix(np.zeros([K, K]))
    for i in range(K):
        A[i,:] = np.random.rand(K)
        A[i,:] = normalise(A[i,:]) # the lines must sum to 1 (probability of
                                   # getting *somewhere* starting from i)
    return A


def initialise_B(K, N):
    """
    See 'initialise_parameters' for usage.
    Remainder: B is the estimated posterior probability at time t.
               B[i, j] = p(x_t = j | z_t = i)
    K: the number of states the hidden variables can take
    N: the number of states the visible variables can take
    """
    B = np.matrix(np.zeros([K, N]))
    for i in range(K):
        B[i,:] = np.random.rand(N)
        B[i,:] = normalise(B[i,:]) # the lines must sum to 1 (probability of
                                   # observing a visible state if the latent
                                   # variable is i)
    return B


def initialise_pi(K):
    """
    See 'initialise_parameters' for usage.
    Remainder: pi: the initial state distribution. pi(j) = p(z_(t=1) = j)
               pi is a column vector.
    K: the number of states the hidden variables can take
    """
    pi = np.matrix(np.zeros([1, K]))
    pi[0,:] = np.random.rand(K)
    pi[0,:] = normalise(pi[0,:])
    pi = pi.T # is now a column vector

    return pi
