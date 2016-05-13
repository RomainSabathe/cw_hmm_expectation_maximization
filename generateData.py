import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt

plt.style.use('ggplot')

def generate_HMM_data(n, T, pi, A, E, outModel='multinomial'):
    """
    Generates n samples from an HMM of length T with the following parameters:
      pi is the initial distribution of states,
      A is the transition probability matrix,
      E are the emission parameters:
        - for a multinomial output alphabet, this is an emission
          probability matrix, p(x_i|y_i)
        - for normal distributed output, this is a dict. where the 'mu'
          key is a vector of means and the 'sigma2' key is a vector of
          variances,
        - for an autoregressive process this is a vector of AR
          coefficients,
      outModel is a string specifying what kind of output model one
      wants:
        - 'multinomial' (default): multinomial output alphabet,
        - 'normal': normal distributed alphabet.
        - 'ar1': autoregressive process of order 1 output.

    Returns: [Y, S] where
     Y: the observations
     S: the latent variables that generated the observations
     """
    # Start generating samples.
    S = np.zeros([n,T]) # Hidden States
    Y = np.zeros([n,T]) # Observations

    for i in range(n):
        # Generate initial state and observation.
        S[i,0] = sample_state(pi)
        latent_state = S[i,0]
        if 'multinomial' in outModel:
            Y[i,0] = sample_state(E[latent_state,:])
        elif 'normal' in outModel:
            Y[i,0] = randn() * np.sqrt( E['sigma2'][latent_state] ) + E['mu'][latent_state]
        elif 'ar1' in outModel:
            Y[i,0] = randn() * 0.01
        else:
            error('Unknown observation model.')

        # Generate the rest of the observations.
        for l in range(1, T):
            previous_state = S[i,l-1]
            S[i,l] = sample_state( A[previous_state, :] )
            latent_state = S[i,l]
            if 'multinomial' in outModel:
                Y[i,l] = sample_state(E[latent_state])
            elif 'normal' in outModel:
                Y[i,l] = randn() * np.sqrt( E['sigma2'][latent_state] ) + E['mu'][latent_state]
            elif 'ar1' in outModel:
                Y[i,l] =  (E[latent_state] * Y[i,l]) + randn()*0.01
            else:
                error('Unknown observation model.')

    return [Y.T, S] # along the lines: T, along columns: the samples


def sample_state(pi):
    """
    Sample a state given the current distribution pi.
    E.g if pi = [0.2, 0.4, 0.3, 0.1], sample_state will mostly return 1, then
    2, 0 and 3.
    """
    r = np.random.rand()
    cumsum = np.cumsum(pi)
    return np.sum(r > cumsum)


def generate_continuous_data(use_default=True, N=None, T=None, pi=None, A=None, E=None):
    if use_default:
        N = 10
        T = 100
        pi = np.array([0.5, 0.5])
        A = np.array([[0.4, 0.6], [0.4, 0.6]])
        E = {'mu': np.array([0.1, 0.5]),
             'sigma2': np.array([0.4, 0.8])}

    return generate_HMM_data(N, T, pi, A, E, 'normal')


def generate_discrete_data(use_default=True, N=None, T=None, pi=None, A=None, E=None):
    if use_default:
        N = 10
        T = 100
        pi = np.array([0.5, 0.5])
        A = np.array([[0.4, 0.6], [0.4, 0.6]])
        E = np.array([[1./6, 1./6, 1./6, 1./6, 1./6, 1./6.],
                      [1./10, 1./10, 1./10, 1./10, 1./10, 1./2]])

    return generate_HMM_data(N, T, pi, A, E)
