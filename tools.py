import numpy as np

def normalise(vect):
    """ Normalise a vector so that its sum equals 1. """
    return vect / np.sum(vect)


def hadamard(vect1, vect2):
    if not isinstance(vect1, np.matrixlib.defmatrix.matrix):
        vect1 = np.matrix(vect1)
    if not isinstance(vect2, np.matrixlib.defmatrix.matrix):
        vect2 = np.matrix(vect2)
    return np.multiply(vect1, vect2)


def normal(x, mu, sigma2):
    factor = 1. / (np.sqrt(2. * np.pi * sigma2))
    arg_exp = (1./2.) * (x - mu) / sigma2

    return factor * np.exp(arg_exp)


def compute_B_normal(x, E):
    """
    Computes the equivalent of the B matrix for the multinomial distrib.
    Equivalent of the 'psi' in Murphy's book.
    """
    K = len(E['mu'])
    B = np.matrix(np.zeros([K, 1]))
    for k in range(K):
        B[k,0] = normal(x, E['mu'][k], E['sigma2'][k])

    return B
