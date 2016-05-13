import numpy as np

def load_umbrella_data():
    A = np.matrix([[0.7, 0.3], [0.3, 0.7]])
    B = np.matrix([[0.9, 0.1], [0.2, 0.8]])
    pi = np.matrix([[0.5, 0.5]]).T
    data = np.matrix([[0, 0, 1, 0, 0]]).T # column vector

    return [data, A, B, pi]

def load_umbrella_qt():
    data, A, B, pi = load_umbrella_data()

    alphas = hmm_fit.compute_alphas(data[:,0], A, B, pi)
    betas = hmm_fit.compute_betas(data[:,0], A, B, pi)
    gammas = hmm_fit.compute_gammas(data[:,0], A, B, pi)
    xis = hmm_fit.compute_xis(data[:,0], A, B, pi)

    return [alphas, betas, gammas, xis]

def load_words_data():
    """ Based on the 'Example of Baum-Welch Algorithm' by Larry Moss. """
    A = np.matrix([[0.3, 0.7], [0.1, 0.9]])
    B = np.matrix([[0.4, 0.6], [0.5, 0.5]])
    pi = np.matrix([[0.85, 0.15]]).T
    data = np.matrix([[0, 1, 1, 0], [1, 0, 1, 1]]).T

    return [data, A, B, pi]

def assertEqualMatrices( matrix1, matrix2):
    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)

    for c1, c2 in zip(matrix1, matrix2):
        for v1, v2 in zip(c1, c2):
            assertEqualApprox(v1, v2)

def assertEqualApprox( v1, v2):
    assert np.abs(v1 - v2) <= 5.e-2




