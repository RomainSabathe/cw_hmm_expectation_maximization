from tests_common import *
from generateData import *
import numpy as np
import hmm_fit

"""

NOTE
-
In this file, we set up tests to make sure our code runs as it has been
intended to.
It is NOT a file where we test the results of the EM/Viterbi algorithms. This is
done in 'main.py'

"""

def test_compute_alphas():
    """ Based on the 'Example of Baum-Welch Algorithm' by Larry Moss. """
    data, A, B, pi = load_words_data()
    alphas = hmm_fit.compute_alphas(data[:,0], A, B, pi)

    expected_alphas = [np.matrix([[0.809523],       #[np.matrix([[0.34],
                                  [0.1904761]]),      #           [0.08]]),
                       np.matrix([[0.298642],     #np.matrix([[0.06600],
                                  [0.70135746]]),   #           [0.15500]]),
                       np.matrix([[0.185740594],     #np.matrix([[0.02118],
                                  [0.814259405]]),   #           [0.09285]]),
                       np.matrix([[0.11273448],     #np.matrix([[0.00625],
                                  [0.8872655]])]   #           [0.04919]])]

    assert len(alphas) == len(expected_alphas)
    for i in range(4):
        assertEqualMatrices(alphas[i], expected_alphas[i])


def test_compute_alphas_normal():
    np.random.seed(0)
    data, _ = generate_continuous_data(use_default=True)
    n_hidden_states = 2
    T,_ = data.shape
    A, E, pi = hmm_fit.initialise_parameters(2, None, model='normal')
    alphas = hmm_fit.compute_alphas(data[:,0], A, E, pi, model='normal')

    assert len(alphas) == T


def test_compute_betas():
    """ Based on the 'Example of Baum-Welch Algorithm' by Larry Moss. """
    data, A, B, pi = load_words_data()
    betas = hmm_fit.compute_betas(data[:,0], A, B, pi)

    expected_betas =  [np.matrix([[0.5],         ###[np.matrix([[1.],
                                  [0.5]]),       ###            [1.]]),
                       np.matrix([[0.489583],    ### np.matrix([[0.47000],
                                  [0.510416]]),  ###            [0.49000]]),
                       np.matrix([[0.5073296],    ### np.matrix([[0.25610],
                                  [0.49267036]]),  ###            [0.24870]]),
                       np.matrix([[0.511250],    ### np.matrix([[0.13315],
                                  [0.488749]])]  ###            [0.12729]])]
    expected_betas = expected_betas[::-1]

    assert len(betas) == len(expected_betas)
    for i in range(4):
        assertEqualMatrices(betas[i], expected_betas[i])

def test_compute_gammas():
    """ Based on the 'Example of Baum-Welch Algorithm' by Larry Moss. """
    data, A, B, pi = load_words_data()
    gammas = hmm_fit.compute_gammas(data[:,0], A, B, pi)

    expected_gammas = [np.matrix([[0.81654],
                                  [0.18366]]),
                       np.matrix([[0.30488],
                                  [0.68532]]),
                       np.matrix([[0.17955],
                                  [0.82064]]),
                       np.matrix([[0.11273],
                                  [0.88727]]),]

    assert len(gammas) == len(expected_gammas)
    for i in range(4):
        assertEqualMatrices(gammas[i], expected_gammas[i])

def test_compute_xis():
    """ Based on the 'Example of Baum-Welch Algorithm' by Larry Moss. """
    data, A, B, pi = load_words_data()
    xis = hmm_fit.compute_xis(data[:,0], A, B, pi)

    expected_xis =  [np.matrix([[0.28271, 0.53383],
                                  [0.02217, 0.16149]]),
                       np.matrix([[0.10071, 0.20417],
                                  [0.07884, 0.61648]]),
                       np.matrix([[0.04584, 0.13371],
                                  [0.06699, 0.75365]]),]

    assert len(xis) == len(expected_xis)
    for i in range(3):
        assertEqualMatrices(xis[i], expected_xis[i])

def test_compute_all_gammas():
    np.random.seed(0)
    data = np.matrix([[0, 1, 1, 0, 0, 1], [1, 1, 0, 0, 1, 1], [1, 1, 1, 0, 1, 1]]).T
    A, B, pi = hmm_fit.initialise_parameters(3, 2)
    all_gammas = hmm_fit.compute_all_gammas(data, A, B, pi)

    assert all_gammas.shape[0] == 3
    assert all_gammas.shape[1] == 6

def test_compute_all_xis():
    np.random.seed(0)
    data = np.matrix([[0, 1, 1, 0, 0, 1], [1, 1, 0, 0, 1, 1], [1, 1, 1, 0, 1, 1]]).T
    A, B, pi = hmm_fit.initialise_parameters(3, 2)
    all_xis = hmm_fit.compute_all_xis(data, A, B, pi)

    assert all_xis.shape[0] == 3
    assert all_xis.shape[1] == 5
