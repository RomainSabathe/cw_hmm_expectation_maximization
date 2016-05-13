from tests_common import *
import numpy as np
import hmm_fit
from generateData import *

"""

NOTE
-
In this file, we set up tests to make sure our code runs as it has been
intended to.
It is NOT a file where we test the results of the EM/Viterbi algorithms. This is
done in 'main.py'

"""

def test_update_A():
    """ Based on the 'Example of Baum-Welch Algorithm' by Larry Moss. """
    data, A, B, pi = load_words_data()
    all_gammas = hmm_fit.compute_all_gammas(data, A, B, pi)
    all_xis = hmm_fit.compute_all_xis(data, A, B, pi)

    updated_A = hmm_fit.update_A(data, all_gammas, all_xis)
    expected_result = np.matrix([[0.298, 0.702], [0.106, 0.894]])

    assertEqualMatrices(updated_A, expected_result)


def test_update_B():
    """ Based on the 'Example of Baum-Welch Algorithm' by Larry Moss. """
    data, A, B, pi = load_words_data()
    all_gammas = hmm_fit.compute_all_gammas(data, A, B, pi)
    all_xis = hmm_fit.compute_all_xis(data, A, B, pi)

    updated_B = hmm_fit.update_B(data, all_gammas, all_xis)
    expected_result = np.matrix([[0.4362124, 0.563787], [0.424698, 0.575301]])

    assertEqualMatrices(updated_B, expected_result)


def test_update_pi():
    """ Based on the 'Example of Baum-Welch Algorithm' by Larry Moss. """
    data, A, B, pi = load_words_data()
    all_gammas = hmm_fit.compute_all_gammas(data, A, B, pi)
    all_xis = hmm_fit.compute_all_xis(data, A, B, pi)

    updated_pi = hmm_fit.update_pi(data, all_gammas, all_xis)
    expected_result = np.matrix([[0.846], [0.154]])

    assertEqualMatrices(updated_pi, expected_result)


def test_expectation_maximization():
    """ Based on the 'Example of Baum-Welch Algorithm' by Larry Moss. """
    data, expected_A, expected_B, expected_pi = load_words_data()

    A, B, pi = hmm_fit.expectation_maximization(data, 2, 2, steps=5)

    assertEqualMatrices(A, expected_A)
    assertEqualMatrices(B, expected_B)
    assertEqualMatrices(pi, expected_pi)


def test_expectation_maximization_normal():
    np.random.seed(4)
    data, _ = generate_continuous_data(use_default=True)
    K = 2

    A, E, pi = hmm_fit.expectation_maximization(data, K, None, steps=5, model='normal')

