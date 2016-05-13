from tests_common import *
from generateData import *
import numpy as np
from expected_counts import *
from compute_quantities import *

"""

NOTE
-
In this file, we set up tests to make sure our code runs as it has been
intended to.
It is NOT a file where we test the results of the EM/Viterbi algorithms. This is
done in 'main.py'

"""

def test_expected_count_j_k():
    """ Based on the 'Example of Baum-Welch Algorithm' by Larry Moss. """
    data, A, B, pi = load_words_data()
    all_xis = compute_all_xis(data, A, B, pi)

    count = expected_count_j_k(0, 0, all_xis) - 0.0591176 # extra value
          # to accomodate with Moss' paper.
    expected_result = 0.74397
    assertEqualApprox(count, expected_result)

    count = expected_count_j_k(0, 1, all_xis) - 0.114950 # same
    expected_result = 1.683539
    assertEqualApprox(count, expected_result)

    count = expected_count_j_k(1, 0, all_xis) - 0.09716 # same
    expected_result = 0.272109
    assertEqualApprox(count, expected_result)

    count = expected_count_j_k(1, 1, all_xis) - 0.72876299 # same
    expected_result = 2.35282
    assertEqualApprox(count, expected_result)


def test_expected_count_m_j_l():
    """ Based on the 'Example of Baum-Welch Algorithm' by Larry Moss. """
    data, A, B, pi = load_words_data()
    all_gammas = compute_all_gammas(data, A, B, pi)

    count = expected_count_m_j_l(0, 0, all_gammas, data)
    expected_result = 1.17325
    assertEqualApprox(count, expected_result)

    count = expected_count_m_j_l(0, 1, all_gammas, data) - 0.15628604 # extra value
          # to accomodate with Moss' paper.
    expected_result = 1.516378
    assertEqualApprox(count, expected_result)

    count = expected_count_m_j_l(1, 0, all_gammas, data)
    expected_result = 1.852878
    assertEqualApprox(count, expected_result)

    count = expected_count_m_j_l(1, 1, all_gammas, data) - 0.84371396 # same
    expected_result = 2.50993
    assertEqualApprox(count, expected_result)
