from tests_common import *
from generateData import *
from tools import *
import numpy as np
from initialisation import *

"""

NOTE
-
In this file, we set up tests to make sure our code runs as it has been
intended to.
It is NOT a file where we test the results of the EM/Viterbi algorithms. This is
done in 'main.py'

"""

def load_K():
    K = 3 # number of possible states of the latent variables
    return K


def load_N():
    N = 10 # number of possible states of the visible variables
    return N

def test_initialise_A():
    K = load_K()

    A = initialise_A(K)
    assert A.shape == (3, 3) # checking the shape

    for i in range(3): # checking that rows are normalised
        assertEqualApprox(A[i,:].sum(), 1.)


def test_initialise_B():
    K = load_K()
    N = load_N()

    B = initialise_B(K, N)
    assert B.shape == (3, 10) # checking shape

    for i in range(3): # checking that rows are normalised
        assertEqualApprox(B[i,:].sum(), 1.)


def test_initialise_pi():
    K = load_K()

    pi = initialise_pi(K)
    assert pi.shape == (3, 1) # checking shape
    assertEqualApprox(pi.sum(), 1.) # checking normality
