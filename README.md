Coursework - Hidden Markov Model and Expectation Maximization
==============================================================

How to run the code
------------

The file main.py offers to see the computation of both multinomial HMM and normal HMM, as well as a comparison between what has been estimated and the real parameter values. 
To run it, just use the command *python main.py*


About the code
----------------

Each file is associated with a test file. To run the tests, one can use the package *py.test* and run the command:
py.test

Apart from that, here are the role of the different files:
1. *main*: runs a simulation and tries to fit a HMM.
2. *tests_common*: some useful commands used by the test files.
3. *tools*: useful functions used by different files.
4. *initialisation*: creates initial matrices A, B and mu (or E for normal HMM).
5. *compute_quantities*: provides functions to compute alphas, betas, gammas and xis. 
6. *expected_counts*: computes expected values that are used when updating the parameters.
7. *hmm_fit*: all functions to run expectation maximization.
8. *generateData*: generates samples from a markov chain.

References
-----------

- Kevin P. Murphy, "Inference in HMMs," *Machine Learning: a Probabilistic Perspective* pp. 606-617, 2013 ([link](https://www.cs.ubc.ca/~murphyk/MLbook/)).
