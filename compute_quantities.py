import numpy as np
from tools import *

def compute_alphas(data, A, B, pi, model='multinomial'):
    """
    Compute the alpha quantities as given in Murphy's book, that is so say:
    for each t, alpha_t = p(z_t | x_1:t).
    data: the observations [x_t=1, x_t=2, ... x_t=T]. It is ONE vector only.
    A: the transition matrix for the latent variables. A[i,j] = p(z_t = j | z_t-1 = i)
       A corresponds to Psi in Murphy's book.
    B: the estimated posterior probability at time t.
       B[i, j] = p(x_t = j | z_t = i)
       OR
       The matrix 'E' where E[mu] and E[sigma2] are the parameters of a normal
       distribution
    pi: the initial state distribution. pi(j) = p(z_(t=1) = j)
    """

    def compute_alpha_aux(t):
        """
        Same as compute_alpha, but here we use the variable 't'.
        When 't' equals 1, it is the base case, otherwise we do a recursive call.
        """
        if t == 0:
            if model == 'multinomial':
                evidence = B[:,data[0]]
            elif model == 'normal':
                evidence = compute_B_normal(data[0], B)

            result_unnormalised = hadamard(evidence, pi)
            return [normalise(result_unnormalised)]
        else:
            previous_alphas = compute_alpha_aux(t-1)
            previous_alpha = previous_alphas[-1]
            new_state_distribution = A.T * previous_alpha

            if model == 'multinomial':
                evidence_vector = B[:,data[t]]
            elif model == 'normal':
                evidence_vector = compute_B_normal(data[t], B)

            result_unnormalised = hadamard(evidence_vector, new_state_distribution)
            result_normalised = normalise(result_unnormalised)

            previous_alphas.append(result_normalised)
            return previous_alphas

    if isinstance(data, np.matrixlib.defmatrix.matrix):
        data = np.array(data) # making sure it is a sequence of observed variables
    T = len(data) # length of the Markov Chain
    alphas = compute_alpha_aux(T-1)

    return alphas


def compute_betas(data, A, B, pi, model='multinomial'):
    """
    Compute the beta quantities as given in Murphy's book, that is so say:
    for each t, beta_t = p(x_t+1:T | z_t=j).
    data: the observations [x_t=1, x_t=2, ... x_t=T]. It is ONE vector only.
    A: the transition matrix for the latent variables. A[i,j] = p(z_t = j | z_t-1 = i)
       A corresponds to Psi in Murphy's book.
    B: the estimated posterior probability at time t.
       B[i, j] = p(x_t = j | z_t = i)
       OR
       The matrix 'E' where E[mu] and E[sigma2] are the parameters of a normal
       distribution
    pi: the initial state distribution. pi(j) = p(z_(t=1) = j)
    """

    T = len(data) # the length of the Markov Chain
    def compute_beta_aux(t):
        """
        Same as compute_beta, but here we use the variable 't'.
        When 't' equals T, it is the base case, otherwise we do a recursive call.
        """
        if t == T-1:
            K = A.shape[0] # number of hidden states
            beta = np.matrix(np.ones([K, 1]))
            return [normalise(beta)]
        else:
            next_betas = compute_beta_aux(t+1)
            next_beta = next_betas[-1]

            if model == 'multinomial':
                evidence_vector = B[:,data[t+1]]
            elif model == 'normal':
                evidence_vector = compute_B_normal(data[t+1], B)
            beta = A * hadamard(evidence_vector, next_beta)
            beta = normalise(beta)

            next_betas.append(beta)
            return next_betas

    if isinstance(data, np.matrixlib.defmatrix.matrix):
        data = np.array(data) # making sure it is a sequence of observed variables
    betas = compute_beta_aux(0)

    return betas[::-1] # reversing so that it goes along time


def compute_gammas(data, A, B, pi, model='multinomial'):
    """
    Compute the gamma quantities as given in Murphy's book, that is so say:
    for each t, gamma_t = p(z_t | x_1:T), probability distribution of the latent
                variables given all the observations
    data: the observations [x_t=1, x_t=2, ... x_t=T]. It is ONE vector only.
    A: the transition matrix for the latent variables. A[i,j] = p(z_t = j | z_t-1 = i)
       A corresponds to Psi in Murphy's book.
    B: the estimated posterior probability at time t.
       B[i, j] = p(x_t = j | z_t = i)
       OR
       The matrix 'E' where E[mu] and E[sigma2] are the parameters of a normal
       distribution
    pi: the initial state distribution. pi(j) = p(z_(t=1) = j)
    """
    alphas = compute_alphas(data, A, B, pi, model)
    betas  = compute_betas(data, A, B, pi, model)
    gammas = []
    for alpha, beta in zip(alphas, betas):
        gamma = hadamard(alpha, beta)
        gammas.append(normalise(gamma))

    return gammas

def compute_all_gammas(data, A, B, pi, model='multinomial'):
    """
    Use 'compute_gammas' but here, data is not a column vector anymore.
    data: of size [T, S] where T is the length of the Markov Chains and S is the
          number of samples we derived from the chain.
          For now, the values of 'data' can only be discrete.
    returns: what can be considered as a [S, T] numpy array
    """
    data = np.array(data)
    T, S = data.shape
    all_gammas = []

    for i, column in enumerate(data.T):
        # column has size T. It is a whole sample of the Markov Chain
        gammas = compute_gammas(column, A, B, pi, model)
        all_gammas.append(gammas)

    return np.array(all_gammas)


def compute_xis(data, A, B, pi, model='multinomial'):
    """
    Compute the xi quantities as given in Murphy's book, that is so say:
    for each t, xi_t,t+1 = p(z_t, z_t+1 | x_1:T)
    data: the observations [x_t=1, x_t=2, ... x_t=T]. It is ONE vector only.
    A: the transition matrix for the latent variables. A[i,j] = p(z_t = j | z_t-1 = i)
       A corresponds to Psi in Murphy's book.
    B: the estimated posterior probability at time t.
       B[i, j] = p(x_t = j | z_t = i)
       OR
       The matrix 'E' where E[mu] and E[sigma2] are the parameters of a normal
       distribution
    pi: the initial state distribution. pi(j) = p(z_(t=1) = j)
    """
    alphas = compute_alphas(data, A, B, pi, model)
    betas = compute_betas(data, A, B, pi, model)
    xis = []
    if isinstance(data, np.matrixlib.defmatrix.matrix):
        data = np.array(data) # making sure it is a sequence of observed variables

    for t, (alpha, beta) in enumerate(zip(alphas, betas[1:])):
        if model == 'multinomial':
            evidence = B[:,data[t+1]]
        elif model == 'normal':
            evidence = compute_B_normal(data[t+1], B)

        post_evidence = hadamard(evidence, beta)
        total_evidence = alpha * post_evidence.T
        xi = hadamard(A, total_evidence)

        xi = xi / np.sum(xi) # normalisation

        xis.append(xi)

    return xis


def compute_all_xis(data, A, B, pi, model='multinomial'):
    """
    Use 'compute_xis' but here, data is not a column vector anymore.
    data: of size [T, S] where T is the length of the Markov Chains and S is the
          number of samples we derived from the chain.
          For now, the values of 'data' can only be discrete.
    returns: what can be considered as a [S, T] numpy array
    """
    data = np.array(data)
    T, S = data.shape
    all_xis = []

    for i, column in enumerate(data.T):
        # column has size T. It is a whole sample of the Markov Chain
        xis = compute_xis(column, A, B, pi, model)
        all_xis.append(xis)

    return np.array(all_xis)
