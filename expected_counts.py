import numpy as np

def expected_count_1_k(k, all_gammas):
    """
    Computes E[N_k^1] as defined in Murphy's book. Where:
    E[N_k^1] = sum_i^S p(zi1 = k | xi) = sum_i^S gammas[i,0,k]
    k: the hidden state we want to infere the probability from
    all_gammas: can be regarded as a [S, T] numpy array where:
            T is the length of the Markov Chain
            S is the number of samples we have from the Markov Chain
            'gammas' is obtained using the function compute_all_gammas
    """
    S, _,_,_ = all_gammas.shape
    expected_count = 0.
    for sample in range(S):
        gammas = all_gammas[sample]
        gammas_t0 = gammas[0]
        gammas_t0_k = gammas_t0[k]

        expected_count += gammas_t0_k

    return expected_count


def expected_count_j(j, all_gammas):
    """
    Computes E[N_j] as defined in Murphy's book. Where:
    E[N_j] = sum_i^S sum_t^T p(z(i,t) = j | xi) = sum_i^S sum_t^T gammas[i,t,j]
    j: the hidden state we want to infere the probability from
    all_gammas: can be regarded as a [S, T] numpy array where:
            T is the length of the Markov Chain
            S is the number of samples we have from the Markov Chain
            'gammas' is obtained using the function compute_all_gammas
    """
    S, T,_,_ = all_gammas.shape
    expected_count = 0.
    for sample in range(S):
        for t in range(T):
            gammas = all_gammas[sample]
            gammas_t = gammas[t]
            gammas_t_j = gammas_t[j]

            expected_count += gammas_t_j

    return expected_count


def expected_count_j_k(j, k, all_xis):
    """
    Computes E[N_jk] as defined in Murphy's book. Where:
    E[N_jk] = sum_i^S sum_t=1^T p(z(i,t-1) = j, z(i,t) = k | xi)
            = sum_i^S sum_t=1^T xis[i,t,j,k]
    j,k: the hidden states we want to infere the probability from
    all_xis: can be regarded as a [S, T-1] numpy array where:
            T is the length of the Markov Chain
            S is the number of samples we have from the Markov Chain
            'xis' is obtained using the function compute_all_xis
    """
    S, T,_,_ = all_xis.shape
    expected_count = 0.
    for sample in range(S):
        for t in range(T):
            xis = all_xis[sample]
            xis_t = xis[t]
            xis_t_jk = xis_t[j,k]

            expected_count += xis_t_jk

    return expected_count


def expected_count_m_j_l(j, l, all_gammas, data):
    """
    Computes E[M_jl] as defined in Murphy's book. Where:
    E[M_jl] = sum_i^S sum_t=1^T gamma[i,t,j] * kronecker(data[t,i] = l)
    all_gammas: can be regarded as a [S, T] numpy array where:
            T is the length of the Markov Chain
            S is the number of samples we have from the Markov Chain
            'gammas' is obtained using the function compute_all_gammas
    """
    S, T, _, _ = all_gammas.shape
    expected_count = 0.
    for sample in range(S):
        for t in range(T):
            if data[t,sample] == l:
                gamma = all_gammas[sample][t][j][0]
                expected_count += gamma

    return expected_count


def expected_count_x_k(k, all_gammas, data):
    """
    Computes E[x_k] as defined in Murphy's book. Where:
    E[x_k] = sum_i^S sum_t=1^T gamma[i,t,j] * data[t,i]
    all_gammas: can be regarded as a [S, T] numpy array where:
            T is the length of the Markov Chain
            S is the number of samples we have from the Markov Chain
            'gammas' is obtained using the function compute_all_gammas
    NOTE: here x is a scalar. It can (and is supposed to) be a vector. This is
    not the case in the data we generate in this CW.
    """
    S, T, _, _ = all_gammas.shape
    expected_count = 0.
    for sample in range(S):
        for t in range(T):
            gamma = all_gammas[sample][t][k][0]
            expected_count += (gamma * data[t,sample])

    #expected_count /= np.sum(gammas) # dividing by the sum of weights
    return expected_count


def expected_count_x_x_k(k, all_gammas, data):
    """
    Computes E[xx_k.T] as defined in Murphy's book. Where:
    E[x_k] = sum_i^S sum_t=1^T gamma[i,t,j] * data[t,i] * data[t,i].T
    all_gammas: can be regarded as a [S, T] numpy array where:
            T is the length of the Markov Chain
            S is the number of samples we have from the Markov Chain
            'gammas' is obtained using the function compute_all_gammas
    NOTE: here x is a scalar. It can (and is supposed to) be a vector. This is
    not the case in the data we generate in this CW.
    """
    S, T, _, _ = all_gammas.shape
    expected_count = 0.
    for sample in range(S):
        for t in range(T):
            gamma = all_gammas[sample][t][k][0]
            expected_count += gamma * data[t,sample]**2

    return expected_count
