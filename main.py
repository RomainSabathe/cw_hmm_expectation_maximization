from hmm_fit import *
from generateData import *

np.random.seed(0)

###############################################################################

print ' ----- MULTINOMIAL HMM ------'

N = 10
T = 100
pi = np.array([0.5, 0.5])
A = np.array([[0.4, 0.6], [0.4, 0.6]])
E = np.array([[1./6, 1./6, 1./6, 1./6, 1./6, 1./6.],
              [1./10, 1./10, 1./10, 1./10, 1./10, 1./2]])

data,hidden_states = generate_discrete_data(use_default=False,
                                            N=N, T=T, pi=pi, A=A, E=E)

n_hidden_states = 2
n_visible_states = 6

A_em, B_em, pi_em = expectation_maximization(data, n_hidden_states, n_visible_states, 10)

print '- - - - - - - - - -'
print 'Transition matrix :\n\texpected'
print A
print '\n\tobtained'
print A_em

print '\n- - - - - - - - - -'
print 'Outcome matrix :\n\texpected'
print E
print '\n\tobtained'
print B_em

print '\n- - - - - - - - - -'
print 'Initial probability distribution matrix :\n\texpected'
print pi
print '\n\tobtained'
print pi_em


###############################################################################

print '\n\n\n ----- NORMAL HMM ------'


N = 10
T = 100
pi = np.array([0.5, 0.5])
A = np.array([[0.4, 0.6], [0.4, 0.6]])
E = {'mu': np.array([0.1, 0.5]),
     'sigma2': np.array([0.4, 0.8])}

data,hidden_states = generate_continuous_data(use_default=False,
                                            N=N, T=T, pi=pi, A=A, E=E)
n_hidden_states = 2

A_em, E_em, pi_em = expectation_maximization(data, n_hidden_states, None, 30, model='normal')

print '- - - - - - - - - -'
print 'Transition matrix :\n\texpected'
print A
print '\n\tobtained'
print A_em

print '\n- - - - - - - - - -'
print 'Normal distribution parameter (mu) :\n\texpected'
print E['mu']
print '\n\tobtained'
print E_em['mu']

print '\n- - - - - - - - - -'
print 'Normal distribution parameter (sigma squared) :\n\texpected'
print E['sigma2']
print '\n\tobtained'
print E_em['sigma2']

print '\n- - - - - - - - - -'
print 'Initial probability distribution matrix :\n\texpected'
print pi
print '\n\tobtained'
print pi_em
