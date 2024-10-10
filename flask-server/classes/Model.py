from hmmlearn import hmm
import numpy as np

class Model:
    def __init__(self, Class, label, m_num_of_HMMStates, m_num_of_mixtures, m_transmatPrior, m_startprobPrior,
                 m_covarianceType='diag', m_n_iter=10, n_features_traindata=6):
        self.traindata = np.zeros((0, n_features_traindata))
        self.Class = Class
        self.label = label
        self.model = hmm.GMMHMM(n_components=m_num_of_HMMStates, n_mix=m_num_of_mixtures,
                                transmat_prior=m_transmatPrior, startprob_prior=m_startprobPrior,
                                covariance_type=m_covarianceType, n_iter=m_n_iter)
