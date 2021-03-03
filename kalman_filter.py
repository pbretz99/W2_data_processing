# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 17:19:04 2021

@title: Kalman Filter mk. 2
@author: Philip Bretz
"""

import numpy as np

# Note: all parameters take the form [m, sd]
class DLM:
    def __init__(self, state_params, space_params, state_prior):
        self.state_params = state_params
        self.space_params = space_params
        self.filter_params = state_prior
    
    def update(self, y):
        state, space = self.state_params, self.space_params
        G, G_trans, W = state[0], state[0].transpose(), state[1]
        F, F_trans, V = space[0], space[0].transpose(), space[1]
        # One-step ahead state
        a = G.dot(self.filter_params[0])
        R = G.dot(self.filter_params[1].dot(G_trans))+W
        # One-step ahead predictive
        f = F.dot(a)
        Q = F.dot(R.dot(F_trans))+V
        # Filtered parameters
        Q_inv = np.linalg.inv(Q)
        m = a+R.dot(F_trans.dot(Q_inv.dot(y-f)))
        C = R-R.dot(F_trans.dot(Q_inv.dot(F.dot(R))))
        self.filter_params = [m, C]
        return [m, C]
    
    def kfilter(self, Y):
        fit, var = [], []
        for y in Y:
            current_params = self.update(y)
            fit.append(current_params[0])
            var.append(current_params[1])
        return np.array(fit), np.array(var)
    
    def N_step_ahead(self, N):
        state = self.state_params
        G, G_trans, W = state[0], state[0].transpose(), state[1]
        [m, C] = self.filter_params
        for i in range(N):
            m_new = G.dot(m)
            C_new = G.dot(C.dot(G_trans))+W
            m, C = m_new, C_new
        return [m, C]