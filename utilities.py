# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 08:42:02 2021

@title: Utilities
@author: Philip Bretz
"""

import numpy as np

import kalman_filter as kf

'''
Creates a default state-space model
for filtering the W2 curves, as a 
random walk with noise process
'''
def default_model():
    # Create default model
    V, W = 0.5, 0.01
    W = np.array([[W]])
    space_params = [np.array([[1]]), np.array([[V]])]
    state_params = [np.array([[1]]), W]
    prior = [np.array([[0]]), 
             np.array([[1]])]
    model = kf.DLM(state_params, space_params, prior)
    return model
