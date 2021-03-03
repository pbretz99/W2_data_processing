# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 09:43:45 2021

@title: Oscillation Removing Functions
@author: Philip Bretz
"""

import numpy as np
import statsmodels.api as sm

def clean(data, skip, window_size, freq, tol=0.00001):
    cleaned_data = np.zeros(data.shape)
    N = data.size
    init, final = 0, window_size
    while final < N:
        cleaned_data[init:final] = clean_single(data[init:final], freq, tol)
        init += skip
        final += skip
    cleaned_data[(final-skip):N] = data[(final-skip):N]
    return cleaned_data

def clean_single(data, freq, tol=0.00001):
    # data is a numpy array
    tol_array = np.array([tol]*data.size)
    t = np.arange(data.size)
    s = np.sin(freq*2*np.pi*t)
    c = np.cos(freq*2*np.pi*t)
    X = np.stack((s, c), axis=1)
    X = sm.add_constant(X)
    model = sm.OLS(data, X)
    results = model.fit()
    if results.f_pvalue <= 0.05:
        ret = data-results.fittedvalues+results.params[0]
        ret = np.maximum(ret, tol_array)
    else:
        ret = data
    return ret