# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 09:43:45 2021

@title: Code for Processing W2 Curves
@author: Philip Bretz
"""

# Libraries
import json
import numpy as np
import pandas as pd

# Local Code
import clean_osc as cl
import kalman_filter as kf
import utilities as ut

# Open W2 data (from json file)
with open('w2_curves.json') as f:
  data = json.load(f)
W2B0 = np.array(data['w2b0'])
W2B1 = np.array(data['w2b1'])

# Run Oscillation Remover
# Skip every 10 frames, window size 25, and remove 0.17 Hz
W2B0_cleaned = cl.clean(W2B0, 10, 25, 0.17)
W2B1_cleaned = cl.clean(W2B1, 10, 25, 0.17)

# Run Kalman Filter on log of W2
log_W2B0, log_W2B1 = np.log(W2B0_cleaned), np.log(W2B1_cleaned)
model_B0, model_B1 = ut.default_model(), ut.default_model()
fit_values_B0, fit_var = model_B0.kfilter(log_W2B0)
fit_values_B1, fit_var = model_B0.kfilter(log_W2B1)

# Transform back to original scale
W2B0_filtered, W2B1_filtered = np.exp(fit_values_B0[::, 0, 0]), np.exp(fit_values_B1[::, 0, 0])
W2B0_filter_error, W2B1_filter_error = W2B0_cleaned-W2B0_filtered, W2B1_cleaned-W2B1_filtered

# Save important info in dictionary structure (for json)
cleaned_data = {'w2b0_cleaned': W2B0_cleaned.tolist(),
                'w2b1_cleaned': W2B1_cleaned.tolist(),
                'w2b0_filtered': W2B0_filtered.tolist(),
                'w2b1_filtered': W2B1_filtered.tolist(),
                'w2b0_filter_error': W2B0_filter_error.tolist(),
                'w2b1_filter_error': W2B1_filter_error.tolist()}

# Save to json file
with open('w2_filtered.json', 'w') as write_file:
    json.dump(cleaned_data, write_file)
