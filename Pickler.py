import numpy as np
import os
import pandas as pd
import pickle


def oc_matrix(u, freq=None, phase=None):
    u_data = v1_data[v1_data.u == u]
    
    if freq is not None:
        u_data = u_data[u_data.grat_spat_freq == freq]
    if phase is not None:
        u_data = u_data[u_data.grat_phase == phase]
    
    M = np.zeros((len(contrasts), len(orientations)))
    Sd = np.zeros_like(M)
    
    for i, c in enumerate(contrasts):
        con_data = u_data[u_data.grat_contrast == c]
        for j, th in enumerate(orientations):
            ori_data = con_data[con_data.grat_orientation == th]
            response = np.array(ori_data.response.tolist())
            M[i,j] = np.mean(response)  # Seems inefficient
            Sd[i,j] = np.std(response)
            
    return M, Sd


k_data_path = os.path.join('Data', 'K-Data.csv')
k_data = pd.read_csv(k_data_path)

v1_data = k_data[k_data.region == 'V1']
    
print(v1_data.m.unique())
m = v1_data.m.unique()[2]
v1_data = v1_data[v1_data.m == m]
print(len(v1_data))

v1_data.head()


orientations = v1_data.grat_orientation.unique()
contrasts = v1_data.grat_contrast.unique()
spatial_frequencies = v1_data.grat_spat_freq.unique()
phases = v1_data.grat_phase.unique()
units = sorted(v1_data.u.unique())


n_u = len(units)
n_spat = len(spatial_frequencies)

tuning_curves = np.zeros([n_u * n_spat, len(contrasts), len(orientations)])


for j, u in enumerate(units):
    for i, spf in enumerate(spatial_frequencies):
        M, Sd = oc_matrix(u, spf)
        tuning_curves[j*n_spat + i] = M  # Seems inefficient

with open(os.path.join('Data', 'data_save.pkl'), 'wb') as outp:
    pickle.dump(tuning_curves, outp, pickle.HIGHEST_PROTOCOL)
