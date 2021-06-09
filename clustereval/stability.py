#%%
import pandas as pd
import numpy as np 
import glob
import os 
import re
import pickle
from multiprocessing import Pool

def entropy(exp): # both are dfs with two columsn, Barcode,cluster
    # calc H_tot
    if exp.shape[1] > 2:
        exp = exp.iloc[:,:2]
    entropy = (exp
             .groupby("labels")
             .count()
             .reset_index(drop=True)
             .assign(prop=lambda x: x/exp.shape[0],
                     H=lambda x: x['prop'] * np.log(x['prop']))
             ['H'].sum()*-1)
    return entropy
    

def H_k(ref, exp):
    if exp.shape[1] > 2:
        exp = exp.iloc[:,:2]
        ref = ref.iloc[:,:2]
    exp = exp[exp.Barcode.isin(ref['Barcode']) ]
    if exp.shape[0] == 0:
        return 0
    else:
        h_k = entropy(exp)
        return h_k


def calc_stability(tup):
    ref = tup[0].iloc[:,:2]
    meta_df = tup[1]
    exp_df_list = tup[2]
    runname = tup[3]
    # try:
    H_k_scores = np.asarray([
        [H_k(group[1], exp.iloc[:,:2]) for exp in exp_df_list] / meta_df['H_tot']
        for group in ref.groupby("labels")
    ]).sum(axis=1)
    H_k_scores = 1 - (H_k_scores / len(exp_df_list))
    clusters = [group[0] for group in ref.groupby("labels")]
    return pd.DataFrame.from_dict({runname: clusters, 'H_k_scores': H_k_scores})
# %%
