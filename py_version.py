import pandas as pd
import numpy as np 
import glob
import os 
import re
import pickle
from multiprocessing import Pool

def entropy(exp): # both are dfs with two columsn, Barcode,cluster
    # calc H_tot
    entropy = (exp
             .groupby("labels")
             .count()
             .reset_index(drop=True)
             .assign(prop=lambda x: x/exp.shape[0],
                     H=lambda x: x['prop'] * np.log(x['prop']))
             ['H'].sum()*-1)
    return entropy
    

def H_k(ref, exp):
    exp = exp[exp.Barcode.isin(ref['Barcode']) ]
    if exp.shape[0] == 0:
        return 0
    else:
        h_k = entropy(exp)
        return h_k


def calc_stability(tup):
    ref = tup[0]
    meta_df = tup[1]
    exp_df_list = tup[2]
    runname = tup[3]
    # try:
    H_k_scores = np.asarray([
        [H_k(group[1], exp) for exp in exp_df_list] / meta_df['H_tot']
        for group in ref.groupby("labels")
    ]).sum(axis=1)
    H_k_scores = 1 - (H_k_scores / len(exp_df_list))
    clusters = [group[0] for group in ref.groupby("labels")]
    return pd.DataFrame.from_dict({runname: clusters, 'H_k_scores': H_k_scores})

files = glob.glob('cluster_out/*_.csv.gz')
test_clusters  = [ pd.read_csv(i, names = ['Barcode', 'labels']) for i in files]
scores = pd.DataFrame({'H_tot':  [entropy(i) for i in test_clusters]})


exp_stability_gen = (
    calc_stability( 
    (test_clusters[i], 
     scores.drop(axis=1, index=i), test_clusters[0:i] +
     test_clusters[(i+1):], 
     "sd") 
                     )
    for i in range(len(test_clusters))
     )
    #]
pool = Pool(16)
exp_stability_dfs = pool.map(calc_stability, exp_stability_gen)
print([i.H_k_scores.sum() for i in exp_stability_gen])
