import os
import pandas as pd
import pytest
import clustereval as ce 

@pytest.fixture
def data():
    return pd.read_csv('clustereval/data/testdata.csv.gz')

def test_vanilla_cluster_louvain(data):
    ce.cluster.run_clustering(data, 'louvain',  1.0, 30, perturb=False,edge_permut_frac=None, 
                              weight_permut_range=None, local_pruning=False,
                              global_pruning=False, min_cluster_size=10, verbosity=0)


def test_vanilla_cluster_leiden(data):
    ce.cluster.run_clustering(data, 'leiden',  1.0, 30, perturb=False, edge_permut_frac=None, 
                              weight_permut_range=None, local_pruning=False,
                              global_pruning=False, min_cluster_size=10, verbosity=0)


def test_full_cluster_louvain_edge_perturb(data):
    ce.cluster.run_clustering(data, 'louvain',  1.0, 30, perturb=True, edge_permut_frac=.05, 
                              weight_permut_range=None, local_pruning=True,
                              global_pruning=True, min_cluster_size=10, verbosity=0)
def test_full_cluster_louvain_weight_perturb(data):
    ce.cluster.run_clustering(data, 'louvain',  1.0, 30, perturb=True,edge_permut_frac=None, 
                              weight_permut_range=(.4, 1.4), local_pruning=True,
                              global_pruning=True, min_cluster_size=10, verbosity=0)


def test_full_cluster_leiden_edge_perturb(data):
    ce.cluster.run_clustering(data, 'leiden',  1.0, 30, perturb=True,edge_permut_frac=.05, 
                              weight_permut_range=None, local_pruning=True,
                              global_pruning=True, min_cluster_size=10, verbosity=0)
def test_full_cluster_leiden_weight_perturb(data):
    ce.cluster.run_clustering(data, 'leiden',  1.0, 30, perturb=True, edge_permut_frac=None,
                              weight_permut_range=(.4, 1.4), local_pruning=True,
                              global_pruning=True, min_cluster_size=10, verbosity=0)

def test_dup_row_error_FAILS():
    data = pd.read_csv('clustereval/data/testdata.csv.gz', index_col=0)
    try:
        ce.cluster.run_clustering(data, 'louvain',  1.0, 30, perturb=False,edge_permut_frac=None, 
                              weight_permut_range=None, local_pruning=False,
                              global_pruning=False, min_cluster_size=10, verbosity=0)
        assert 1==2
    except ce.cluster.DuplicateRowError:
        pass

def test_umap(data):
    clu_obj = ce.cluster.ClusterExperiment(data ,verbosity=2)
    clu_obj.buildNeighborGraph(knn=10, nn_space='l2', ef_construction=150,
                               local_pruning=True, global_pruning=True, jac_std_global='median', dist_std_local = 3)
    #embedding = clu_obj.run_UMAP()