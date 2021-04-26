import os
import pandas as pd
import pytest
import clustereval as ce 

@pytest.fixture
def data():
    return pd.read_csv('clustereval/data/testdata.csv.gz')

def test_vanilla_cluster_louvain(data):
    ce.cluster.run_full_experiment(reduction = data, 
                                   alg = 'louvain', 
                                   k=30, 
                                   global_pruning_jac_threshold=None, 
                                   local_pruning_dist_threshold=None,
                                   quality_function='RBConfigurationVertexPartition',
                                   cluster_kwargs={},
                                   n_perturbations=0, 
                                   edge_permut_frac=None, 
                                   weight_permut_range=None, 
                                   min_cluster_size=10, 
                                   experiment_name='clusterEval', 
                                   verbosity=0
                                   )


def test_louvain_prune(data):
    ce.cluster.run_full_experiment(reduction=data,
                                   alg='louvain',
                                   k=30,
                                   global_pruning_jac_threshold='median', 
                                   local_pruning_dist_threshold=3,
                                   quality_function='ModularityVertexPartition',
                                   cluster_kwargs={},
                                   n_perturbations=0,
                                   edge_permut_frac=None,
                                   weight_permut_range=None,
                                   min_cluster_size=10,
                                   experiment_name='clusterEval',
                                   verbosity=0
                                   )


def test_louvain_alt_quality_function(data):
    ce.cluster.run_full_experiment(reduction=data,
                                   alg='leiden',
                                   k=30,
                                   global_pruning_jac_threshold=None, 
                                   local_pruning_dist_threshold=None,
                                   quality_function='RBConfigurationVertexPartition',
                                   cluster_kwargs={},
                                   n_perturbations=0,
                                   edge_permut_frac=None,
                                   weight_permut_range=None,
                                   min_cluster_size=10,
                                   experiment_name='clusterEval',
                                   verbosity=0
                                   )


def test_vanilla_cluster_leiden(data):
    ce.cluster.run_full_experiment(reduction=data,
                                   alg='leiden',
                                   k=30,
                                   global_pruning_jac_threshold=None, 
                                   local_pruning_dist_threshold=None,
                                   quality_function='RBConfigurationVertexPartition',
                                   cluster_kwargs={'resolution_parameter': 1.0, 'n_iterations':5},
                                   n_perturbations=0,
                                   edge_permut_frac=None,
                                   weight_permut_range=None,
                                   min_cluster_size=10,
                                   experiment_name='clusterEval',
                                   verbosity=0
                                   )


def test_leiden_prune(data):
    ce.cluster.run_full_experiment(reduction=data,
                                   alg='leiden',
                                   k=30,
                                   global_pruning_jac_threshold=.2, 
                                   local_pruning_dist_threshold=3,
                                   quality_function='RBConfigurationVertexPartition',
                                   cluster_kwargs={
                                       'resolution_parameter': 1.0, 'n_iterations': 5},
                                   n_perturbations=0,
                                   edge_permut_frac=None,
                                   weight_permut_range=None,
                                   min_cluster_size=10,
                                   experiment_name='clusterEval',
                                   verbosity=0
                                   )


def test_leiden_alt_quality_function(data):
    ce.cluster.run_full_experiment(reduction=data,
                                   alg='leiden',
                                   k=30,
                                   global_pruning_jac_threshold=None,
                                   local_pruning_dist_threshold=None,
                                   quality_function='ModularityVertexPartition',
                                   cluster_kwargs={'n_iterations': 5},
                                   n_perturbations=0,
                                   edge_permut_frac=None,
                                   weight_permut_range=None,
                                   min_cluster_size=10,
                                   experiment_name='clusterEval',
                                   verbosity=0
                                   )



def test_edge_perturb(data):
    ce.cluster.run_full_experiment(reduction=data,
                                   alg='louvain',
                                   k=30,
                                   global_pruning_jac_threshold=None,
                                   local_pruning_dist_threshold=None,
                                   quality_function='RBConfigurationVertexPartition',
                                   cluster_kwargs={},
                                   n_perturbations=1,
                                   edge_permut_frac=.05,
                                   weight_permut_range=None,
                                   min_cluster_size=10,
                                   experiment_name='clusterEval',
                                   verbosity=0
                                   )

def test_weight_perturb(data):
    ce.cluster.run_full_experiment(reduction=data,
                                   alg='leiden',
                                   k=30,
                                   global_pruning_jac_threshold=None,
                                   local_pruning_dist_threshold=None,
                                   quality_function='RBConfigurationVertexPartition',
                                   cluster_kwargs={
                                       'resolution_parameter': 1.0, 'n_iterations': 5},
                                   n_perturbations=2,
                                   edge_permut_frac=None,
                                   weight_permut_range=(.5,1.5),
                                   min_cluster_size=10,
                                   experiment_name='clusterEval',
                                   verbosity=0
                                   )
    


def test_dup_row_error_fails():
    data = pd.read_csv('clustereval/data/testdata.csv.gz', index_col=0)
    try:
        ce.cluster.run_full_experiment(reduction=data,
                                       alg='leiden',
                                       k=30,
                                       global_pruning_jac_threshold=None,
                                       local_pruning_dist_threshold=None,
                                       quality_function='RBConfigurationVertexPartition',
                                       cluster_kwargs={
                                           'resolution_parameter': 1.0, 'n_iterations': 5},
                                       n_perturbations=2,
                                       edge_permut_frac=None,
                                       weight_permut_range=(.5, 1.5),
                                       min_cluster_size=10,
                                       experiment_name='clusterEval',
                                       verbosity=0
                                       )
        assert 1==2
    except ce.cluster.DuplicateRowError:
        pass

# def test_umap(data):
#     clu_obj = ce.cluster.ClusterExperiment(data ,verbosity=2)
#     clu_obj.buildNeighborGraph(knn=10, nn_space='l2',
#                                local_pruning=True, global_pruning=True, jac_std_global='median', dist_std_local = 3)
#     embedding = clu_obj.run_UMAP()


def test_unsorted_metric_input_fails(data):
    metrics, labels, pertubations =  ce.cluster.run_full_experiment(reduction=data,
                                   alg='leiden',
                                   k=30,
                                   global_pruning_jac_threshold=None, 
                                   local_pruning_dist_threshold=None,
                                   quality_function='RBConfigurationVertexPartition',
                                   cluster_kwargs={
                                       'resolution_parameter': 1.0, 'n_iterations': 5},
                                   n_perturbations=2,
                                   edge_permut_frac=None,
                                   weight_permut_range=(.5, 1.5),
                                   min_cluster_size=10,
                                   experiment_name='clusterEval',
                                   verbosity=0
                                   )
    labels = labels.sample(labels.shape[0])
    try:
        ce.metrics.calculate_metrics(labels, pertubations)
    except:
        pass
    return
