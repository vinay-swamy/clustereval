import os
import pandas as pd
import pytest
import clustereval as ce

@pytest.fixture
def clus():
    data= pd.read_csv('clustereval/data/testdata.csv.gz')
    clus = [ce.cluster.run_clustering(data, 'louvain',  1.0, i, perturb=True, edge_permut_frac=.5,
                                      weight_permut_range=(.6, 1.6), local_pruning=False,
                                      global_pruning=False, min_cluster_size=10, verbosity=1) for i in range(30, 60, 5)]
    return clus

@pytest.fixture 
def clus_dicts(clus):
    return[i.assign(Barcode = lambda x: x.Barcode.astype(str), 
                    labels=lambda x: x.labels.astype(str)).to_dict('list') for i in clus]

def test_singleway_metric_calc(clus_dicts):
    ref_clu = clus_dicts[0]
    clus_dicts = clus_dicts[1:]
    ce.calc_metrics.oneway_metric_calculation(ref_clu, clus_dicts, 'test', 'test_singleway.csv')

def test_pairwise_metric_frommem(clus_dicts):
    names = [str(i) for i in range(30, 60, 5)]
    ce.calc_metrics.pairwise_metric_calculation_frommem(clus_dicts,names, 2)


def test_pairwise_metric_fromdisk(clus):
    [clus[i].to_csv(f'test_pwmf_{str(i)}.csv.gz', header = False, index = False) for i in range(len(clus))]
    ce.calc_metrics.pairwise_metric_calculation_fromdisk('test_pwmf_*.csv.gz', 2)
    try:
        ce.calc_metrics.pairwise_metric_calculation_fromdisk(
        'ASDfsadfasdfasdfasdfasfasdfasdfasdfasdfasdfdsfd', 2)
        assert 1==2
    except:
        pass

def test_clean_up(clus):
    os.remove('test_singleway.csv') 
    [os.remove(f'test_pwmf_{str(i)}.csv.gz') for i in range(len(clus))]
    return
    



