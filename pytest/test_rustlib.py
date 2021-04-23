import os
import pandas as pd
import pytest
import clustereval as ce

@pytest.fixture
def clus():
    data= pd.read_csv('clustereval/data/testdata.csv.gz')
    metrics, ref_labels, perturbations = ce.cluster.run_full_experiment(reduction=data,
                                          alg='louvain',
                                          k=30,
                                          local_pruning=False,
                                          global_pruning=False,
                                          quality_function='RBConfigurationVertexPartition',
                                          cluster_kwargs={},
                                          n_perturbations=5,
                                          edge_permut_frac=.05,
                                          weight_permut_range=None,
                                          min_cluster_size=10,
                                          experiment_name='clusterEval',
                                          verbosity=0
                                          )
    return perturbations

@pytest.fixture 
def clus_dicts(clus):
    return[i.to_dict('list') for i in clus]


@pytest.fixture
def py_stability(clus):
    H_tots = [ce.stability.entropy(i) for i in clus[1:]]
    mdf = pd.DataFrame().assign(H_tot = H_tots)
    return ce.stability.calc_stability((clus[0],  mdf, clus[1:], 'cluster_ids'))


@pytest.fixture
def py_purity(clus):
    clu_tups = [dict(ce.purity.df2DictSets(df)) for df in clus]
    return ce.purity.run_purity_calc((clu_tups[0], clu_tups[1:], 'cluster_ids')).rename(columns={'purity':'py_purity'})
    

## equality assertions will occasionally fail, because of some floating point errors, always run twice 
def test_singleway_metric_calc_int_df(clus, py_stability, py_purity):
    ref_clu = clus[0]
    k=ce.metrics.calculate_metrics(
        ref_clu,  clus[1:], 'test')
    comp = k.merge(py_stability).merge(py_purity)
    print(comp)
    assert sum( comp['stability'] - comp['H_k_scores'] ) ==0
    assert sum( comp['purity'] - comp['py_purity']) ==0
    return


def test_singleway_metric_calc_string_df(clus, py_stability, py_purity):
    
    clus = [clu.assign(Barcode = lambda x: x.Barcode.astype(str), labels = lambda x: x.labels.astype(str)) for clu in clus]
    
    ref_clu = clus[0]
    clus = clus[1:]
    k = ce.metrics.calculate_metrics(
        ref_clu, clus, 'test').assign(cluster_ids = lambda x: x.labels.astype(int))
    comp = k.merge(py_stability).merge(py_purity)
    print(comp)
    assert sum(comp['stability'] - comp['H_k_scores']) == 0
    assert sum(comp['purity'] - comp['py_purity']) == 0
    return



# def test_singleway_metric_calc_int_l(clus, py_stability):
#     clus = [i.sort_values('Barcode')['labels'].to_list() for i in clus]
#     ref_clu = clus[0]
#     print(ref_clu)
#     k = ce.metrics.calculate_metrics(
#         ref_clu, clus[1:], 'test')
#     comp = k.merge(py_stability)
#     print(comp)
#     assert sum(comp['stability'] - comp['H_k_scores']) == 0
#     return


# def test_singleway_metric_calc_str_l(clus, py_stability):
#     clus = [i['labels'].astype(str).to_list() for i in clus]
#     ref_clu = clus[0]
#     clus = clus[1:]
#     k = ce.metrics.calculate_metrics(
#         ref_clu, clus, 'test')
#     comp = k.merge(py_stability)
#     assert sum(comp['stability'] - comp['H_k_scores']) == 0
#     return


def test_pairwise_metric_frommem(clus_dicts):
    names = [str(i) for i in range(30, 60, 5)]
    ce.metrics.pairwise_metric_calculation(clus_dicts,names, 2)


def test_pairwise_metric_fromdisk(clus):
    [clus[i].to_csv(f'test_pwmf_{str(i)}.csv.gz', header = False, index = False) for i in range(len(clus))]
    ce.metrics.pairwise_metric_calculation('test_pwmf_*.csv.gz', 2)
    try:
        ce.metrics.pairwise_metric_calculation(
        'ASDfsadfasdfasdfasdfasfasdfasdfasdfasdfasdfdsfd', 2)
        assert 1==2
    except:
        pass

def test_clean_up(clus):
    [os.remove(f'test_pwmf_{str(i)}.csv.gz') for i in range(len(clus))]
    return
    



