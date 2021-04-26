# Clustereval: metrics for evaluating graph partitioning algorithmn

## Motivation
Graph partitioning algorithms divide a graph in set of partitions, where partitions are more connected to each other than the rest of the graph. Any tabular data can be converted into a graph via an approximate nearest neighbor(ANN), where observations that are closer to each other in space are connected. The combination of ANN + graph partitioning is an efficient way to partition large datasets. A major step in this process is the choice of the number of nearest neighbors to obtain(k) and hyperparameters for the clustering algorithm. This pacakge aims to assist in the task of choosing the best hyperparameters for this style of clustering methods. 

A proposed method for evaluating the results of clustering method is by using perturbation experiments. A data set is clustered to obtain an initial set of labels, then portions of the data are randomly changed, and the data is re-clustered. This perturbation is repeated several times, and the original set of labels are compared to the perturbed sets of labels to see how sensitive the original labels are to small changes in the data. 

## The Metrics. 
To facilitate comparisons between sets of labels, this package provides two  metrics Stability and Purity, which are calculated on a per-cluster basis. In general terms, Stability is useful for quantifying how often a given cluster is split into smaller ones. Purity is useful for identifying how often a cluster is merged into another, larger cluster. More formal definitions of the metrics are available here


## Example usage
`Clustereval` provides a simple interface to conduct perturbations in graphs and calculate purity and stability metrics based on those perturbations.

```
import pandas as pd
import clustereval as ce
data = pd.read_csv('clustereval/data/testdata.csv.gz')
metrics,labels, perturbations = ce.cluster.run_full_experiment(reduction = data, 
                                         alg = 'louvain', 
                                         k=30,
                                         n_perturbations=23,
                                         edge_permut_frac=.05,
                                         experiment_name='example'
                                         )
metrics
```
|   cluster_ids |   stability |   purity | exp_name   |
|--------------:|------------:|---------:|:-----------|
|             3 |    0.708558 | 0.53825  | example    |
|             4 |    0.943322 | 0.666248 | example    |
|             0 |    0.608236 | 0.991759 | example    |
|             2 |    1        | 0.972109 | example    |
|             1 |    1        | 0.944761 | example    |





