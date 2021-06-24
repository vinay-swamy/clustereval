import numpy as np
import pandas as pd
import hnswlib
from scipy.sparse import csr_matrix
import igraph as ig
import time
import os 
import sys
import copy
import importlib
from .metrics import calculate_metrics
from joblib import Parallel, delayed
def reassign_small_cluster_cells(labels, small_pop_list, small_cluster_list, neighbor_array):
    for small_cluster in small_pop_list:
        for single_cell in small_cluster:
            old_neighbors = neighbor_array[single_cell]
            group_of_old_neighbors = labels[old_neighbors]
            group_of_old_neighbors = list(group_of_old_neighbors.flatten())
            available_neighbours = set(
                group_of_old_neighbors) - set(small_cluster_list)
            if len(available_neighbours) > 0:
                available_neighbours_list = [value for value in group_of_old_neighbors if
                                             value in list(available_neighbours)]
                best_group = max(available_neighbours_list,
                                 key=available_neighbours_list.count)
            else:
                best_group = max(set(group_of_old_neighbors),
                                 key=group_of_old_neighbors.count)
            labels[single_cell] = best_group
    return labels


class SimpleMessager:
    def __init__(self, name, verbosity):
        self.name=name
        if verbosity  > 0:
            self.level = 'DEBUG'
        else:
            self.level = 'INFO'
        self.always_print = set(['WARNING', 'ERROR'])
    
    def message(self, message, level):
        if (level == self.level) or (level in self.always_print):
            outstr = f'{self.name}:{level}:{message}'
            print(outstr)

class DuplicateRowError(Exception):
    pass

class ClusterExperiment:
    """This is the main python class to do Nearest Neighbor(NN) graph-based clustering and run perturbation experiments.
    :param data: A numeric Pandas DataFrame to cluster. Theoretically could also be a 2D numpy array, but have not tested
    :type data: numeric Pandas DataFrame
    :param knn: The number of nearest neighbors to use for NN graph creation
    :type knn: int
    """
    def __init__(self, data, verbosity):
        self.data = data
        self.time_smallpop = 15
        self.step=0
        self.m=SimpleMessager('ClusterExperiment', verbosity)

    
    def buildNeighborArray(self, max_knn, nn_space,  ef_construction=150, nthreads=1):
        """Build an approximate nearest neighbor graph(NN graph) from input data. Implements local and global edge pruning methods from PARC.

        :param knn: Number of nearest neighbors to build NN graph 
        :type knn: int
        :param nn_space: distance metric to use for NN graph. One of ['l2', 'cosine', 'ip']
        :type nn_space: [str]
        :param global_pruning_jac_threshold: remove jaccard-weighted edges from NN graph below a threshold.Skipped if None, defaults to None
        :type global_pruning_jac_threshold: [float], [0,1] optional
        :param local_pruning_dist_threshold: remove edges that are this many standard deviations away from mean distance, for each node. Skipped if None ; defaults to None
        :type local_pruning_dist_threshold: [float], optional
        :param ef_construction: tune NN accuracy, defaults to 150
        :type ef_construction: int, optional
        :param nthreads: [description], defaults to 1
        :type nthreads: int, optional
        :raises DuplicateRowError: [description]
        """        
           
        self.m.message('Building NN graph', 'DEBUG')
        self.max_knn=max_knn
        ef_query = max(100, max_knn + 1)
        num_dims = self.data.shape[1]
        n_elements = self.data.shape[0]
        p = hnswlib.Index(space=nn_space, dim=num_dims)
        p.set_num_threads(nthreads)

        if (num_dims > 30) & (n_elements <= 50000):
            p.init_index(max_elements=n_elements, ef_construction=ef_construction,
                             M=48)  # good for scRNA seq where dimensionality is high
        else:
            p.init_index(max_elements=n_elements, ef_construction=ef_construction, M=24 )
        
        p.add_items(self.data)
        p.set_ef(ef_query)
        neighbor_array, distance_array = p.knn_query(self.data, max_knn)
        self.nn_neighbor_array = neighbor_array
        self.nn_distance_array = distance_array

    def neighborArray2Graph(self,target_knn, global_pruning_jac_threshold=None, local_pruning_dist_threshold=None,):
        neighbor_array = self.nn_neighbor_array[:,:target_knn]
        distance_array = self.nn_distance_array[:,:target_knn]
        self.target_knn = target_knn
        n_neighbors = neighbor_array.shape[1]
        n_cells = neighbor_array.shape[0]
        
        if np.sum(distance_array[:,1:] ==0) > 0:
            self.m.message('There are likely duplicate rows in the input data. remove and re-try', 'ERROR')
            raise DuplicateRowError

        if local_pruning_dist_threshold is not None:
            # for each cell, keep cells that are below dist_std_local standard deviations from mean 
            # for cells above threshold, set to 0. Then, when we select non-zero values from the sparse matrix
            # these pruned values will be dropped
            row_thresh = np.mean(distance_array, axis=1) + \
                local_pruning_dist_threshold  * np.std(distance_array, axis=1)
            keep_cells = distance_array < row_thresh[:,None]
            distance_array = np.where(keep_cells, distance_array, 0)
    
        row_list = np.transpose( np.ones((n_neighbors, n_cells)) * range(0, n_cells)).flatten()
        col_list = neighbor_array.flatten()
        weight_list = distance_array.flatten()# distance array will be 0 at loops
        csr_array = csr_matrix((weight_list, (row_list, col_list)),
                               shape=(n_cells, n_cells))
    
        sources, targets = csr_array.nonzero()

        edgelist = list(zip(sources, targets))
        nn_graph = ig.Graph(edgelist)
        edgelist_copy = edgelist.copy()
        edge_list_copy_array = np.asarray(edgelist_copy)
        sim_list = nn_graph.similarity_jaccard(pairs=edgelist_copy)
        sim_list_array = np.asarray(sim_list)

        if global_pruning_jac_threshold is not None:
            # remove edges below an edge weight threshold. Edges betwen poorly connected nodes wil have low edge weight 
            self.m.message('Running Global Edge Pruning', 'DEBUG')
            n_elements = self.data.shape[0]
            if global_pruning_jac_threshold == 'median':
                threshold = np.median(sim_list)
            else:
                threshold = np.mean(sim_list) - global_pruning_jac_threshold * np.std(sim_list)
            strong_locs = np.where(sim_list_array > threshold)[0]
            new_edgelist = list(edge_list_copy_array[strong_locs])
            sim_list_new = list(sim_list_array[strong_locs])

            nn_graph = ig.Graph(n=n_elements, edges=list(new_edgelist),
                            edge_attrs={'weight': sim_list_new})
        else:
            nn_graph = nn_graph = ig.Graph(
                edgelist, edge_attrs={'weight': sim_list_array})
            # print('Share of edges kept after Global Pruning %.2f' % (len(strong_locs) / len(sim_list)), '%')
        self.nn_graph = nn_graph
        return

    def runClustering(self, alg, quality_function, cluster_kwargs={}, min_cluster_size=10):
        """Run a graph partitioning algorithmn on an NN-graph

        :param alg: paritioning algorithmn to use. one of ['louvain', 'leiden']
        :type alg: str
        :param quality_function: Quality function for determining graph modularity. 
        :type quality_function: str
        :param cluster_kwargs: additional parameters to use for clustering, defaults to {}
        :type cluster_kwargs: dict, optional
        :param min_cluster_size: Minimum size for cluster. clusters smaller than this will be merged into next closest cluster, defaults to 10
        :type min_cluster_size: int, optional
        :raises NotImplementedError: [description]
        :return: pandas DataFrame with observation IDs and clsuter assigments
        :rtype: pandas.DataFrame
        """
        self.m.message(
            f'Running {alg} clustering using {quality_function} quality function', 'DEBUG')

        if alg == 'louvain':
            cm = importlib.import_module('louvain')
        elif alg == 'leiden':
            cm = importlib.import_module('leidenalg')
        else:
            print('Bad Alg')
            raise NotImplementedError

        Q = getattr(cm, quality_function)
        self.nn_graph.simplify(combine_edges='sum')
        n_elements = self.data.shape[0]
        partition = cm.find_partition(
            graph = self.nn_graph,
            partition_type=Q,
            weights='weight',
            **cluster_kwargs
        )
        labels = np.asarray(partition.membership)
        if min_cluster_size > 1:
            labels = self.mergeSingletons(labels, min_cluster_size)

        return pd.DataFrame().assign(Barcode=list(self.data.index), labels=labels, knn=self.target_knn).sort_values('labels')
     
    def mergeSingletons(self, labels, small_pop ):
        """Merge smaller clusters into larger ones 

        :param labels: list of labels to clean
        :type labels: list
        :param small_pop: smallest size for clusters 
        :type small_pop: int
        :return: list of corrected labels 
        :rtype: [type]
        """        
        self.m.message('Merging small clustering', 'DEBUG')
        n_elements = self.data.shape[0]
        
        labels = np.reshape(labels, ( n_elements, 1))
        dummy, labels = np.unique(
            list(labels.flatten()), return_inverse=True)
        small_pop_exist = True
        time_smallpop_start = time.time()
        while (small_pop_exist == True) & ((time.time() - time_smallpop_start) < self.time_smallpop):
            small_pop_list = []
            small_cluster_list = []
            small_pop_exist = False
            
            for cluster in set(list(labels.flatten())):
                population = len(np.where(labels == cluster)[0])
                if population < small_pop:
                    small_pop_exist = True
                    small_pop_list.append(list(np.where(labels == cluster)[0]))
                    small_cluster_list.append(cluster)
            labels = reassign_small_cluster_cells(labels, small_pop_list, small_cluster_list, self.nn_neighbor_array)
        return labels 

    def doPerturbation(self, edge_permut_frac=None, weight_permut_range=None):
        """Run Perturbation experiments descibed in ... Chosen fraction of edges are removed, then same number of edges are randomly added between vertices. Each edge is then multiplied by a noise factor sampled from a uniform distribution of choseen range

        :param edge_permut_frac: fraction of edges to randomly remove and add
        :type edge_permut_frac: float
        :param weight_permut_range: range to uniformly sample from to permute edge weights
        :type weight_permut_range: [type]
        """ 
        if (edge_permut_frac is None) and (weight_permut_range is None):    
            self.m.message('Perturbation run with no parameters', 'INFO')
            raise NotImplementedError
        self.m.message('Running Perturbation', 'DEBUG')
        graph = self.nn_graph
        if edge_permut_frac is not None:
            self.m.message('Add/Remove edges', 'DEBUG')
            old_weights = np.asarray(graph.es['weight'])
            sub_sample_size = int(len(graph.es) * edge_permut_frac)
            edges_to_remove = np.random.choice(
                list(range(len(graph.es))), size=sub_sample_size, replace=False)

            i = list(range(len(graph.vs)))
            edges_to_add = np.random.choice(i, size=(sub_sample_size, 2), replace=True)
            graph.delete_edges(edges_to_remove)
            graph.add_edges(edges_to_add)
            if weight_permut_range is None:
                old_weights_permuted = np.random.choice(old_weights, len(old_weights))
                new_weights =  np.where(old_weights != None, old_weights, old_weights_permuted)
                # where old weights is not none, keep values from old weights, or when it is none, use new weights
                
                graph.es['weight'] = new_weights


        ### add noise to edge weights
        if weight_permut_range is not None:
            self.m.message('Add weight noise', 'DEBUG')
            old_weights = np.asarray(graph.es['weight'])
            # where old_weights is none, keep 1, and where its not none return old weights 
            new_weights = np.where(old_weights == None, 1, old_weights) * \
                np.random.uniform(
                    weight_permut_range[0], weight_permut_range[1], len(old_weights))
            graph.es['weight'] = new_weights
        self.nn_graph = graph
        return
   
    def runUMAP(self, n_components=2, alpha: float = 1.0, negative_sample_rate: int = 5,
                     gamma: float = 1.0, spread=1.0, min_dist=0.1, init_pos='spectral', random_state=1,):
        """ Perform UMAP dimensionality reduction from constructed NN graph 

        :param n_components: number of dimensions to reduce, defaults to 2
        :type n_components: int, optional
        :param alpha: [description], defaults to 1.0
        :type alpha: float, optional
        :param negative_sample_rate: [description], defaults to 5
        :type negative_sample_rate: int, optional
        :param gamma: [description], defaults to 1.0
        :type gamma: float, optional
        :param spread: [description], defaults to 1.0
        :type spread: float, optional
        :param min_dist: [description], defaults to 0.1
        :type min_dist: float, optional
        :param init_pos: [description], defaults to 'spectral'
        :type init_pos: str, optional
        :param random_state: [description], defaults to 1
        :type random_state: int, optional
        :return: [description]
        :rtype: [type]
        """        
        ## pre-process data for umap
        n_neighbors = self.nn_neighbor_array.shape[1]
        n_cells = self.nn_neighbor_array.shape[0]
        row_list = np.transpose(
            np.ones((n_neighbors, n_cells)) * range(0, n_cells)).flatten()
        row_min = np.min(self.nn_distance_array, axis=1)
        row_sigma = np.std(self.nn_distance_array, axis=1)

        distance_array = (
            self.nn_distance_array - row_min[:, np.newaxis])/row_sigma[:, np.newaxis]
        col_list = self.nn_neighbor_array.flatten().tolist()
        distance_array = distance_array.flatten()
        distance_array = np.sqrt(distance_array)
        distance_array = distance_array * -1

        weight_list = np.exp(distance_array)

        threshold = np.mean(weight_list) + 2 * np.std(weight_list)

        weight_list[weight_list >= threshold] = threshold

        weight_list = weight_list.tolist()

        graph = csr_matrix((np.array(weight_list), (np.array(row_list), np.array(col_list))),
                           shape=(n_cells, n_cells))

        graph_transpose = graph.T
        prod_matrix = graph.multiply(graph_transpose)

        graph = graph_transpose + graph - prod_matrix
        from umap.umap_ import find_ab_params, simplicial_set_embedding
        import matplotlib.pyplot as plt

        a, b = find_ab_params(spread, min_dist)
        X_umap = simplicial_set_embedding(data=self.data, graph=graph, n_components=n_components, initial_alpha=alpha, a=a, b=b, n_epochs=0, metric_kwds={
        }, gamma=gamma, negative_sample_rate=negative_sample_rate, init=init_pos,  random_state=np.random.RandomState(random_state), metric='euclidean', verbose=False, densmap=False, densmap_kwds={}, output_dens=False)
        return X_umap
        #return 

    def runPerturbations(self, alg, quality_function, n_perturbations,cluster_kwargs={},  edge_permut_frac=None, weight_permut_range=None, min_cluster_size=10, verbosity=1):
        """run multiple perturbation experiments on NN graph 

        :param alg: algorithmn to re-cluster with 
        :type alg: str
        :param quality_function: quality function to use for clustering 
        :type quality_function: str
        :param n_perturbations: Number of perturbation experiments to run
        :type n_perturbations: int
        :param cluster_kwargs: additional parameters for clustering algorithmn, defaults to {}
        :type cluster_kwargs: dict, optional
        :param edge_permut_frac: fraction of edges to randomly remove and add
        :type edge_permut_frac: float
        :param weight_permut_range: range to uniformly sample from to permute edge weights
        :type weight_permut_range: [type]
        :param small_pop: smallest size for clusters 
        :type small_pop: int
        :param verbosity: how much to print, defaults to 1
        :type verbosity: int, optional
        :return: [description]
        :rtype: [type]
        """        
        
        out_labels = [None] * n_perturbations
        for i in range(n_perturbations):
            perturbed_clu = copy.deepcopy(self)
            perturbed_clu.doPerturbation(edge_permut_frac, weight_permut_range)
            ptb_labels = perturbed_clu.runClustering(alg,quality_function, cluster_kwargs,min_cluster_size)
            out_labels[i] = ptb_labels

        return out_labels
    
    
def cluster_and_perturb_knn_range(reduction,njobs=1, alg='louvain', min_knn = 5, max_knn=100, knn_step=5, global_pruning_jac_threshold=None, local_pruning_dist_threshold=None, quality_function='RBConfigurationVertexPartition', cluster_kwargs={}, n_perturbations=100, edge_permut_frac=.05, weight_permut_range=None,  min_cluster_size=10, experiment_name='clusterEval', verbosity=0):
    
    if experiment_name == 'auto': 
        experiment_name = f'knn-{k}_alg-{alg}_localPruningDist-{local_pruning_dist_threshold}_globalPruningJac-{global_pruning_jac_threshold}_nPerturbations-{n_perturbations}_edgePermutFrac-{edge_permut_frac}_weightPermuteRange-{weight_permut_range}_minCluster=Size-{min_cluster_size}'

    clu_obj = ClusterExperiment(data=reduction, 
                                verbosity=verbosity, 
                                )
    clu_obj.buildNeighborArray(max_knn=max_knn, 
                               nn_space='l2',
                               nthreads = njobs
    )
    
    
    
    res = Parallel(n_jobs=njobs)(delayed(wrap_cluster)(k,clu_obj,global_pruning_jac_threshold, local_pruning_dist_threshold, alg, quality_function, cluster_kwargs, n_perturbations, edge_permut_frac, weight_permut_range,  min_cluster_size, experiment_name, verbosity) for k in range(min_knn, max_knn,knn_step))
    
    all_label_df =pd.concat([i[0] for i in res]).pivot(index = 'Barcode', columns = 'knn', values = 'labels')
    all_label_df.columns = [f'knn_{str(i)}' for i in all_label_df.columns]
    all_label_df = all_label_df.reset_index(drop = False)
    all_metric_df = pd.concat(i[1] for i in res)
    return ((all_label_df, all_metric_df))


    
def wrap_cluster(k,clu_obj, global_pruning_jac_threshold, local_pruning_dist_threshold, alg, quality_function, cluster_kwargs, n_perturbations, edge_permut_frac, weight_permut_range,  min_cluster_size, experiment_name, verbosity):
    clu_obj.neighborArray2Graph(k, global_pruning_jac_threshold, local_pruning_dist_threshold)
    labels = clu_obj.runClustering(alg=alg, 
                                quality_function=quality_function, 
                                cluster_kwargs =  cluster_kwargs,
                                min_cluster_size = min_cluster_size)
    perturbations = clu_obj.runPerturbations(alg = alg,
                                                quality_function='RBConfigurationVertexPartition',
                                                n_perturbations =n_perturbations, 
                                                edge_permut_frac=edge_permut_frac, 
                                                weight_permut_range=weight_permut_range, 
                                                min_cluster_size=min_cluster_size, 
                                                verbosity=verbosity)
    metrics = calculate_metrics(labels, perturbations,experiment_name, k)
    return((labels, metrics))


