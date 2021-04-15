import numpy as np
import pandas as pd
import hnswlib
from scipy.sparse import csr_matrix
import igraph as ig
import leidenalg
import time
import os 
import sys
import louvain


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
    :type client: numeric Pandas DataFrame
    :param knn: The number of nearest neighbors to use for NN graph creation
    :type knn: int
    :param nthreads: number of threads to use for NN graph creation
    :type nthreads: int
    """
    def __init__(self, data, verbosity):
        self.data = data
        self.time_smallpop = 15
        self.step=0
        self.m=SimpleMessager('ClusterExperiment', verbosity)

    

    def buildNeighborGraph(self, knn, nn_space, ef_construction, local_pruning, global_pruning, jac_std_global, dist_std_local, nthreads=1):
        """Build an approximate nearest neighbor graph from input data. Implements local and global edge pruning methods from PARC.

        :param nn_space: distance metric to use for graph creation. one of ['l2', 'ip', 'cosine']
        :type nn_space: str
        :param ef_construction: construction time/accuracy trade off. See https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
        :type ef_construction: int
        :param local_pruning: Run local edge pruning 
        :type local_pruning: bool
        :param global_pruning: run global edge pruning
        :type global_pruning: bool
        :param jac_std_global: CHANGEME Not totally sure tbh, need to look into it more 
        :type jac_std_global: str
        """        
        self.m.message('Building NN graph', 'DEBUG')
        ef_query = max(100, knn + 1)
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
        neighbor_array, distance_array = p.knn_query(self.data, knn)
        n_neighbors = neighbor_array.shape[1]
        n_cells = neighbor_array.shape[0]
        
        if np.sum(distance_array[:,1:] ==0) > 0:
            self.m.message('There are likely duplicate rows in the input data. remove and re-try', 'ERROR')
            raise DuplicateRowError

        if local_pruning:
            # for each cell, keep cells that are below dist_std_local standard deviations from mean 
            # for cells above threshold, set to 0. Then, when we select non-zero values from the sparse matrix
            # these pruned values will be dropped
            row_thresh = np.mean(distance_array, axis=1)  + dist_std_local * np.std(distance_array, axis = 1)
            keep_cells = distance_array < row_thresh[:,None]
            distance_array = np.where(keep_cells, distance_array, 0)
    
        row_list = np.transpose( np.ones((n_neighbors, n_cells)) * range(0, n_cells)).flatten()
        col_list = neighbor_array.flatten()
        weight_list = distance_array.flatten()# distance array will be 0 at loops
        csr_array = csr_matrix((weight_list, (row_list, col_list)),
                               shape=(n_cells, n_cells))
        self.nn_neighbor_array = neighbor_array
        self.nn_distance_array = distance_array
        sources, targets = csr_array.nonzero()

        edgelist = list(zip(sources, targets))
        nn_graph = ig.Graph(edgelist)
        edgelist_copy = edgelist.copy()
        edge_list_copy_array = np.asarray(edgelist_copy)
        sim_list = nn_graph.similarity_jaccard(pairs=edgelist_copy)
        sim_list_array = np.asarray(sim_list)

        if global_pruning:
            # remove edges below an edge weight threshold. Edges betwen poorly connected nodes wil have low edge weight 
            self.m.message('Running Global Edge Pruning', 'DEBUG')
            n_elements = self.data.shape[0]
            if jac_std_global == 'median':
                threshold = np.median(sim_list)
            else:
                threshold = np.mean(sim_list) - jac_std_global * np.std(sim_list)
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


    def run_perturbation(self, edge_permut_frac=None, weight_permut_range=None):
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


    def run_leiden(self, vertex_partition_method, n_iter, resolution, jac_weighted_edges=None, seed=None):
        """Run leiden clustering data on built nn graph

        :param vertex_partition_method: Quality function to optimize to find partitions. See leiden docs for more info 
        :type vertex_partition_method: leidenalg.VertexPartition.MutableVertexPartition
        :param n_iter: Number of iterations to run the Leiden algorithm. If the number of iterations is negative, the Leiden algorithm is run until an iteration in which there was no improvement.
        :type n_iter: int
        :param resolution: Cluster resolution that roughly correlates with number of clusters
        :type resolution: float
        :param jac_weighted_edges: list of weights or name of edge attribute within NN graph , defaults to None
        :type jac_weighted_edges: [type], optional
        :return: a 1D numpy array with lenght equal to nrows of input matrix 
        :rtype: numpy.ndarray
        """        
        self.m.message('Running Leiden clustering', 'DEBUG')
        self.nn_graph.simplify(combine_edges='sum')
        n_elements = self.data.shape[0]
        partition = leidenalg.find_partition(self.nn_graph, vertex_partition_method,
                                             weights=jac_weighted_edges, n_iterations=n_iter, 
                                             resolution_parameter = resolution, seed=seed)
        labels = np.asarray(partition.membership)
        return labels
    def run_louvain(self, vertex_partition_method, resolution,  jac_weighted_edges=None, seed=None):
        """Run louvain clustering data on built nn graph

        :param vertex_partition_method: Quality function to optimize to find partitions. See leiden docs for more info 
        :type vertex_partition_method: louvainalg.VertexPartition.MutableVertexPartition
        :param resolution: Cluster resolution that roughly correlates with number of clusters
        :type resolution: float
        :param jac_weighted_edges: list of weights or name of edge attribute within NN graph , defaults to None
        :type jac_weighted_edges: [type], optional
        :return: a 1D numpy array with lenght equal to nrows of input matrix 
        :rtype: numpy.ndarray
        """
        self.m.message('Running Louvain clustering', 'DEBUG')
        self.nn_graph.simplify(combine_edges='sum')
        n_elements = self.data.shape[0]
        partition = louvain.find_partition(self.nn_graph, vertex_partition_method,
                                             weights=jac_weighted_edges, 
                                             resolution_parameter = resolution, seed=seed)
        labels = np.asarray(partition.membership)
        
        return labels

    def merge_singletons(self, labels, small_pop ):
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
    def run_UMAP(self, n_components=2, alpha: float = 1.0, negative_sample_rate: int = 5,
                     gamma: float = 1.0, spread=1.0, min_dist=0.1, init_pos='spectral', random_state=1,):
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

def run_clustering(reduction,alg,  res, k, perturb = False,edge_permut_frac=None, weight_permut_range=None, local_pruning=False, global_pruning=False, min_cluster_size=10, verbosity=1):
    """Run clustering based on input parameters from start to finish

    :param reduction: input data to cluster
    :type reduction: numeric pandas DataFram
    :param alg: clustering algorithmn,either ['louvain', 'leiden']
    :type alg: str
    :param res: cluster resolution (>0)
    :type res: float
    :param k: number of nearest neighbors to use for NN graph construction
    :type k: int
    :param perturb: run a perturbation experiment, defaults to False
    :type perturb: bool, optional
    :param local_pruning: run local pruning, defaults to False
    :type local_pruning: bool, optional
    :param global_pruning: run global pruning, defaults to False
    :type global_pruning: bool, optional
    :param min_cluster_size: clusters below this size are merged into closest cluster, defaults to 10
    :type min_cluster_size: int, optional
    :return: pandas DataFrame containing sample ids and cluster labels 
    :rtype: pandas.DataFrame
    """      
    
    clu_obj = ClusterExperiment(data=reduction, verbosity=verbosity)
    clu_obj.buildNeighborGraph(knn=k, nn_space='l2', ef_construction=150,
                               local_pruning=local_pruning, global_pruning=global_pruning, jac_std_global='median', dist_std_local = 3)
    if perturb:
        clu_obj.run_perturbation(edge_permut_frac, weight_permut_range)
    
    if alg == 'louvain':
        labels = clu_obj.run_louvain(
            vertex_partition_method=louvain.RBConfigurationVertexPartition,
            resolution=res,
            jac_weighted_edges='weight'
        )
    elif alg == 'leiden':
        labels = clu_obj.run_leiden(
            vertex_partition_method=leidenalg.RBConfigurationVertexPartition,
            n_iter=5,
            resolution=res,
            jac_weighted_edges='weight'
        )
    else:
        print('BAD ALG')
        raise NotImplementedError
    
    if min_cluster_size > 1:
        labels = clu_obj.merge_singletons(labels, min_cluster_size)
    outdf = pd.DataFrame(
        {"Barcode": list(reduction.index), 'labels': labels}).sort_values('labels')
    return outdf
