import pandas as pd 
import numpy as np 
import operator 

'''
Strategy 1 - double recluster:
    1. Cluster on knn range
    2. pick pairs of high and low knn that have the same number of clusters with high metrics( ie align clusters)
    3. remove barcodes common to both low and high knn labels
    4. Re-cluster
Strategy 2 - single recluster
    1. Cluster on knn rangg
    2. for each clustering, remove clusters that meet metric thresholds 
    3. re-cluster
'''

## helpers 
def max_frac(x):
    values, counts = np.unique(x, return_counts = True)
    return np.max(counts)/ len(x)

def which_max_frac(x):
    values, counts = np.unique(x, return_counts = True)
    return values[np.argmax(counts)] 

def nunique(i):
    return(len(np.unique(i)))

def strategy_1_recluster(all_permute_metrics, all_labels, stability_threshold=.8, purity_threshold=.8,max_sets_to_match=3,min_pass_cluster_delta = 2, min_knn_delta  = 20, max_results_to_eval=3 ):
    all_permute_metrics = all_permute_metrics.assign(nCluPass_stability = lambda x: x['stability'] > stability_threshold, 
                                                     nCluPass_purity = lambda x: x['purity'] > purity_threshold ).assign(nCluPass_both = lambda x: x['nCluPass_stability'] & x['nCluPass_purity'] )

    metric_summary = (all_permute_metrics
                    .groupby('knn')
                    .agg(nclusters = ('cluster_ids',nunique ), 
                        n_pass_metrics = ('nCluPass_both', sum),
                        avg_permute_stability = ('stability', np.mean),
                        avg_permute_purity = ('purity', np.mean))
                    .assign(avg_score = lambda x: (x['avg_permute_stability'] + x['avg_permute_purity']) / 2)
                    .reset_index(drop=False)
                    .sort_values('avg_score', ascending =False)
                    )
    target_sets =  metric_summary.drop_duplicates('n_pass_metrics').iloc[:max_sets_to_match,:].reset_index(drop=False)
    match_results= []
    for index, row in target_sets.iterrows():
        n_clusters_pass = row['n_pass_metrics']
        n_clusters_total = row['nclusters'] 
        query_knn = int(row['knn'])
        c_set_passing_clusters = all_permute_metrics[all_permute_metrics['nCluPass_both']].query('knn == @query_knn')['cluster_ids'].sort_values().to_numpy()
        sets_match_passing_clusters=(metric_summary
        .query('n_pass_metrics == @n_clusters_pass')
        .assign(delta_pass_clusters = lambda x: abs(x['nclusters'] - n_clusters_total),
                delta_knn = lambda x: query_knn - x['knn'])
        .query('delta_pass_clusters  >= @min_pass_cluster_delta')
        .query('delta_knn >= @min_knn_delta')
        )
        for _, matched_set in sets_match_passing_clusters.iterrows():
            matched_set_knn = int(matched_set['knn'])
            major_k = f'knn_{str(query_knn)}'
            minor_k = f'knn_{str(matched_set_knn)}'
            #major_k = query_knn
            #minor_k = matched_set_knn
            matched_sets_labels = all_labels[['Barcode', major_k, minor_k]]
            final_mapping = all_labels[[major_k, minor_k]].groupby(major_k).agg(max_frac = (minor_k, max_frac), which_max_frac = (minor_k, which_max_frac) ).reset_index()
            final_mapping = final_mapping[final_mapping[major_k].isin(c_set_passing_clusters)]
            avg_cluster_overlap = final_mapping.max_frac.mean()
            match_results.append((query_knn, matched_set_knn, avg_cluster_overlap, final_mapping) )
    match_results.sort(key = operator.itemgetter(2), reverse=True)
    print(f"Number of Matches: {str(len(match_results))}")
    
    results_to_recluster=[]
    for mapping in match_results[:max_results_to_eval]:
        query_knn = int(mapping[0])
        target_knn = int(mapping[1])
        major_k = f'knn_{str(query_knn)}'
        minor_k = f'knn_{str(target_knn)}'
        mapping_result = mapping[3]
        t_labels = all_labels[['Barcode', major_k, minor_k]]
        res=[]
        for i,j in zip(mapping_result[major_k].to_list(),  mapping_result.which_max_frac.to_list() ):
            res += t_labels[(t_labels[major_k] == i) & (t_labels[minor_k] == j )]['Barcode'].to_list()
        
        labels_to_recluster = t_labels[~t_labels.Barcode.isin(res)]
        results_to_recluster.append((query_knn, target_knn, mapping_result, labels_to_recluster ))
    return results_to_recluster