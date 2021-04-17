from ._calc_metrics import *
import pandas as pd
import numpy as np

class MissingInputError(Exception):
    pass
def calculate_metrics(ref_label, query_labels, experiment_name='clusterEval'):
    """calculate Stability and Purity for sets of custering label

    :param ref_label: A set of reference labels, for which metrics can be calculate. must be a pandas DataFrame, where column one is sample labels, and column2 is cluster labels. MUST BE SORTED BY SECOND COLUMN
    :type ref_label: List/vector OR pandas DataFrame
    :param query_labels: 2 or more sets of labels to compare ref_label againt. Either List of lists, or list of DataFrames(MUST MATCH ref_label FORMAT)
    :type query_labels: List of Lists or List of Pandas DataFrames
    :param experiment_name: name to assign current experiment
    :type experiment_name: str
    """ 
    if type(ref_label) == type(pd.DataFrame()):
        if ref_label.shape[1] > 2:
            print('DF longer than 2: extra columns will be ignored')

        ref_label = ref_label.iloc[:, :2]
        old_label_col = ref_label.columns[1]
        ref_label.columns = ['Barcode', 'labels']
        query_labels = [df.rename(
            columns={df.columns[0]:'Barcode', df.columns[1]:'labels'}) for df in query_labels]

        label_converted = ref_label['labels'].dtype != 'int'
        if ref_label['Barcode'].dtype != 'int':
            id_conv_df = pd.DataFrame().assign(Barcode=ref_label['Barcode'],
                                               new_bc=list(range(ref_label.shape[0])))
            ref_label['Barcode'] = id_conv_df['new_bc']
            query_labels = [df.merge(id_conv_df, how='inner').drop(
                columns=['Barcode']).rename(columns={'new_bc': 'Barcode'})[['Barcode', 'labels']]
                for df in query_labels
            ]
        if label_converted:
            converter_df = pd.DataFrame(
                {'labels': ref_label['labels'].unique()})
            converter_df['new_lab'] = list(range(converter_df.shape[0]))
            ref_label = ref_label.merge(converter_df, how='inner', on='labels').drop(
                columns=['labels']).rename(columns={'new_lab': 'labels'})[['Barcode', 'labels']].to_dict('list')

            ## query labels are independent from ref labels, so just convert to numeric
            query_labels = [df.assign(labels=pd.factorize(df['labels'])[0]).to_dict('list')
                            for df in query_labels]
        else:
            ref_label = ref_label.to_dict('list')
            query_labels = [df.to_dict('list')
                            for df in query_labels]

        
        er_class =  metric_calculation_fromdf(ref_label, query_labels, experiment_name )
        exp_results = pd.DataFrame().assign(cluster_ids = er_class.cluster_ids, 
                                            stability = er_class.stability_scores,
                                            purity = er_class.purity_scores,
                                            exp_name = er_class.exp_param)
        if label_converted:

            exp_results =  exp_results.rename(columns = {'cluster_ids':'new_lab'}).merge(converter_df, how = 'inner').drop(columns = ['new_lab']).rename({'labels':old_label_col})
            
        else:
            exp_results = exp_results.rename({'cluster_ids': old_label_col})
        return exp_results

    else:
        raise MissingInputError
        ## list mode
        # if type(ref_label) == type([]):
        #     ref_label = np.asarray(ref_label)
        #     query_labels = np.asarray(query_labels)
        # is_int = np.issubdtype(ref_label.dtype, np.integer)
        # sample_ids = list(range(len(ref_label)))
        # if is_int:
        #     print('issy in')
        #     ref_label = {'Barcode': sample_ids, 'labels': ref_label}
        #     pd.DataFrame(ref_label).to_csv('debug_ref.csv')
        #     query_labels = [{'Barcode': sample_ids, 'labels': ql}  for ql in query_labels ]
        # else:
        #     converter_tab = pd.DataFrame().assign(label =ref_label, cluster_ids = pd.factorize(ref_label)[0])
        #     print(converter_tab)
        #     ref_label = {'Barcode': sample_ids, 'labels': converter_tab['cluster_ids'].to_list()} 
        #     query_labels = [{'Barcode': sample_ids,
        #                      'labels': pd.factorize(ql)[0]} for ql in query_labels]
        # #print(ref_label) 
        # er_class = metric_calculation_fromdf(
        #         ref_label, query_labels, experiment_name)
        
        # exp_results = pd.DataFrame().assign(cluster_ids=er_class.cluster_ids,
        #                                     stability=er_class.stability_scores,
        #                                     purity=er_class.purity_scores,
        #                                     exp_name=er_class.exp_param)

        # if is_int:
        #     return exp_results
        # else:
        #     return exp_results.merge(converter_tab.drop_duplicates()).drop(columns = 'cluster_ids')

        
def pairwise_metric_calculation(input,names=None, nthreads=1):
    if type(input) == type(''):
        return pairwise_metric_calculation_fromdisk(input, nthreads)
    else:
        if names is None:
            raise MissingInputError
        else:
            return pairwise_metric_calculation_frommem(input, names, nthreads)
