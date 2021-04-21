use std::fs ;
use std::collections::HashMap;
use std::collections::HashSet;
use std::iter::FromIterator;
use std::iter::Iterator;
use ndarray::Array2;
use rayon;
use rayon::prelude::*;
use flate2::read::GzDecoder;
use std::io::prelude::*;
use glob::glob;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[derive(Debug)]
struct ClusterResults {
    barcodes:Vec<i64>,
    labels: Vec<i64> , 
    barcode_set:HashSet<i64>,
    grouped_barcodes: HashMap<i64, HashSet<i64>>,
    h_tot: f64,
    exp_name:String
}

#[pyclass]
struct ExperimentResults{
    #[pyo3(get)]
    exp_param :String,
    #[pyo3(get)]
    cluster_ids : Vec<i64>,
    #[pyo3(get)]
    stability_scores: Vec<f64>,
    #[pyo3(get)]
    purity_scores:Vec<f64>
}


impl ExperimentResults{
    fn pprint(&self){
        for i in 0..self.cluster_ids.len(){
            println!("{},{},{}",&self.cluster_ids[i], &self.stability_scores[i],&self.purity_scores[i] )
        }
    }
    fn write_csv(&self, outpath:&str)->std::io::Result<()>{

        let mut lines: Vec<String> = vec![String::new();self.cluster_ids.len()];
        for i in 0..self.cluster_ids.len(){
            lines[i] = format!("{},{},{}\n",self.cluster_ids[i], self.stability_scores[i],self.purity_scores[i])
        }
        let outfile = format!("{}/{}", outpath, self.exp_param);
        let outstring = lines.join("");
        fs::write(outfile, outstring).unwrap();
        Ok(())
    }
    fn write_csv_simple(&self, outfile:&str)->std::io::Result<()>{

        let mut lines: Vec<String> = vec![String::new();self.cluster_ids.len()];
        for i in 0..self.cluster_ids.len(){
            lines[i] = format!("{},{},{}\n",self.cluster_ids[i], self.stability_scores[i],self.purity_scores[i])
        }
        let outstring = lines.join("");
        fs::write(outfile, outstring).unwrap();
        Ok(())

    }
}

fn entropy(group_map: &HashMap<i64, HashSet<i64>>, labels:&Vec<i64> ) -> f64{
        let n = labels.len() as f64;
        let res: f64 = group_map.values().map(|i|{
            let p = i.len() as f64 /n;
            p * p.ln()
        }).sum();
        return res * -1 as f64
    }

impl ClusterResults{
    fn new(barcodes:Vec<i64>, labels: Vec<i64>, exp_name: String) -> ClusterResults{
        let barcode_set: HashSet<i64> = HashSet::from_iter(barcodes.clone());
        let mut grouped_barcodes:HashMap<i64, HashSet<i64>> = HashMap::new();
        let mut old_label = &labels[0];
        let mut current_label = &labels[0];// declare out here so we can add the last set back in 
        let mut current_set: HashSet<i64> = HashSet::new();
        for i in 0..barcodes.len(){
            current_label = &labels[i];
            let current_barcode = &barcodes[i];
            if  current_label == old_label{
                current_set.insert(current_barcode.clone());
            }else{// reach a new cluster 
                let dup_check = grouped_barcodes.insert(old_label.clone(), current_set);
                if !dup_check.is_none(){ // HashMap.insert returns None when new key is added
                    panic!("A duplicate key was added when making a ClusterResults; input data is not sorted by label")
                }
                let ns: HashSet<i64> = HashSet::new();
                current_set = ns;
                current_set.insert(current_barcode.clone());
                old_label = current_label;
            }
        }
        grouped_barcodes.insert(current_label.clone(), current_set);
        let h_tot = entropy(&grouped_barcodes, &labels);
        ClusterResults{barcodes, labels, barcode_set, grouped_barcodes, h_tot, exp_name}
    }
    fn head(&self){
        println!("{:?}", &self.barcodes[0..5]);
        println!("{:?}", &self.labels[0..5])
    }
    
}

fn stability_k(ref_bc: &HashSet<i64>, query:&ClusterResults) -> f64{
        let intersect: HashSet<i64> = ref_bc.intersection(&query.barcode_set).cloned().collect::<HashSet<i64>>();
        if intersect.len() == 0{
            return 0.0
        } else{
            let mut new_bc :Vec<i64> = vec![-1; intersect.len()];
            let mut new_labels : Vec<i64> = vec![-1; intersect.len()];
            let mut j=0;
            for i in 0..query.barcodes.len(){
                if intersect.contains(&query.barcodes[i]){
                    new_bc[j] = query.barcodes[i].clone();
                    new_labels[j] = query.labels[i].clone();
                    j+=1;
                }
            }
            let new_clu = ClusterResults::new(new_bc, new_labels, String::new());//use an empty string for these guys, as they get deleted later 
            return entropy(&new_clu.grouped_barcodes, &new_clu.labels);
        }
    }
fn decode_reader(bytes: Vec<u8>) -> std::io::Result<String> {
   let mut gz = GzDecoder::new(&bytes[..]);
   let mut s = String::new();
   gz.read_to_string(&mut s)?;
   Ok(s)
}

fn read_cluster_results( file: &str) ->ClusterResults {
    let mut handle = fs::File::open(file).expect("Bad file input");
    let mut buffer  = Vec::new();
    handle.read_to_end(&mut buffer).expect("couldnt read file");
    let file_string = decode_reader(buffer).expect("bad gzip");
    let file_string: Vec<&str> = file_string.lines().collect();
    let mut barcodes: Vec<i64> = vec![-1; file_string.len()];
    let mut labels: Vec<i64> = vec![-1; file_string.len()];
    for i in 0..file_string.len(){
        let line_split : Vec<&str> = file_string[i].split(",").collect();
        barcodes[i] = String::from(line_split[0]).parse::<i64>().unwrap();
        labels[i] = String::from(line_split[1]).parse::<i64>().unwrap();
    }
    let exp_name  = file.split("/").last().unwrap() ;
    ClusterResults::new(barcodes,labels, String::from(exp_name))
}

fn calculate_metrics(ref_cluster:&ClusterResults, query_clusters: &Vec<&ClusterResults>) -> ExperimentResults{
    let mut stability_results = Array2::<f64>::zeros(( ref_cluster.grouped_barcodes.len() ,query_clusters.len() ));
    let mut purity_results = Array2::<f64>::zeros(( ref_cluster.grouped_barcodes.len() ,query_clusters.len() ));
    for (i, cluster) in ref_cluster.grouped_barcodes.values().enumerate(){
        for (j,  experiment) in query_clusters.iter().enumerate() {
            let mut stab = stability_k(&cluster, &experiment) / experiment.h_tot ; 
            if stab.is_nan(){// cant compare a naturally occuring NAN to f64::NAN
                stab = 1.0;
            }
            stability_results[[i, j]]= stab ; 
            purity_results[[i,j]] = purity_k(&cluster, &experiment.grouped_barcodes)
        }

    }
    let stability_scores = stability_results.rows().into_iter().map(|x| 1.0 - x.mean().unwrap()).collect::<Vec<f64>>();
    let purity_scores = purity_results.rows().into_iter().map( |x| {
        let mut v = x.to_vec();
        v.retain(|x| *x != f64::NAN); // in purity_k f64::NAN is explicitly returned, so this works. Consider changing for conistency
        return vmean(v) 
    } ).collect::<Vec<f64>>();   
    let cluster_ids: Vec<i64> = ref_cluster.grouped_barcodes.keys().cloned().collect::<Vec<i64>>() ;
    let exp_param = ref_cluster.exp_name.clone();
    return ExperimentResults{ exp_param,cluster_ids, stability_scores, purity_scores }
}

fn vmean(v:Vec<f64>) -> f64{
 return v.iter().sum::<f64>() / v.len() as f64

}

fn purity_k(ref_bc_set: &HashSet<i64>, query_map: &HashMap<i64, HashSet<i64>>) -> f64{
    let mut max_overlap = 0;
    let mut max_overlap_key:i64 = -100000000;
    for query_key in query_map.keys(){
        let q_cluster_set = query_map.get(query_key).unwrap();
        let overlap = ref_bc_set.intersection(q_cluster_set).count();
        if overlap > max_overlap{
            max_overlap = overlap;
            max_overlap_key = *query_key;
        }
    }
    if max_overlap_key == -100000000{
        return f64::NAN;
    } else{
        return max_overlap as f64 / query_map.get(&max_overlap_key).unwrap().len() as f64
    }
}

fn run_pairwise_calculation_threaded(experiment_list:&Vec<&ClusterResults>, nthreads:usize) ->Vec<ExperimentResults>{
    
    let pool = rayon::ThreadPoolBuilder::new().num_threads(nthreads).build().unwrap();
    let dummy_array: Vec<usize> = (0..experiment_list.len()).collect();
    let res: Vec<ExperimentResults> = pool.install(|| dummy_array.into_par_iter()
                                         .map(|i:usize| { 
                                            let ref_clust = experiment_list[i];
                                            let mut query_clusts = experiment_list.clone();
                                            query_clusts.remove(i);
                                            return calculate_metrics(ref_clust, &query_clusts)
                                            })
                                         .collect()
                                        );                                                      
    return res 
    

}

#[pyfunction]
fn pairwise_metric_calculation_fromdisk(file_glob: &str, nthreads:usize) -> Vec<ExperimentResults> {
    let test_clusters_objs:Vec<ClusterResults> = glob(file_glob)
                                .expect("Failed to read glob pattern")
                                .map(|x|{let file =  String::from(x.unwrap().to_str().expect("Failed to unwrap filename"));
                                         return read_cluster_results(&file)}
                                        )
                                .collect();
    if test_clusters_objs.len() == 0{
        panic!("The provided glob string did not match any files!!")
    }
    
    let test_cluster_refs: Vec<&ClusterResults> = test_clusters_objs.iter().collect();
    let c_res :Vec<ExperimentResults> = run_pairwise_calculation_threaded(&test_cluster_refs, nthreads);
    return c_res
}

#[pyfunction]
fn pairwise_metric_calculation_frommem(mut cluster_dfs: Vec<HashMap<String, Vec<i64>>>, exp_names:Vec<String>, nthreads:usize) -> Vec<ExperimentResults> {
    let clusters_objs_owned = cluster_dfs.into_iter().enumerate().map(|(i, mut x)|{
        ClusterResults::new(x.remove(&String::from("Barcode")).unwrap(), 
                            x.remove(&String::from("labels")).unwrap(), 
                            exp_names[i].clone() )}).collect::<Vec<ClusterResults>>();
                        
    

    let clusters_objs_refs: Vec<&ClusterResults> = clusters_objs_owned.iter().collect();
    let c_res :Vec<ExperimentResults> = run_pairwise_calculation_threaded(&clusters_objs_refs, nthreads);
    return c_res
}

#[pyfunction]
fn metric_calculation_fromdf(mut ref_df: HashMap<String, Vec<i64>>, query_dfs:Vec<HashMap<String, Vec<i64>>>, exp_name: String)->ExperimentResults{
    let ref_cluster = ClusterResults::new(ref_df.remove(&String::from("Barcode")).unwrap(), 
                                          ref_df.remove(&String::from("labels")).unwrap(), 
                                          exp_name);
    let query_clusters_owned = query_dfs.into_iter().map(|mut x|ClusterResults::new(x.remove(&String::from("Barcode")).unwrap(), 
                                                                                x.remove(&String::from("labels")).unwrap(), 
                                                                                String::from("perturbation") )
                                                        ).collect::<Vec<ClusterResults>>();
    let query_clusters_refs = query_clusters_owned.iter().collect::<Vec<&ClusterResults>>();

    let res = calculate_metrics(&ref_cluster, &query_clusters_refs);
    return res

}

// fn calc_metrics(module: &PyModule) -> PyResult<()> {
//     module.add_function(wrap_pyfunction!(pairwise_metric_calculation_fromdisk, module)?)?;
//     module.add_function(wrap_pyfunction!(pairwise_metric_calculation_frommem, module)?)?;
//     module.add_function(wrap_pyfunction!(oneway_metric_calculation, module)?)?;
//     module.add_class::<ExperimentResults>()?;
//     Ok(())
// }

#[pymodule]
fn _calc_metrics(py: Python, module: &PyModule) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(pairwise_metric_calculation_fromdisk, module)?)?;
    module.add_function(wrap_pyfunction!(pairwise_metric_calculation_frommem, module)?)?;
    module.add_function(wrap_pyfunction!(metric_calculation_fromdf, module)?)?;
    module.add_class::<ExperimentResults>()?;
    Ok(())
}

#[test]
fn check_reader(){
    let obj = read_cluster_results("test_data/exp-0_resolution-0.4_knn-15_.csv.gz");
    assert_eq!(obj.barcodes.len(), obj.labels.len());
}
