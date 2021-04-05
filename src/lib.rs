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
use pyo3::{wrap_pyfunction, wrap_pymodule};

#[derive(Debug)]
struct ClusterResults {
    barcodes:Vec<String>,
    labels: Vec<String> , 
    barcode_set:HashSet<String>,
    grouped_barcodes: HashMap<String, HashSet<String>>,
    h_tot: f64,
    exp_name:String
}

#[derive(Debug)]
struct ExperimentResults{
    exp_param :String,
    cluster_ids : Vec<String>,
    h_k_scores: Vec<f64>
}


impl ExperimentResults{
    fn pprint(&self){
        for i in 0..self.cluster_ids.len(){
            println!("{},{}",&self.cluster_ids[i], &self.h_k_scores[i])
        }
    }
}

fn entropy(group_map: &HashMap<String, HashSet<String>>, labels:&Vec<String> ) -> f64{
        let n = labels.len() as f64;
        let res: f64 = group_map.values().map(|i|{
            let p = i.len() as f64 /n;
            p * p.ln()
        }).sum();
        return res * -1 as f64
    }

impl ClusterResults{
    fn new(barcodes:Vec<String>, labels: Vec<String>, exp_name: String) -> ClusterResults{
        let barcode_set: HashSet<String> = HashSet::from_iter(barcodes.clone());
        let mut grouped_barcodes:HashMap<String, HashSet<String>> = HashMap::new();
        let mut old_label = &labels[0];
        let mut current_label = &labels[0];// declare out here so we can add the last set back in 
        let mut current_set: HashSet<String> = HashSet::new();
        for i in 0..barcodes.len(){
            current_label = &labels[i];
            let current_barcode = &barcodes[i];
            if  current_label == old_label{
                current_set.insert(current_barcode.clone());
            }else{// reach a new cluster 
                grouped_barcodes.insert(old_label.clone(), current_set);
                let ns: HashSet<String> = HashSet::new();
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

fn H_k(ref_bc: &HashSet<String>, query:&ClusterResults) -> f64{
        let intersect: HashSet<String> = ref_bc.intersection(&query.barcode_set).cloned().collect::<HashSet<String>>();
        if intersect.len() == 0{
            return 0.0
        } else{
            let mut new_bc :Vec<String> = vec![String::new(); intersect.len()];
            let mut new_labels : Vec<String> = vec![String::new(); intersect.len()];
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
    let mut barcodes: Vec<String> = vec![String::new(); file_string.len()];
    let mut labels: Vec<String> = vec![String::new(); file_string.len()];
    for i in 0..file_string.len(){
        let line_split : Vec<&str> = file_string[i].split(",").collect();
        barcodes[i] = String::from(line_split[0]);
        labels[i] = String::from(format!("clu{}", line_split[1]) );
    }
    ClusterResults::new(barcodes,labels, String::from(file))
}

fn calculate_stability(ref_cluster:&ClusterResults, query_clusters: &Vec<&ClusterResults>) -> ExperimentResults{
    let mut exp_result = Array2::<f64>::zeros(( ref_cluster.grouped_barcodes.len() ,query_clusters.len() ));
    for (i, cluster) in ref_cluster.grouped_barcodes.values().enumerate(){
        for (j,  experiment) in query_clusters.iter().enumerate() {
            exp_result[[i, j]]= H_k(&cluster, &experiment) / experiment.h_tot ; 
        }

    }
    let h_k_scores = exp_result.rows().into_iter().map(|x| 1.0 - x.mean().unwrap()).collect::<Vec<f64>>();
    let cluster_ids: Vec<String> = ref_cluster.grouped_barcodes.keys().cloned().collect::<Vec<String>>() ;
    let exp_param = ref_cluster.exp_name.clone();
    return ExperimentResults{ exp_param,cluster_ids, h_k_scores }
}

fn run_pairwise_calculation(experiment_list:&Vec<&ClusterResults>) ->Vec<ExperimentResults>{
    let mut res: Vec<ExperimentResults> = Vec::new();
    for i in 0..experiment_list.len(){
        let ref_clust = experiment_list[i];
        let mut query_clusts = experiment_list.clone();
        query_clusts.remove(i);
        res.push(calculate_stability(ref_clust,  &query_clusts, ) );
    }
    return res;

}


fn run_pairwise_calculation_threaded(experiment_list:&Vec<&ClusterResults>) ->Vec<ExperimentResults>{
    
    let pool = rayon::ThreadPoolBuilder::new().num_threads(16).build().unwrap();
    let dummy_array: Vec<usize> = (0..experiment_list.len()).collect();
    let res: Vec<ExperimentResults> = pool.install(|| dummy_array.into_par_iter()
                                         .map(|i:usize| { 
                                            let ref_clust = experiment_list[i];
                                            let mut query_clusts = experiment_list.clone();
                                            query_clusts.remove(i);
                                            return calculate_stability(ref_clust, &query_clusts)
                                            })
                                         .collect()
                                        );                                                      
    return res 
    

}

#[pyfunction]
fn run_stability_calculation(file_glob: &str) {
    let test_clusters_objs:Vec<ClusterResults> = glob(file_glob)
                                .expect("Failed to read glob pattern")
                                .map(|x|{let file =  String::from(x.unwrap().to_str().expect("asd"));
                                         return read_cluster_results(&file)}
                                        )
                                .collect();
    

    let test_cluster_refs: Vec<&ClusterResults> = test_clusters_objs.iter().collect();
    let c_res :Vec<f64> = run_pairwise_calculation_threaded(&test_cluster_refs)
                          .into_iter()
                          .map(|x| x.h_k_scores.iter().sum() )
                          .collect() ;
    println!("cRES{:?}", c_res);

}

fn stability_rust(module: &PyModule) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(run_stability_calculation, module)?)?;
    Ok(())
}

#[pymodule]
fn clustereval(py: Python, m: &PyModule) -> PyResult<()> {
    let submod = PyModule::new(py, "stability_rust")?;
    stability_rust(submod)?;
    m.add_submodule(submod)?;
    Ok(())
}

