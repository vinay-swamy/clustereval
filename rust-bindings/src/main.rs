use std::fs ;
use std::collections::HashMap;
use std::collections::HashSet;
use std::iter::FromIterator;
use std::iter::Iterator;
use ndarray::{array,Array2, s};

#[derive(Debug)]

struct ClusterResults {
    barcodes:Vec<String>,
    labels: Vec<String> , 
    barcode_set:HashSet<String>,
    grouped_barcodes: HashMap<String, HashSet<String>>
}

// struct ExperimentResults{
//     exp_param :String,
//     h_k_scores: &[f64]
// }


impl ClusterResults{
    fn new(barcodes:Vec<String>, labels: Vec<String>) -> ClusterResults{
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
        ClusterResults{barcodes, labels, barcode_set, grouped_barcodes}
    }
    fn head(&self){
        println!("{:?}", &self.barcodes[0..5]);
        println!("{:?}", &self.labels[0..5])
    }
    fn entropy(&self) -> f64{
        let mut freq_table: HashMap<String, usize>  = HashMap::new();
        for label in self.labels.iter(){
            if let Some(x)  = freq_table.get_mut(label){
                *x = *x + 1  ;
            } else{
                freq_table.insert(label.clone() , 1);
            }
        }
        let n = self.labels.len() as f64;
        let res: f64 = freq_table.values().map(|i|{
            let p = *i as f64 /n;
            p * p.ln()
        }).sum();

        return res * -1 as f64
    }
    
    fn H_k(&self, query:&ClusterResults) -> f64{
        let intersect: HashSet<String> = self.barcode_set.intersection(&query.barcode_set).cloned().collect::<HashSet<String>>();
        println!("Helo{}" , intersect.len());
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
            return ClusterResults::new(new_bc, new_labels).entropy();
        }
    }
}



fn read_cluster_results( file: &str) ->ClusterResults {
    let file_string = fs::read_to_string(file).expect("Bad input file ");
    let file_string: Vec<&str> = file_string.lines().collect();
    let mut barcodes: Vec<String> = vec![String::new(); file_string.len()];
    let mut labels: Vec<String> = vec![String::new(); file_string.len()];
    for i in 0..file_string.len(){
        let line_split : Vec<&str> = file_string[i].split(",").collect();
        barcodes[i] = String::from(line_split[0]);
        labels[i] = String::from(format!("clu{}", line_split[1]) );
    }
    ClusterResults::new(barcodes,labels)
}

// fn run_calculation(ref_cluster:&ClusterResults, query_clusters: &Vec<ClusterResults>, h_tot_scores: &[f64], exp_name: String) -> ExperimentResults{

// }

fn main() {
    let k = read_cluster_results("test_sorted.csv");
    //let j = read_cluster_results("query.csv");
    let j:usize = k.grouped_barcodes.values().map(|x| x.len() ).sum();
    let mut x : Vec<usize> = k.grouped_barcodes.values().map(|x| x.len() ).collect();
    x.sort();
    let m = k.barcodes.len();    
    //let p = k.grouped_barcodes.get(&String::from("clu34"));
    println!("{:?}", j);
    println!("{:?}", m)
    //println!(xv)
    //println!("{:?}", k.)
    //println!("{}", m)
}
