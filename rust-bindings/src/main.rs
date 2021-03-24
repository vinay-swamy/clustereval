use std::fs ;
use std::collections::HashMap;
use std::collections::HashSet;
use std::iter::FromIterator;
use std::iter::Iterator;

#[derive(Debug)]

struct ClusterResults {
    barcodes:Vec<String>,
    labels: Vec<String> , 
    barcode_set:HashSet<String>
}


impl ClusterResults{
    fn new(barcodes:Vec<String>, labels: Vec<String>) -> ClusterResults{
        let barcode_set: HashSet<String> = HashSet::from_iter(barcodes.clone());
        ClusterResults{barcodes, labels, barcode_set }
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
    
    fn H_tot(&self, query:ClusterResults) -> f64{
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
    let mut barcodes = Vec::new();
    let mut labels = Vec::new();
    for i in 1..file_string.len(){
        let line_split : Vec<&str> = file_string[i].split(",").collect();
        barcodes.push(String::from(line_split[0]));
        labels.push(String::from(format!("clu{}", line_split[1]) ));
    }
    ClusterResults::new(barcodes,labels)
}
//read a data frame into a ha 


fn main() {
    let k = read_cluster_results("test.csv");
    let j = read_cluster_results("query.csv");
    println!("{}", k.H_tot(j))

}
