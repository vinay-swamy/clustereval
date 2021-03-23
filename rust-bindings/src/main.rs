use std::fs ;

struct ClusterResults {
    barcodes:Vec<String>,
    labels: Vec<String>  
}

impl ClusterResults{
    fn new(barcodes:Vec<String>, labels: Vec<String>) -> ClusterResults{
        ClusterResults{barcodes, labels }
    }
}

fn read_cluster_results( file: &str) ->ClusterResults {
    let file_string = fs::read_to_string(file)?;
    let file_string = file_string.split("\n").collect()
    barcodes = Vec::with_capacity(file_string.len())
    for line in file_string{
        line = line.split(",").collect()
    }
}
//read a data frame into a ha 



fn main() {
    println!("Hello, world!");
}
