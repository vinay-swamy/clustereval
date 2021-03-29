## Evaluate Clustering

Main goal: design a package for evaluating clustering accuracy / identifying optimal clusters, particularly for scRNA-seq

### Background

Motivation: Different clustering algs/ params produce different clusters, which ones are the best 

[Sanes and Regev](https://www.sciencedirect.com/science/article/pii/S0092867416310078?via%3Dihub#app2) define two cluster accuracy metrics Stability and Purity. These are the ones I started with. From some digging through literature, conceptually these metrics are somewhat well established, though the exact implementations vary.

### Roadmap/things to do

I've impleneted these metrics, and can use them to pick optimal clusters, and have [evaluated the behavior](https://github.com/davemcg/scEiaD/blob/subCellType/analysis/clusterEval_metrix_analysis.ipynb) of these metrics , buuuuuut
- implementation is ridculously ineffcient(we're talking ~30Hours x 32 cpus x 1000G ram bad for stability)
- need to tie clustering optimization to some sort of biological function
- (ideally) need to show theoretical origins of metrics 
- compare to alternative methods for cluster evaluation
- need to apply these metrics to multiple datasets, comment about louvain and leiden algs 

#### Computational Efficiency
I have standalone implementations of louvain and leiden, neighbor graph via hsnw, perturbations, and local/global pruning implemented in python(thanks to parc), and it's likely to stay in python.
To remedy metrics ineffiency, I'm, going to re-implement the metric portion in Rust, and provide python bindings to use them. I've got the Stability metric working as a stand alone binary
TODO:
- refactor code into a lib
- Make python bindings 
- implement purity in Rust
- implement multi-threading
- add tests and setup continuous integration
- distribute pre-compiled binaries via conda/pypi
- potentially comment on the utility of Rust 

#### Connecting clustering optimization to biological function

Does the optimiatuing clusters even matter within the context of biology?
- compare DiffExp/somthing else of worst vs best clusters

#### Theoretical origins of metrics

Luxburg et al is a whole [book](https://arxiv.org/pdf/1007.1075.pdf) on this topics, and provides a lot of good background on evaluating clustering stability. 
- define metrics and perturbations within the theorectical context described in Luxburg et al .
- for Stability, Variation of Information distance and Conditional mutual information might be a good place to start.

#### Comparison to other methods for cluster evaluation

- [Clustree](https://github.com/lazappi/clustree) - visual based way for picking best clusterins 
- other [statistical tests ](https://en.wikipedia.org/wiki/Cluster_analysis#Internal_evaluation)for choosing clusters 
- maybe something about graph-based clustering compared to centroid based clustersing?

#### Applications 

Create a sereies of start to finish exampls on:
- synthetic data. Find some way to simulate clustering data 
- a small real dataset (4K PBMC's is a good place to start)
- a large real dataset (Sanes Amacrine dataset)
- give some advice on choosing the best parameter space to iterate over
