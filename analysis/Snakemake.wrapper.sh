#!/bin/bash

# to run snakemake as batch job
# run in the data folder for this project


sbcmd="sbatch --cpus-per-task={cluster.cpus-per-task} \
--mem={cluster.mem} \
--time={cluster.time} \
--partition={cluster.partition} \
--output={cluster.output} \
--error={cluster.error} \
--job-name={cluster.name} \
{cluster.extra}"


snakefile=$1
cluster_json=$2

/data/swamyvs/anaconda3/envs/clusterEval/bin/snakemake -s $snakefile \
-pr --local-cores 2 --jobs 1999 \
--cluster-config $cluster_json \
--cluster "$sbcmd"  --latency-wait 120  \
-k --restart-times 0

