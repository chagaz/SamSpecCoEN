# SamSpecCoEN
Build sample-specific co-expression networks.


Overview
========
The goal of this code is to build sample-specific co-expression networks: 
All samples have the same network *structure*, but edge weights depend on the sample.
The co-expression networks are built from Pearson's correlation between gene expressions.
Each individual weights reflects the sample's contribution/deviation from this population-wide correlation.

The two options are:  
1. LIONESS [Kuijjer et al., 2015]: For a given sample, an edge-weight is the contribution of this sample to the global correlation.  
2. For a given sample, the edge-weight is the distance of that sample to the regression line fitting the expression of both genes. This quantifies how much this sample deviates from the population-wide behaviour.  

We also use ideas from [Zhang and Horvath, 2005] to build the global (population-wide) network, i.e. that a co-expression network should be approximately scale-free.  

This code is meant to be used on the ACES gene expression data [Staiger et al., 2013].  


Requirements
============
Python packages
---------------
* hd5py  
* matplotlib  
* numpy  
* memory_profiler (optional, for profiling memory usage)
* timeit (optional, for timing functions)

ACES
----
* Download from [http://ccb.nki.nl/software/aces/]http://ccb.nki.nl/software/aces/ 
* untar in this (SamSpecCoEN) directory
* add an empty file __init__.py under ACES/experiments
* make sure to have the required Python packages (in particular, xlrd)

Usage
=====
Creating sample-specific co-expression networks
-----------------------------------------------

The class for creating sample-specific co-expression neworks is `CoExpressionNetwork.py`. It can be used in the following manner:

### Networks for a single data set
```python CoExpressionNetwork.py DMFS outputs/U133A_combat_DMFS```
creates sample-specific coexpression networks for the entire dataset. The network structure (list of edges) is stored under `outputs/U122A_combat_DMFS/edges.gz` and its weights under `outputs/U122A_combat_DMFS/lioness/edges_weights.gz` (for the LIONESS approach) or `outputs/U122A_combat_DMFS/regline/edges_weights.gz` (for the regression line approach).

### Networks in a cross-validation setting
```python CoExpressionNetwork.py DMFS outputs/U133A_combat_DMFS -k 5```
creates 5 folds for cross-validation. The network structure (list of edges) is stored under `outputs/U122A_combat_DMFS/edges.gz` and its weights, for each fold from k=0 to k=4, under `outputs/U122A_combat_DMFS/<k>/lioness/edges_weights.gz` (for the LIONESS approach) or `outputs/U122A_combat_DMFS/<k>/regline/edges_weights.gz` (for the regression line approach). 

`outputs/U122A_combat_DMFS/<k>` also contains training and test indices and labels.

### Networks in a subtype-stratified cross-validation setting
In order to evaluate the expressiveness of the representation of samples by their sample-specific co-expression network, we use a subtype-stratified cross-validation setting similar to that described in [Allahyar & de Ridder, 2015].

To create the corresponding data folds and networks:
```
    for repeat in {0..9}
    do
	python setUpSubTypeStratifiedCV_writeIndices.py data/SamSpecCoEN ${repeat}
    done

    for repeat in {0..9}
    do
	for fold in {0..9}
	do
		python setUpSubTypeStratifiedCV_computeNetworks.py data/SamSpecCoEN ${fold} ${repeat}
	done
    done
```
This process can easily be parallelized.


Cross-validation experiments
----------------------------
The class for running a cross-validation experiment is `OuterCrossVal.py`. Internally, it uses `InnerCrossVal.py` to determine the best hyperparameters for the learning algorithm.

```python OuterCrossVal.py outputs/U133A_combat_DMFS lioness -o 5 -k 5 -m 400``` 
runs a 5-fold cross-validation experiment on the data stored in folds under `outputs/U133A_combat_DMFS`, for the LIONESS edge weights, using a 5-fold inner cross-validation loop, and returning at most 400 genes (following ACES/FERAL), for (for now) an L1-regularized logistic regression.


References
==========
Allahyar, A. and de Ridder, J. (2015). FERAL: network-based classifier with application to breast cancer outcome prediction. Bioinformatics, 31 (12).

Kuijjer, M.L., Tung, M., Yuan, G., Quackenbush, J., and Glass, K. (2015). Estimating sample-specific regulatory networks. arXiv:1505.06440 [q-Bio].  
 
Staiger, C., Cadot, S., Györffy, B., Wessels, L.F.A., and Klau, G.W. (2013). Current composite-feature classification methods do not outperform simple single-genes classifiers in breast cancer prognosis. Front Genet 4.  
  
Zhang, B., and Horvath, S. (2005). A General Framework for Weighted Gene Co-Expression Network Analysis. Statistical Applications in Genetics and Molecular Biology 4.

