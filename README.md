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
1. For a given sample, the edge-weight is the distance of that sample to the regression line fitting the expression of both genes. This quantifies how much this sample deviates from the population-wide behaviour.  

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
* ACES:	Download from [http://ccb.nki.nl/software/aces/]http://ccb.nki.nl/software/aces/ and untar in this (SamSpecCoEN) directory


References:  
-----------
Kuijjer, M.L., Tung, M., Yuan, G., Quackenbush, J., and Glass, K. (2015). Estimating sample-specific regulatory networks. arXiv:1505.06440 [q-Bio].  
 
Staiger, C., Cadot, S., Györffy, B., Wessels, L.F.A., and Klau, G.W. (2013). Current composite-feature classification methods do not outperform simple single-genes classifiers in breast cancer prognosis. Front Genet 4.  
  
Zhang, B., and Horvath, S. (2005). A General Framework for Weighted Gene Co-Expression Network Analysis. Statistical Applications in Genetics and Molecular Biology 4.

