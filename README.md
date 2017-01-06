# SamSpecCoEN
Build sample-specific co-expression networks.


Overview
========
The goal of this code is to build sample-specific co-expression networks: 
All samples have the same network *structure*, but edge *weights* depend on the sample.

To do this, we combine an existing network structure, given by a PPI network, with gene expression data.

Edge weights reflect the sample's contribution to the population-wide correlation, its deviation from the control correlation, or whether this edge can be considered active or not.

Related work:  
LIONESS [Kuijjer et al., 2015]: For a given sample, an edge-weight is the contribution of this sample to the global correlation.  


Data:
This code is meant to be used on the ACES gene expression data [Staiger et al., 2013].  In this data set, the expression level of each gene is given *relative to the mean expression of the gene in the entire data set*: 

x^i_j = \log\left( \frac{I^i_j}{\frac{1}{n} \sum{u=1}^n I^u_j} \right\)

where I^i_j is the intensity of probe j (corresponding to gene j) for sample i.

Propositions:


1. REGLINE: For a given sample, the edge weight is the distance of that sample to the regression line fitting the expression of both genes, *on a healthy reference population*. This quantifies how much this sample deviates from the "normal" behaviour.  
2. SUM: For a given sample, the weight of edge (x_1, x_2) is x_1+x_2.
3. EUCLIDE: For a given sample, the weight of edge (x_1, x_2) is the Euclidean distance between x_1 and x_2.
4. EUCLTHR: For a given sample, the weight of edge (x_1, x_2) is the Euclidean distance between x_1 and x_2, unless x_1 or x_2 is negative, in which case it is 0. (Rationale: the edge can't be "on" if one of the genes is underexpressed).

Evaluation:
While one of the goals here is to use network-specific algorithms, at first we want to see whether we can build edge weights that are at least as expressive as node weights (that is to say, gene expression levels). We quantify this by cross-validated performance of a L1-regularized logistic regression trained on these weights.

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
This is the breast cancer gene expression data we want to classify (according to recurrence-free vs. not-recurrence-free after five years). It also contains several PPI networks.
* Download from [http://ccb.nki.nl/software/aces/](http://ccb.nki.nl/software/aces/)
* untar in this (SamSpecCoEN) directory
* add an empty file __init__.py under ACES/experiments
* make sure to have the required Python packages (in particular, xlrd)

ArrayExpress E-MTAB-62
----------------------
This is a reference healthy population data.
* Download from [https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-62/](ArrayExpress)
* Store under ArrayExpress/ at the root of this repository
* run `cd code; python preprocess-MTAB-62.py` to pre-process the data (similar normalization as the ACES data).

Usage
=====
Creating sample-specific co-expression networks
-----------------------------------------------

The class for creating sample-specific co-expression neworks is `CoExpressionNetwork.py`. It can be used in the following manner:

`python CoExpressionNetwork.py RFS ../ACES/experiments/data/KEGG_edges1210.sif ../ArrayExpress/postproc/MTAB-62.h5 ../outputs/U133A_combat_RFS`

creates sample-specific coexpression networks for the entire dataset. The network structure (list of edges), which corresponds to that given by the .sif file `../ACES/experiments/data/KEGG_edges1210.sif`, is stored under `../outputs/U133A_combat_RFS/edges.gz`. The weights are stored under `outputs/U133A_combat_RFS/<method>/edges_weights.gz`, where `<method>` is one of `regline`, `sum`, `euclide`, `euclthr`.

Cross-validation experiments
----------------------------
The class for running a cross-validation experiment is `OuterCrossVal.py`. Internally, it uses `InnerCrossVal.py` to determine the best hyperparameters for the learning algorithm.

### Subtype-stratified cross-validation 
This experiment is meant to be comparable to that of the FERAL paper: we build 10-fold cross-validations, with each fold having roughly the same number of samples from each class, and repeat this 10 times.

#### Create train/test folds
` python setUpSubTypeStratifiedCV_writeIndices.py RFS ../outputs/U133A_combat_RFS/subtype_stratified 10 0`
creates train and test indices for `repeat0` of the cross-validation procedure, stored as `train.indices` and `test.indices` under `../outputs/U133A_combat_RFS/subtype_stratified/repeat0/<fold idx>/` for `<fold idx>` ranging from 0 to 9.

#### Inner cross-validation
`python InnerCrossVal.py ../ACES ../outputs/U133A_combat_RFS ../outputs/U133A_combat_RFS/subtype_stratified/repeat0/fold0/ regline -k 5 -m 1000` 
runs an inner cross-validation loop (5 folds) on the training part of fold 0 of repeat 0, using the REGLINE edge weights as features, to determine optimal parameter(s) of the learning algorithm.
Results are stored under `../outputs/U133A_combat_RFS/subtype_stratified/repeat0/fold0/results/`

To run on all 10 folds:
`for fix in {0..9}; do python InnerCrossVal.py ../ACES ../outputs/U133A_combat_RFS ../outputs/U133A_combat_RFS/subtype_stratified/repeat0/fold${fix}/ regline -k 5 -m 1000; done` 

#### Outer cross-validation
Once the optimal parameters have been determined by inner cross-validation,
`python OuterCrossVal.py ../ACES ../outputs/U133A_combat_RFS/subtype_stratified/repeat0 regline ../outputs/U133A_combat_RFS/subtype_stratified/repeat0/results/regline -o 10 -k 5 -m 1000`
runs the outer loop of 10-fold cross-validation on repeat0, using the REGLINE edge weights as features, and selecting at most 1000 features.

The `--nodes` option allows you to run the exact same algorithm on the exact same folds, but using the node weights (i.e. gene expression data) directly instead of the edge weights, for comparison purposes (the network type is required but won't be used).

The `--sfan` option allows you to run sfan [Azencott et al., 2013] to select nodes, using the structure of the co-expression network, using an l2-regularized logistic regression on the values (normalize gene expression) of the selected nodes for final prediction. In this case ```${sfan_dir}``` points to the ```sfan/code``` folder that you can obtain from [sfan's github repository](https://github.com/chagaz/sfan). This will also run an l2-regularized logistic regression only on the nodes that are connected in the network (i.e. not using sfan at all).

The `--enet` option allows you to run an elastic net (l1/l2 regularization). Currently, it uses scikit-learn's implementation, which has some issues. This should be re-implemented using [spams](http://spams-devel.gforge.inria.fr/) or [L1L2py](https://pypi.python.org/pypi/L1L2Py/1.0.5). 

#### Structure of the outputs/results directory
under `../outputs/U133A_combat_RFS/subtype_stratified/repeat<repeat_index>`:
```
- fold<fold_index>/: 
  - test.indices, test.labels, train.indices, train.labels
  - results/:
    - <network type>/: ('regline', 'euclide', etc.)
      - featuresList, predValues, yte: results of inner cross-validation (l1-regularized logistic regression on edge weights).
      - enet/:
          featuresList, predValues, yte: results of inner cross-validation (l1/l2-regularized logistic regression on edge weights).
     - nodes/:
        - featuresList, predValues, yte: results of inner cross-validation (l1-regularized logistic regression on node weights).
        - enet/:
           featuresList, predValues, yte: results of inner cross-validation (l1/l2-regularized logistic regression on node weights).
        - sfan/:
            featuresList, predValues, yte: results of inner cross-validation (sfan on node weights).
            
- results/:
    - <network type>/: ('regline', 'euclide', etc.)
      - cix.pdf, fov.pdf, results.txt: results of outer cross-validation (l1-regularized logistic regression on edge weights).
      - enet/:
          cix.pdf, fov.pdf, results.txt: results of outer cross-validation (l1/l2-regularized logistic regression on edge weights).
     - nodes/:
        - cix.pdf, fov.pdf, results.txt: results of outer cross-validation (l1-regularized logistic regression on node weights).
        - enet/:
           cix.pdf, fov.pdf, results.txt: results of outer cross-validation (l1/l2-regularized logistic regression on node weights).
        - sfan/:
            cix.pdf, fov.pdf, results.txt: results of outer cross-validation (sfan on node weights).
```

Task list
=========
- [ ] Validate the different weights of computing edge weights (i.e. ensure they do what they're supposed to) in CoExpressionNetwork.py
- [ ] Compare the performance of the l1-regularized subtype-stratified cross-validation using the various edge weights as features to that of using directly the gene expression levels as features.
- [ ] Propose new weights of computing edge weights.
- [ ] Implement the `--enet` option with [spams](http://spams-devel.gforge.inria.fr/) or [L1L2py](https://pypi.python.org/pypi/L1L2Py/1.0.5). 



References
==========
Allahyar, A. and de Ridder, J. (2015). FERAL: network-based classifier with application to breast cancer outcome prediction. Bioinformatics, 31 (12).

Azencott, C.-A., Grimm, D., Sugiyama, M., Kawahara, Y., and Borgwardt, K. M. (2013). Efficient network-guided multi-locus association mapping with graph cuts. Bioinformatics, 29(13): i171—i179.

Kuijjer, M.L., Tung, M., Yuan, G., Quackenbush, J., and Glass, K. (2015). Estimating sample-specific regulatory networks. arXiv:1505.06440 [q-Bio].  
 
Staiger, C., Cadot, S., Györffy, B., Wessels, L.F.A., and Klau, G.W. (2013). Current composite-feature classification methods do not outperform simple single-genes classifiers in breast cancer prognosis. Front Genet 4.  
