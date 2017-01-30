# SamSpecCoEN
Build sample-specific co-expression networks.


Overview
========
The goal of this code is to build sample-specific co-expression networks: 
All samples have the same network *structure*, but edge *weights* depend on the sample.

To do this, we combine an existing network structure, given by a PPI network, with gene expression data.

Edge weights reflect the sample's deviation from the control correlation, or the "activity" of the corresponding edge.

Related work:  
LIONESS [Kuijjer et al., 2015]: For a given sample, an edge-weight is the contribution of this sample to the global correlation.  


Data:
This code is meant to be used on the ACES gene expression data [Staiger et al., 2013].  In this data set, the expression level of each gene is given *relative to the mean expression of the gene in the entire data set*: 

x^i_j = \log\left( \frac{I^i_j}{\frac{1}{n} \sum{u=1}^n I^u_j} \right\)

where I^i_j is the intensity of probe j (corresponding to gene j) for sample i.

Propositions:


1. REGLINE: For a given sample, the edge weight is the distance of that sample to the regression line fitting the expression of both genes, *on a healthy reference population*. This quantifies how much this sample deviates from the "normal" behaviour.  
2. MAHALANOBIS: For a given sample, the edge weight is the Mahalanobis distance of that sample to the 2D-Gaussian fitting the expression of both genes, *on a healthy reference population*. This quantifies how much this sample deviates from the "normal" behaviour.
3. SUM: For a given sample, the weight of edge (x_1, x_2) is x_1+x_2.
4. EUCLIDE: For a given sample, the weight of edge (x_1, x_2) is the Euclidean distance between x_1 and x_2.
5. EUCLTHR: For a given sample, the weight of edge (x_1, x_2) is the Euclidean distance between x_1 and x_2, unless x_1 or x_2 is negative, in which case it is 0. (Rationale: the edge can't be "on" if one of the genes is underexpressed).


Evaluation:
While one of the goals here is to use network-specific algorithms, at first we want to see whether we can build edge weights that are at least as expressive as node weights (that is to say, gene expression levels). We quantify this by cross-validated performance of both an L1-regularized and an L1/L2-regularized logistic regressions, trained on these weights.

Requirements
============
Python packages
---------------
* hd5py  
* matplotlib  
* numpy  
* memory_profiler (optional, for profiling memory usage)
* timeit (optional, for timing functions)
* python-glmnet for elastic net, see https://github.com/civisanalytics/python-glmnet

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

creates sample-specific coexpression networks for the entire dataset. The network structure (list of edges), which corresponds to that given by the .sif file `../ACES/experiments/data/KEGG_edges1210.sif`, is stored under `../outputs/U133A_combat_RFS/edges.gz`. The weights are stored under `outputs/U133A_combat_RFS/<method>/edges_weights.gz`, where `<method>` is one of `regline`, `mahalanobis`, `sum`, `euclide`, `euclthr`.

Cross-validation experiments
----------------------------
The class for running a cross-validation experiment is `OuterCrossVal.py`. Internally, it uses `InnerCrossVal.py` to determine the best hyperparameters for the learning algorithm.

### Subtype-stratified cross-validation 
This experiment is meant to be comparable to that of the FERAL paper: we build 10-fold cross-validations, with each fold having roughly the same number of samples from each class, and repeat this 10 times.

#### Create train/test folds
` python setUpSubTypeStratifiedCV_writeIndices.py RFS ../outputs/U133A_combat_RFS/subtype_stratified 10 0`
creates train and test indices for `repeat0` of the cross-validation procedure, stored as `train.indices` and `test.indices` under `../outputs/U133A_combat_RFS/subtype_stratified/repeat0/<fold idx>/` for `<fold idx>` ranging from 0 to 9.

#### Inner cross-validation
`python InnerCrossVal.py ../ACES ../outputs/U133A_combat_RFS ../outputs/U133A_combat_RFS/subtype_stratified/repeat0/fold0/ -t regline -k 5 -m 1000` 
runs an inner cross-validation loop (5 folds) on the training part of fold 0 of repeat 0, using the REGLINE edge weights as features, to determine optimal parameter(s) of the learning algorithm.
Results are stored under `../outputs/U133A_combat_RFS/subtype_stratified/repeat0/fold0/results/`

To run on all 10 folds:
`for fix in {0..9}; do python InnerCrossVal.py ../ACES ../outputs/U133A_combat_RFS ../outputs/U133A_combat_RFS/subtype_stratified/repeat0/fold${fix}/ regline -k 5 -m 1000; done` 

#### Outer cross-validation
Once the optimal parameters have been determined by inner cross-validation,
`python OuterCrossVal.py ../ACES ../outputs/U133A_combat_RFS/ ../outputs/U133A_combat_RFS/subtype_stratified/repeat0 -t regline -o 10 -k 5 -m 1000`
runs the outer loop of 10-fold cross-validation on repeat0, using the REGLINE edge weights as features, and selecting at most 1000 features.

The `--nodes` option allows you to run the exact same algorithm on the exact same folds, but using the node weights (i.e. gene expression data) directly instead of the edge weights, for comparison purposes (the network type is required but won't be used).

The `--cnodes` option allows you to run the exact same algorithm on the exact same folds, but using the node weights (i.e. gene expression data) directly instead of the edge weights, for comparison purposes (the network type is required but won't be used), and restricting the nodes used to the genes that are connected in the network.

The `--sfan` option allows you to run sfan [Azencott et al., 2013] to select nodes, using the structure of the co-expression network, using an l2-regularized logistic regression on the values (normalize gene expression) of the selected nodes for final prediction. In this case ```${sfan_dir}``` points to the ```sfan/code``` folder that you can obtain from [sfan's github repository](https://github.com/chagaz/sfan). 

The `--enet` option allows you to run an elastic net (l1/l2 regularization), using the python-glmnet package from [Civis Analytics](https://github.com/civisanalytics/python-glmnet).

#### Structure of the outputs/results directory
under `../outputs/U133A_combat_RFS/subtype_stratified/repeat<repeat_index>`:
```
- fold<fold_index>/: 
  - test.indices, test.labels, train.indices, train.labels
  - results/:
    - <network type>/: ('regline', 'euclide', etc.)
      featuresList, predValues, yte: results of inner cross-validation (l1-regularized logistic regression on edge weights).
     - nodes/:
        featuresList, predValues, yte: results of inner cross-validation (l1-regularized logistic regression on node weights).
     - nodes/:
        featuresList, predValues, yte: results of inner cross-validation (l1-regularized logistic regression on *connected* node weights).
    - sfan/:
		featuresList, predValues, yte: results of inner cross-validation (sfan on node weights).
    Each of <network type>/, nodes/ and cnodes/ contains
		  - enet/:
			same files, l1/l2-regularized version.

- results/:
    - <network type>/: ('regline', 'euclide', etc.)
      cix.pdf, fov.pdf, results.txt: results of outer cross-validation (l1-regularized logistic regression on edge weights).
    - nodes/:
      cix.pdf, fov.pdf, results.txt: results of outer cross-validation (l1-regularized logistic regression on node weights).
    - cnodes/:
      cix.pdf, fov.pdf, results.txt: results of outer cross-validation (l1-regularized logistic regression on *connected* node weights).
     - sfan/:
       cix.pdf, fov.pdf, results.txt: results of outer cross-validation (sfan on node weights).
    Each of <network type>/, nodes/ and cnodes/ contains
		  - enet/:
			same files, l1/l2-regularized version.
```

Analysis of results
-------------------
### Get final set of selected features
To generate the set of features selected in k of the 10 cross-validation folds, as well as their predictivity.

` python analyze_final.py ../ACES ../outputs/U133A_combat_RFS \
         ../outputs/U133A_combat_RFS/subtype_stratified/repeat0  \
         -t regline -o 10 -k 5 -m 1000 -u 10`
creates, under `../outputs/U133A_combat_RFS/subtype_stratified/repeat0/results/regline/`, the following files:
* `final_selection_genes.txt`: List of EntrezIDs of selected genes + number of edges they belong to
* `final_selection_results.txt`: Cross-validated predictivity (ridge regression) of selected features.

### Map Entrez IDs to gene symbols
`python map_Entrez_to_gene_symbol.py \
	../outputs/U133A_combat_RFS/KEGG_edges1210/subtype_stratified/repeat0/results/nodes/final_selection_genes.txt \
  ../outputs/U133A_combat_RFS/KEGG_edges1210/subtype_stratified/repeat0/results/nodes/final_selection_genes_symbols.txt` 
converts a list of Entrez IDs in gene symbols.

### Compare selected genes to reference gene sets
`python compare_genesets.py \
    ../outputs/U133A_combat_RFS/KEGG_edges1210/subtype_stratified/repeat0/results/regline/ \
    ../FERAL_supp_data/Allahyar.285.sup.1 \
    -l ../outputs/U133A_combat_RFS/KEGG_edges1210/genes_in_network_GeneSymbols.txt`
compares (hypergeometric test) the list of selected genes to all the reference gene sets under `../FERAL_supp_data/Allahyar.285.sup.1`.

### GO enrichment analysis
Can be done using the web server at http://pantherdb.org/ (overrepresentation test)
To easily copy-paste the list of selected genes: xsel -b < final_selection_genes_symbols.txt

Analysis of multiple repeats
-----------------------------
### Plot performance metrics (mean + std)
`python create_plots.py \
	../outputs/U133A_combat_RFS/KEGG_edges1210/subtype_stratified -r 5` plots
	number of selected features, consistency index, Fisher's overlap, and AUROC
	averaged over 5 repeated cross-validated experiments.

Creates, under `../outputs/U133A_combat_RFS/KEGG_edges1210/subtype_stratified/results`,
the following plots:
* `subtype_stratified_auc.pdf` (AUROC)
* `subtype_stratified_numf.pdf` (number of features)
* `subtype_stratified_cix.pdf` (consistency index)
* `subtype_stratified_fov.pdf` (Fisher's overlap).


### Analyze the features that were selected across a fraction of folds (repeated)
`python multiple_repeats_analysis.py ../ACES ../outputs/U133A_combat_RFS/KEGG_edges1210 \
         ../outputs/U133A_combat_RFS/subtype_stratified/KEGG_edges1210  \
         -t regline -o 10 -r 5 -u 50` creates, under `../outputs/U133A_combat_RFS/subtype_stratified/results/regline/`, the following files:
* `final_selection_genes.txt`: List of EntrezIDs of selected genes + number of edges they belong to
* `final_selection_results.txt`: Cross-validated predictivity (ridge regression) of selected features.
 

Task list
=========
- [ ] Propose new ways of computing edge weights.



References
==========
Allahyar, A. and de Ridder, J. (2015). FERAL: network-based classifier with application to breast cancer outcome prediction. Bioinformatics, 31 (12).

Azencott, C.-A., Grimm, D., Sugiyama, M., Kawahara, Y., and Borgwardt, K. M. (2013). Efficient network-guided multi-locus association mapping with graph cuts. Bioinformatics, 29(13): i171—i179.

Kuijjer, M.L., Tung, M., Yuan, G., Quackenbush, J., and Glass, K. (2015). Estimating sample-specific regulatory networks. arXiv:1505.06440 [q-Bio].  
 
Staiger, C., Cadot, S., Györffy, B., Wessels, L.F.A., and Klau, G.W. (2013). Current composite-feature classification methods do not outperform simple single-genes classifiers in breast cancer prognosis. Front Genet 4.  
