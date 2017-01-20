# @Author 
# Chloe-Agathe Azencott
# chloe-agathe.azencott@mines-paristech.fr

import argparse
import h5py

import matplotlib # in a non-interactive environment
matplotlib.use('Agg') # in a non-interactive environment
import matplotlib.pyplot as plt

# import warnings
# warnings.filterwarnings("error", category=RuntimeWarning)

import gzip
import numpy as np
import os
import scipy.stats as st
import sys

from sklearn import metrics, linear_model, model_selection

import glmnet

import InnerCrossVal

orange_color = '#d66000'
blue_color = '#005599'
network_types = ['regline', 'mahalan', 'sum', 'euclide', 'euclthr'] # possible network weight types

def color_boxplot(box):
    """
    Customize the colors of a box plot.

    Input
    -----
    box: dict
        matplotlib boxplot dictionary, as returned by plt.boxplot(patch_artist=True)
    """
    for patch in box['boxes']:
        patch.set_edgecolor(blue_color)
        patch.set_facecolor('white')
    for patch in box['medians']:
        patch.set_color(orange_color)
    for patch in box['whiskers']:
        patch.set_color(blue_color)            
    for patch in box['fliers']:
        patch.set_color(blue_color)            
    for patch in box['caps']:
        patch.set_color(blue_color)            


class OuterCrossVal(object):
    """ Manage the outer cross-validation loop for learning on sample-specific co-expression networks.

    Attributes
    ----------
    self.aces_data_root: path
        Path to folder containing ACES data
    self.innercv_root: path
        Path to folder containing outputs of inner cross-validation.
    self.network_root: path
        Path to folder containing network skeleton and edges.
    self.network_type: string
        Type of network to work with
    self.nr_inner_folds: int
        Number of folds for the inner cross-validation loop.
    self.nr_outer_folds: int
        Number of folds for the outer cross-validation loop.
    self.max_nr_feats: int
        Maximum number of features to return.
        Default value=400, as in [Staiger et al.]
    self.true_labels: (numSamples, ) array
        True labels for all samples.
    self.pred_labels: (numSamples, ) array
        Predicted labels for all samples, in the same order as true_labels.
    self.pred_values: (numSamples, ) array
        Probability estimates for all samples, in the same order as true_labels
    self.features_list: list of list
        List of list of indices of the selected features.
    self.num_features: float
        Total number of features
    self.use_nodes: bool
        Whether to use node weights rather than edge weights as features.
    self.sfan_path: path
            Path to sfan code.        

    Reference
    ---------
    Staiger, C., Cadot, S., Gyoerffy, B., Wessels, L.F.A., and Klau, G.W. (2013).
    Current composite-feature classification methods do not outperform simple single-genes
    classifiers in breast cancer prognosis. Front Genet 4.  
    """
    def __init__(self, aces_data_root, ntwk_root, innercv_root, network_type, num_samples,
                 nr_inner_folds, nr_outer_folds, max_nr_feats=400, use_nodes=False,
                 use_sfan=False, sfan_path=None, use_enet=None):
        """
        Parameters
        ----------
        aces_data_root: path
            Path to folder containing ACES data
        ntwk_root: path
            Path to folder containing network skeleton and edges.
        innercv_root: path
            Path to folder containing outputs of inner cross-validation.
        network_type: string
            Type of network to work with
            Correspond to a folder in dataFoldRoot
            Possible value taken from network_types
        nr_inner_folds: int
            Number of folds for the inner cross-validation loop.
        max_nr_feats: int
            Maximum number of features to return.
            Default value=400.
        use_nodes: bool
            Whether to use node weights rather than edge weights as features.
           (This does not make use of the network information.)
        use_sfan: bool
            Whether to use sfan on {node weights + network structure} rather than edge weights.
        use_enet: bool
            Whether to use l1/l2 (elastic net) regularization rather than l1.
        sfan_path: path
            Path to sfan code.
        """
        self.true_labels = np.ones((num_samples, ))
        self.pred_labels = np.ones((num_samples, ))
        self.pred_values = np.ones((num_samples, ))
        self.features_list = []
        
        self.nr_outer_folds = nr_outer_folds
        self.nr_inner_folds = nr_inner_folds
        self.max_nr_feats = max_nr_feats

        self.aces_data_root = aces_data_root
        self.innercv_root = innercv_root
        self.network_root = ntwk_root
        self.network_type = network_type

        self.use_nodes = use_nodes
        self.use_enet = use_enet
        self.use_sfan = use_sfan
        if self.use_sfan:
            self.use_nodes = True

        # Read the number of features
        if self.use_nodes:
            # Read ACES data
            sys.path.append(aces_data_root)
            from datatypes.ExpressionDataset import HDF5GroupToExpressionDataset
            f = h5py.File("%s/experiments/data/U133A_combat.h5" % aces_data_root)
            aces_data = HDF5GroupToExpressionDataset(f['U133A_combat_RFS'], checkNormalise=False)
            self.num_features = aces_data.expressionData.shape[1]
            f.close()
        else:
            self.num_features = np.loadtxt("%s/edges.gz" % self.network_root).shape[0]
        
        
    # ================== l1-regularization ==================
    def run_outer_l1_logreg(self):
        """ Run the outer loop of the experiment, for an l1-regularized logistic regression.

        Updated attributes
        ------------------
        true_labels: (num_samples, ) array
            True labels for all samples.
        pred_labels: (num_samples, ) array
            Predicted labels for all samples, in the same order as true_labels.
        pred_values: (num_samples, ) array
            Probability estimates for all samples, in the same order as true_labels.
        features_list: list of list
            List of list of indices of the selected features.
        """
        for fold in range(self.nr_outer_folds):
            sys.stdout.write("Working on fold number %d\n" % fold)

            # Read the test indices
            data_fold_root = '%s/fold%d' % (self.innercv_root, fold)
            te_indices = np.loadtxt('%s/test.indices' % fold_root, dtype='int')
            
            # Create an InnerCrossVal
            icv = InnerCrossVal.InnerCrossVal(self.aces_data_root, data_fold_root,
                                              self.network_root, self.network_type,
                                              self.nr_inner_folds, self.max_nr_feats, self.use_nodes)
            
            # Get predictions and selected features for the inner loop
            reg_params = [2.**k for k in range(-7, -1)]
            [pred_values_fold, features_fold] = icv.run_inner_l1_logreg(reg_params=reg_params)

            # Update self.true_labels, self.pred_labels, self.features_list
            self.true_labels[te_indices] = icv.y_te
            self.pred_values[te_indices] = pred_values_fold
            self.features_list.append(features_fold)

        # Convert probability estimates in labels
        self.pred_labels = np.array(self.pred_values > 0, dtype='int')
    # ================== End l1-regularization ==================


    # ================== l2-regularization ==================
    def run_outer_l2_logreg(self):
        """ Run the outer loop of the experiment, for an l2-regularized logistic regression.

        Updated attributes
        ------------------
        true_labels: (num_samples, ) array
            True labels for all samples.
        pred_labels: (num_samples, ) array
            Predicted labels for all samples, in the same order as true_labels.
        pred_values: (num_samples, ) array
            Probability estimates for all samples, in the same order as true_labels.
        features_list: list of list
            List of list of indices of the selected features.
        """
        for fold in range(self.nr_outer_folds):
            sys.stdout.write("Working on fold number %d\n" % fold)

            # Read the test indices
            data_fold_root = '%s/fold%d' % (self.data_innercv_root, fold)
            te_indices = np.loadtxt('%s/test.indices' % data_fold_root, dtype='int')
            
            # Create an InnerCrossVal
            icv = InnerCrossVal.InnerCrossVal(self.aces_data_root, data_fold_root,
                                              self.network_root, self.network_type,
                                              self.nr_inner_folds, self.max_nr_feats, self.use_nodes)
            
            # Get predictions and selected features for the inner loop
            reg_params = [10.**k for k in range(-4, 1)]
            [pred_values_fold, features_fold] = icv.run_inner_l2_logreg(reg_params=reg_params)

            # Update self.true_labels, self.pred_labels, self.features_list
            self.true_labels[te_indices] = icv.y_te
            self.pred_values[te_indices] = pred_values_fold
            self.features_list.append(features_fold)

        # Convert probability estimates in labels
        self.pred_labels = np.array(self.pred_values > 0, dtype='int')
    # ================== End l1-regularization ==================


    # ================== l1/l2-regularization ==================
    def run_outer_enet_logreg(self):
        """ Run the outer loop of the experiment, for an l1/l2-regularized logistic regression.

        Updated attributes
        ------------------
        true_labels: (num_samples, ) array
            True labels for all samples.
        pred_labels: (num_samples, ) array
            Predicted labels for all samples, in the same order as true_labels.
        pred_values: (num_samples, ) array
            Probability estimates for all samples, in the same order as true_labels.
        features_list: list of list
            List of list of indices of the selected features.
        """
        for fold in range(self.nr_outer_folds):
            sys.stdout.write("Working on fold number %d\n" % fold)

            # Read the test indices
            data_fold_root = '%s/fold%d' % (self.data_innercv_root, fold)
            te_indices = np.loadtxt('%s/test.indices' % data_fold_root, dtype='int')
            
            # Create an InnerCrossVal
            icv = InnerCrossVal.InnerCrossVal(self.aces_data_root, data_fold_root,
                                              self.network_root, self.network_type,
                                              self.nr_inner_folds, self.max_nr_feats, self.use_nodes)

            # Get predictions and selected features for the inner loop
            lbd_values = []
            l1_ratio_values = [0.5, 0.75, 0.95]
            reg_params=[lbd_values, l1_ratio_values]
            [pred_values_fold, features_fold] = icv.run_inner_enet_logreg(reg_params=reg_params)

            # Update self.true_labels, self.pred_labels, self.features_list
            self.true_labels[te_indices] = icv.y_te
            self.pred_values[te_indices] = pred_values_fold
            self.features_list.append(features_fold)

        # Convert probability estimates in labels
        self.pred_labels = np.array(self.pred_values > 0, dtype='int')
    # ================== End l1/l2-regularization ==================


    def read_inner_results(self, subdir_name=None):
        """ Read the results of the inner loop of the experiment.
        
        Parameters
        ----------
        subdir_name: {string, None}
            Name of subdirectory of innercv_results_dir that contains the results to process.
        
        Updated attributes
        ------------------
        true_labels: (num_samples, ) array
            True labels for all samples.
        pred_labels: (num_samples, ) array
            Predicted labels for all samples, in the same order as true_labels.
        pred_values: (num_samples, ) array
            Probability estimates for all samples, in the same order as true_labels.
        features_list: list of list
            List of list of indices of the selected features.
        """
        for fold_idx in range(self.nr_outer_folds):
            sys.stdout.write("Reading results for fold %d\n" % fold_idx)

            # Read the test indices
            te_indices = np.loadtxt('%s/fold%d/test.indices' % (self.innercv_root, fold_idx),
                                    dtype='int')
            
            # Read results from InnerCrossVal
            if self.use_nodes:
                if self.use_sfan:
                    innercv_results_dir = "%s/fold%d/results/nodes/sfan" % (self.innercv_root, fold_idx)
                elif self.use_enet:
                    innercv_results_dir = '%s/fold%d/results/nodes/enet' % (self.innercv_root, fold_idx)
                else:
                    innercv_results_dir = '%s/fold%d/results/nodes/' % (self.innercv_root, fold_idx)
            elif self.use_enet:
                innercv_results_dir = '%s/fold%d/results/%s/enet' % (self.innercv_root, fold_idx,
                                                                     self.network_type)
            else:
                innercv_results_dir = '%s/fold%d/results/%s/' % (self.innercv_root, fold_idx,
                                                                 self.network_type)

            if subdir_name:
                innercv_results_dir = '%s/%s' % (innercv_results_dir, subdir_name)

            yte_fname = '%s/yte' % innercv_results_dir
            pred_values_fname = '%s/predValues' % innercv_results_dir        
            features_list_fname = '%s/featuresList' % innercv_results_dir

            # Only use fold if number of selected features is NOT maximal
            fold_features_list = np.loadtxt(features_list_fname, dtype='int')
            if not len(fold_features_list) == self.max_nr_feats:
                self.true_labels[te_indices] = np.loadtxt(yte_fname, dtype='int')
                self.pred_values[te_indices] = np.loadtxt(pred_values_fname)
                self.features_list.append(fold_features_list)

        # Convert probability estimates in labels
        self.pred_labels = np.array(self.pred_values > 0, dtype='int')


    # ================== Sfan ==================
    def run_outer_sfan(self):
        """ Run the outer loop of the experiment, for an l2-regularized logistic regression
        with sfan.

        Updated attributes
        ------------------
        true_labels: (num_samples, ) array
            True labels for all samples.
        pred_labels: (num_samples, ) array
            Predicted labels for all samples, in the same order as true_labels.
        pred_values: (num_samples, ) array
            Probability estimates for all samples, in the same order as true_labels.
        features_list: list of list
            List of list of indices of the selected features.
        """
        for fold in range(self.nr_outer_folds):
            sys.stdout.write("Working on fold number %d\n" % fold)

            # Read the test indices
            data_fold_root = '%s/fold%d' % (self.data_root, fold)
            te_indices = np.loadtxt('%s/test.indices' % data_fold_root, dtype='int')
            
            # Create an InnerCrossVal
            icv = InnerCrossVal.InnerCrossVal(self.aces_data_path, data_fold_root,
                                              self.network_type, self.nr_inner_folds,
                                              self.max_nr_feats, use_nodes=True,
                                              use_sfan=True, sfan_path=self.sfan_path)
                                              

            # Get predictions and selected features for the inner loop
            reg_params=[itertools.product([10.**k for k in range(-3, 3)],
                                          [2.**k for k in range(-8, -2)]),
                        [10.**k for k in range(-3, 3)]]
            [pred_values_fold, features_fold] = icv.run_inner_sfan(reg_params=reg_params)

            # Update self.true_labels, self.pred_labels, self.features_list
            self.true_labels[te_indices] = icv.y_te
            self.pred_values[te_indices] = pred_values_fold
            self.features_list.append(features_fold)

        # Convert probability estimates in labels
        self.pred_labels = np.array(self.pred_values > 0, dtype='int')
    # ================== End sfan ==================
            
        
    def compute_auc(self):
        """ Compute the AUC of the experiment.

        Returns
        -------
        auc: float
           Area under the ROC curve for the experiment.
        """
        return metrics.roc_auc_score(self.true_labels, self.pred_values)
        

    def compute_fisher_overlap(self):
        """ Compute the pairwise Fisher overlaps between the sets of selected features.

        Fisher overlap = -log10 of the p-value of the Fisher exact test for the following
        contingency table:
               A    notA
        B   |  a  |  b  |
        notB|  c  |  d  |
        with 'greater' as the alternate hypothesis.
        A small p-value (high -log10) means a small probability to observe
        an overlap between A and B greater than a by chance.

        Returns
        -------
        fovList: list
           List of pairwise Fisher overlaps between the sets of selected features.

        Reference
        ---------
        Staiger, C., Cadot, S., Gyoerffy, B., Wessels, L.F.A., and Klau, G.W. (2013).
        Current composite-feature classification methods do not outperform simple single-genes
        classifiers in breast cancer prognosis. Front Genet 4.  

        """
        allFeatures = set(range(self.num_features))
        fov_list = []
        for set_idx1 in range(len(self.features_list)):
            feature_set1 = set(self.features_list[set_idx1].tolist())
            for set_idx2 in range(set_idx1+1, len(self.features_list)):
                feature_set2 = set(self.features_list[set_idx2].tolist())
                contingency = [[len(feature_set1.intersection(feature_set2)),
                                len(feature_set2.difference(feature_set1))],
                               [len(feature_set1.difference(feature_set2)),
                                len(allFeatures.difference(feature_set1.union(feature_set2)))]]
                pval = st.fisher_exact(contingency, alternative='greater')[1]
                if pval > 0:
                    fov_list.append(-np.log10(pval))
                else:
                    fov_list.append(-200)
        return fov_list


    def compute_consistency(self):
        """ Compute the pairwise consistency indices between the sets of selected features.

        Returns
        -------
        cix_list: list
            List of pairwise consistency indices between the sets of selected features.

        Reference
        ---------
        Kuncheva, L.I. (2007).
        A Stability Index for Feature Selection. AIA, pp. 390--395.
        """
        cix_list = []
        for set_idx1 in range(len(self.features_list)):
            feature_set1 = set(self.features_list[set_idx1])
            for set_idx2 in range(set_idx1+1, len(self.features_list)):
                feature_set2 = set(self.features_list[set_idx2])
                observed = float(len(feature_set1.intersection(feature_set2)))
                expected = len(feature_set1) * len(feature_set2) / float(self.num_features)
                maxposbl = float(min(len(feature_set1), len(feature_set2)))
                if expected == maxposbl:
                    cix_list.append(0.)
                else:
                    cix_list.append((observed - expected) / (maxposbl - expected))
        return cix_list
        
        
    def compute_pearson(self):
        """ Compute pairwise Pearson's correlations between indicator vectors of selected features.

        Returns
        -------
        prs_list: list
            List of pariwise Pearson's correlations between indicator vectors of selected features.

        Reference
        ---------
        Nogueira & Brown (2016).
        Measuring the Stability of Feature Selection, ECML 2016.
        """
        prs_list = []
        for set_idx1 in range(len(self.features_list)):
            feature_set1 = np.zeros((self.num_features, ))
            feature_set1[self.features_list[set_idx1]] = 1
            for set_idx2 in range(set_idx1+1, len(self.features_list)):
                feature_set2 = np.zeros((self.num_features, ))
                feature_set2[self.features_list[set_idx2]] = 1
                prs_list.append(st.pearsonr(feature_set1, feature_set2)[0])
        return prs_list


    def compute_overlap_histogram(self):
        """ Compute the number of features selected by exactly 1 to exactly k folds.

        Returns
        -------
        overlaps_list = list
            List of numbers of features selected by exactly 1 to exactly k folds.
        """
        selected_features_dict = {} # feat_idx:number_of_times_selected
        for feature_set in self.features_list:
            for feat_idx in feature_set:
                if not selected_features_dict.has_key(feat_idx):
                    selected_features_dict[feat_idx] = 1
                else:
                    selected_features_dict[feat_idx] += 1
        num_selected_features = len(selected_features_dict.keys())

        overlaps_dict = {} # number_of_times_selected:number_of_features
        for feat_idx, number_of_times_selected in selected_features_dict.iteritems():
            try:
                assert number_of_times_selected <= len(self.features_list)
            except AssertionError:
                print feat_idx, number_of_times_selected
                sys.stderr.write("Error in computing the number of times a feature was selected.\n")
                sys.stderr.write("Aborting.\n")
                sys.exit(-1)
            if not overlaps_dict.has_key(number_of_times_selected):
                overlaps_dict[number_of_times_selected] = 1
            else:
                overlaps_dict[number_of_times_selected] += 1

        overlaps_list = []
        for number_of_times_selected in range(1, len(self.features_list)+1):
            if not overlaps_dict.has_key(number_of_times_selected):
                overlaps_list.append(0.)
            else:
                #overlaps_list.append(float(overlaps_dict[number_of_times_selected])/num_selected_features)
                overlaps_list.append(overlaps_dict[number_of_times_selected])
        return overlaps_list
        
        
    def write_results(self, results_dir):
        """ Create results files.

        Parameter
        ---------
        results_dir: path
            Where to save results.

        Created files
        -------------
        results.txt:
            Number of selected features (per fold)
            Average number of selected features
            AUC
            Fisher overlaps (per fold)
            Consistency index (per fold)
            Pearson's consistency (per fold).

        fov.pdf:
            Box plot of Fisher overlaps.

        cix.pdf:
            Box plot of consistencies.        

        prs.pdf:
            Box plot of Pearson consistencies.        

        ovl.pdf:
            Bar chat of raction of selected features selected exactly k times.
        """

        # Open results file for writing
        res_fname = '%s/results.txt' % results_dir
        with open(res_fname, 'w') as f:
            # Write number of folds used
            f.write("Number of folds used:\t")
            f.write("%s\n" % len(self.features_list))
            
            # Write number of selected features
            f.write("Number of features selected per fold:\t")
            f.write("%s\n" % " ".join(["%d" % len(x) for x in self.features_list]))

            # Write average number of selected features
            f.write("Average number of selected features:\t")
            f.write("%d\n" % np.mean([len(x) for x in self.features_list]))

            # Write AUC
            f.write("AUC:\t%.2f\n" % self.compute_auc())

            # Write the stability (Fisher overlap)
            fov_list = self.compute_fisher_overlap()
            f.write("Stability (Fisher overlap):\t")
            f.write("%s\n" % ["%.2e" % x for x in fov_list])
            f.write("Average stability (Fisher overlap):\t")
            f.write("%.2e\n" % np.mean(fov_list))

            # Write the stability (consistency index)
            cix_list = self.compute_consistency()
            f.write("Stability (Consistency Index):\t")
            f.write("%s\n" % ["%.2e" % x for x in cix_list])
            f.write("Average stability (Consistency Index):\t")
            f.write("%.2e\n" % np.mean(cix_list))

            # Write the stability (Pearson)
            prs_list = self.compute_pearson()
            f.write("Stability (Pearson):\t")
            f.write("%s\n" % ["%.2e" % x for x in prs_list])
            f.write("Average stability (Pearson):\t")
            f.write("%.2e\n" % np.mean(prs_list))

            # Write the histogram of overlaps
            overlaps_list = self.compute_overlap_histogram()
            f.write("Number of features selected k times:\t")
            f.write("%s\n" % ["%d" % x for x in overlaps_list])
            f.close()
                    

        # Plot the stability (Fisher overlap)
        fov_fname = '%s/fov.pdf' % results_dir
        plt.figure()
        box = plt.boxplot(fov_list, 0, '+', patch_artist=True)
        color_boxplot(box)
        plt.title('Fisher overlap')
        plt.ylabel('-log10(p-value)')
        plt.savefig(fov_fname, bbox_inches='tight')

        # Plot the stability (consistency index)
        cix_fname = '%s/cix.pdf' % results_dir
        plt.figure()
        box = plt.boxplot(cix_list, 0, '+', patch_artist=True)
        color_boxplot(box)
        plt.title('Consistency Index')
        plt.savefig(cix_fname, bbox_inches='tight')

        # Plot the stability (Pearson)
        prs_fname = '%s/prs.pdf' % results_dir
        plt.figure()
        box = plt.boxplot(prs_list, 0, '+', patch_artist=True)
        color_boxplot(box)
        plt.title("Pearson's consistency")
        plt.savefig(prs_fname, bbox_inches='tight')

        # Plot the histogram of overlaps
        ovl_fname = "%s/ovl.pdf" % results_dir
        fig, ax = plt.subplots()
        x_indices = range(len(overlaps_list))
        num_selected_features = np.sum(overlaps_list)
        w = 1.0
        ax.bar(x_indices, [float(x)/num_selected_features for x in overlaps_list],
                width=w, color=blue_color, edgecolor="none") 
        ax.set_xticks([x + w/2 for x in x_indices])
        ax.set_xticklabels(['%s' % (x+1) for x in x_indices])
        ax.set_title('Fraction of selected features selected exactly k times')
        plt.savefig(ovl_fname, bbox_inches='tight')


    def get_selected_genes(self, min_number_folds, aces_data):
        """ Get the genes that have been selected in a given number of folds.
        
        Parameters
        ----------
        min_number_folds: int
            Number of folds in which a feature must appear to be selected.

        aces_data: datatypes.ExpressionDataset.ExpressionDataset
            Data in ACES format, read using HDF5GroupToExpression_dataset.

        Output
        ------
        selected_features_list: list
            indices of selected features.
        
        selected_genes_dict: dict
            keys: selected genes
            values: number of selected edges that gene belongs to;
                    0 if nodes-based selection.
        """
        ### Create list of indices of features that are selected.
        selected_features_dict = {} # feat_idx:number_of_times_selected
        for feature_set in self.features_list:
            for feat_idx in feature_set:
                if not selected_features_dict.has_key(feat_idx):
                    selected_features_dict[feat_idx] = 1
                else:
                    selected_features_dict[feat_idx] += 1

        selected_features_list = []

        for feat_idx, number_of_times_selected in selected_features_dict.iteritems():
            try:
                assert number_of_times_selected <= len(self.features_list)
            except AssertionError:
                print feat_idx, number_of_times_selected
                sys.stderr.write("Error in computing the number of times a feature was selected.\n")
                sys.stderr.write("Aborting.\n")
                sys.exit(-1)
            if number_of_times_selected >= min_number_folds:
                selected_features_list.append(feat_idx)
        selected_features_list.sort()

        print "%d features in the final selection." % len(selected_features_list)
        
        ### Map features to gene names
        if self.use_nodes:
            ## Features are genes
            aces_gene_names = aces_data.geneLabels
            selected_genes_dict = {aces_gene_names[ix]:0 for ix in selected_features_list}
        else:
            ## Features are edges
            selected_genes_dict = {}
            edges_f = '%s/edges_entrez.gz' % self.network_root
            edges_entrez_list = []
            with gzip.open(edges_f) as f:
                edges_entrez_list = [[line.split()[0], line.split()[1]] for line in f.readlines()]
                f.close()
            edges_entrez_list = [edges_entrez_list[i] for i in selected_features_list]
            for edge in edges_entrez_list:
                for g in edge:
                    if not selected_genes_dict.has_key(g):
                        selected_genes_dict[g] = 1
                    else:
                        selected_genes_dict[g] += 1
            print "\t This corresponds to %d distinct genes" % len(selected_genes_dict.keys())
        return selected_features_list, selected_genes_dict


    def final_cv_score(self, selected_features_list, aces_data):
        """ Get the cross-validated predictivity of the selected features
        
        Parameters
        ----------
        selected_features_list: list
            indices of selected features.

        aces_data: datatypes.ExpressionDataset.ExpressionDataset
            Data in ACES format, read using HDF5GroupToExpression_dataset.

        Output
        ------
        cv_scores: float
            cross-validated AUCs of a logistic regression using only
            the selected features.
        
        # cv_score_ridge: float
        #     cross-validated AUC of a rige-regularized logistic regression
        #     using only the selected features.            
        """
        # Get data
        y_data = aces_data.patientClassLabels
        if self.use_nodes:
            print "Using node weights as features"
            x_data = aces_data.expressionData[:, selected_features_list]
        else:
            print "Using edge weights as features"
            x_f = '%s/%s/edge_weights.gz' % (self.network_root, self.network_type)
            x_data = np.loadtxt(x_f).transpose()[:, selected_features_list]
        
        # Initialize logistic regression cross-validation classifier
        cv_clf = linear_model.LogisticRegression(C=1e6, class_weight='balanced')
        
        # 10-fold cross-validation
        cv_folds = model_selection.KFold(n_splits=10).split(y_data)
        pred = np.zeros(y_data.shape)
        for tr, te in cv_folds:
            Xtr = x_data[tr, :]
            ytr = y_data[tr]
            Xte = x_data[te, :]

            # Fit classifier
            cv_clf.fit(Xtr, ytr)

            # Predict probabilities (of belonging to +1 class) on test data
            yte_pred = cv_clf.predict_proba(Xte)
            pred[te] = yte_pred[:, np.nonzero(cv_clf.classes_ == 1)[0][0]]
            
        return metrics.roc_auc_score(y_data, pred)

        
        
    def final_analysis(self, results_dir, min_number_folds):
        """ Create results files.

        Parameter
        ---------
        results_dir: path
            Where to save results.

        Created files
        -------------
        <results_dir>/final_selection_genes.txt
            list of names of selected genes + number of edges they belong to
        <results_dir>/final_selection_results.txt
            - cross-validated predictivity (ridge regression) of selected features
        <results_dir>/final_selection_enrichment.txt
            - BINGO enrichment results.
        """
        # Read ACES data
        sys.path.append(self.aces_data_root)
        from datatypes.ExpressionDataset import HDF5GroupToExpressionDataset
        f = h5py.File("%s/experiments/data/U133A_combat.h5" % self.aces_data_root)
        aces_data = HDF5GroupToExpressionDataset(f['U133A_combat_RFS'],
                                                 checkNormalise=False)
        f.close()
        
        # Get final selection
        selected_features_list, selected_genes_dict = self.get_selected_genes(min_number_folds, aces_data) 
        sel_fname = '%s/final_selection_genes.txt' % results_dir
        with open(sel_fname, 'w') as f:
            for g, v in selected_genes_dict.iteritems():
                f.write("%s %d\n" % (g, v))
            f.close()

        # Cross-validated predictivity of selected features
        cv_scores = self.final_cv_score(selected_features_list, aces_data)
        res_fname = '%s/final_selection_results.txt' % results_dir
        with open(res_fname, 'w') as f:
            # Write
            f.write("Number of features used:\t%d\n" % len(selected_features_list))
            
            # Write AUC
            f.write("Logistic regression AUC:\t%.2f\n" % cv_scores)
                                                                     
            # f.write("Ridge logistic regression AUC:\t%.2f\n" % cv_score_ridge)
            f.close()

        
def main():
    """ Perform a cross-validation experiment on sample-specific co-expression networks
    in the case where inner cross-validation results are available.
    
    Example
    -------
        $ python OuterCrossVal.py ../ACES ../outputs/U133A_combat_RFS \
         ../outputs/U133A_combat_RFS/subtype_stratified/repeat0  \
         regline -o 10 -k 5 -m 1000
    
    Files created
    -------------
    <results_dir>/results.txt
        - number of selected features per fold
        - final AUC
        - pairwise Fisher overlaps between sets of selected features
        - pairwise consistencies between sets of selected features
        - pairwise Pearson consistencies between sets of selected features
    """
    parser = argparse.ArgumentParser(description="Cross-validate sample-specific co-expression networks",
                                     add_help=True)
    parser.add_argument("aces_data_path", help="Folder containing the ACES data")
    parser.add_argument("network_path", help="Folder containing network skeleton and weights")
    parser.add_argument("innercv_path", help="Folder containing the inner cross-validation results")
    parser.add_argument("network_type", help="Type of co-expression network")
    parser.add_argument("-o", "--num_outer_folds", help="Number of outer cross-validation folds",
                        type=int)
    parser.add_argument("-k", "--num_inner_folds", help="Number of inner cross-validation folds",
                        type=int)
    parser.add_argument("-m", "--max_nr_feats", help="Maximum number of selected features",
                        type=int)
    parser.add_argument("-n", "--nodes", action='store_true', default=False,
                        help="Work with node weights rather than edge weights")
    parser.add_argument("-s", "--sfan",
                        help='Path to sfan code (then automatically use sfan + l2 logistic regression)')
    parser.add_argument("-e", "--enet", action='store_true', default=False,
                        help="Run elastic net instead of lasso.")
    args = parser.parse_args()

    try:
        assert args.network_type in network_types
    except AssertionError:
        sys.stderr.write("network_type should be one of ")
        sys.stderr.write(",".join([" '%s'" % nt for nt in network_types]))
        sys.stderr.write(".\n")
        sys.stderr.write("Aborting.\n")
        sys.exit(-1)

    # Get the total number of samples
    num_samples = 0
    for fold_nr in range(args.num_outer_folds):
        with open('%s/fold%d/test.indices' % (args.innercv_path, fold_nr)) as f:
            num_samples += len(f.readlines())
            f.close()
    print "%d samples" % num_samples

    if args.sfan:
        # ========= Sfan =========
        # Initialize OuterCrossVal
        ocv = OuterCrossVal(args.aces_data_path, args.network_path, args.innercv_path, 
                            args.network_type, num_samples,
                            args.num_inner_folds, args.num_outer_folds, 
                            max_nr_feats=args.max_nr_feats,
                            use_nodes=True, use_sfan=True, sfan_path=args.sfan)
        
        # Baseline using only connected features
        # Read outputs from inner cross-validation experiments 
        ocv.read_inner_results('nosel')

        # Write results
        results_dir = '%s/repeat0/results/nodes/sfan/nosel' % args.innercv_path
        # Create results dir if it does not exist
        if not os.path.isdir(results_dir):
            sys.stdout.write("Creating %s\n" % results_dir)
            try: 
                os.makedirs(results_dir)
            except OSError:
                if not os.path.isdir(results_dir):
                    raise
        
        ocv.write_results(results_dir)

        #  Use sfan to select features
        # Read outputs from inner cross-validation experiments
        ocv.read_inner_results()

        # Write results
        results_dir = '%s/results/nodes/sfan' % args.innercv_path
        # Create results dir if it does not exist
        if not os.path.isdir(results_dir):
            sys.stdout.write("Creating %s\n" % results_dir)
            try: 
                os.makedirs(results_dir)
            except OSError:
                if not os.path.isdir(results_dir):
                    raise
        ocv.write_results(results_dir)
        # ========= End sfan =========

    else:
        # Initialize OuterCrossVal
        ocv = OuterCrossVal(args.aces_data_path, args.network_path, args.innercv_path, 
                            args.network_type, num_samples,
                            args.num_inner_folds, args.num_outer_folds, 
                            max_nr_feats=args.max_nr_feats,
                            use_nodes=args.nodes, use_enet=args.enet)
        
        # ========= l1-regularized logistic regression =========\
        if not args.enet:
            # Read outputs from inner cross-validation experiments
            ocv.read_inner_results()

            # Write results
            if args.nodes:
                results_dir = '%s/results/nodes' % args.innercv_path
            else:
                results_dir = '%s/results/%s' % (args.innercv_path, args.network_type)
            # Create results dir if it does not exist
            if not os.path.isdir(results_dir):
                sys.stdout.write("Creating %s\n" % results_dir)
                try: 
                    os.makedirs(results_dir)
                except OSError:
                    if not os.path.isdir(results_dir):
                        raise
            ocv.write_results(results_dir)
        # ========= End l1-regularized logistic regression =========

        
        # ========= l1/l2-regularized logistic regression =========
        else:
            # Read outputs from inner cross-validation experiments
            ocv.read_inner_results()

            # Write results
            if args.nodes:
                results_dir = '%s/results/nodes/enet' % args.innercv_path
            else:
                results_dir = '%s/results/%s/enet' % (args.innercv_path, args.network_type)
            # Create results dir if it does not exist
            if not os.path.isdir(results_dir):
                sys.stdout.write("Creating %s\n" % results_dir)
                try: 
                    os.makedirs(results_dir)
                except OSError:
                    if not os.path.isdir(results_dir):
                        raise
            ocv.write_results(results_dir)
        # ========= End l1/l2-regularized logistic regression =========

    

if __name__ == "__main__":
    main()
        
