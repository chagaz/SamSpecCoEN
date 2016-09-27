# @Author 
# Chloe-Agathe Azencott
# chloe-agathe.azencott@mines-paristech.fr
# April 2016

import argparse
import h5py

import matplotlib # in a non-interactive environment
matplotlib.use('Agg') # in a non-interactive environment
import matplotlib.pyplot as plt

import numpy as np
import os
import scipy.stats as st
import sys

from sklearn import metrics as skm
orange_color = '#d66000'
blue_color = '#005599'

import InnerCrossVal


class OuterCrossVal(object):
    """ Manage the outer cross-validation loop for learning on sample-specific co-expression networks.

    Attributes
    ----------
    self.aces_data_root: path
            Path to folder containing ACES data
    self.network_data_root: path
        Path to folder containing data for all folds.
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
    def __init__(self, aces_data_root, network_data_root, network_type, num_samples,
                 nr_inner_folds, nr_outer_folds, max_nr_feats=400, use_nodes=False,
                 use_sfan=False, sfan_path=None):
        """
        Parameters
        ----------
        aces_data_root: path
            Path to folder containing ACES data
        network_data_root: path
            Path to folder containing data for all folds.
        network_type: string
            Type of network to work with
            Correspond to a folder in dataFoldRoot
            Possible value: 'lioness', 'regline'
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
        self.network_data_root = network_data_root
        self.network_type = network_type

        self.use_nodes = use_nodes
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
            self.num_features = np.loadtxt("%s/fold0/edges.gz" % self.network_data_root).shape[0]
        
        
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
            data_fold_root = '%s/fold%d' % (self.network_data_root, fold)
            te_indices = np.loadtxt('%s/test.indices' % data_fold_root, dtype='int')
            
            # Create an InnerCrossVal
            icv = InnerCrossVal.InnerCrossVal(self.aces_data_path, data_fold_root,
                                              self.network_type, self.nr_inner_folds,
                                              self.max_nr_feats, self.use_nodes)
                                              

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
            data_fold_root = '%s/fold%d' % (self.network_data_root, fold)
            te_indices = np.loadtxt('%s/test.indices' % data_fold_root, dtype='int')
            
            # Create an InnerCrossVal
            icv = InnerCrossVal.InnerCrossVal(self.aces_data_path, data_fold_root,
                                              self.network_type, self.nr_inner_folds,
                                              self.max_nr_feats, self.use_nodes)
                                              

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
            data_fold_root = '%s/fold%d' % (self.network_data_root, fold)
            te_indices = np.loadtxt('%s/test.indices' % data_fold_root, dtype='int')
            
            # Create an InnerCrossVal
            icv = InnerCrossVal.InnerCrossVal(self.aces_data_path, data_fold_root,
                                              self.network_type, self.nr_inner_folds,
                                              self.max_nr_feats, self.use_nodes)
                                              

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


    def read_inner_results(self, inner_cv_resdir):
        """ Read the results of the inner loop of the experiment.

        Parameters
        ----------
        inner_cv_resdir: path
            Path to outputs of InnerCrossVal for each fold

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
            sys.stdout.write("Reading results for fold number %d\n" % fold)

            # Read the test indices
            data_fold_root = '%s/fold%d' % (self.network_data_root, fold)
            te_indices = np.loadtxt('%s/test.indices' % data_fold_root, dtype='int')
            
            # Read results from InnerCrossVal
            yte_fname = '%s/fold%d/yte' % (inner_cv_resdir, fold)
            self.true_labels[te_indices] = np.loadtxt(yte_fname, dtype='int')

            pred_values_fname = '%s/fold%d/predValues' % (inner_cv_resdir, fold)
            self.pred_values[te_indices] = np.loadtxt(pred_values_fname)

            features_list_fname = '%s/fold%d/featuresList' % (inner_cv_resdir, fold)
            self.features_list.append(np.loadtxt(features_list_fname, dtype='int'))

        # Convert probability estimates in labels
        self.pred_labels = np.array(self.pred_values > 0, dtype='int')


    def read_inner_results_subdir(self, inner_cv_resdir, subdir_name):
        """ Read the results of the inner loop of the experiment.

        Parameters
        ----------
        inner_cv_resdir: path
            Path to outputs of InnerCrossVal for each fold.
        subdir_name: folder name
            Name of the subdirectory of inner_cv_resdir/fold<fold_nr> 
            in which to find results.

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
            sys.stdout.write("Reading results for fold number %d\n" % fold)

            # Read the test indices
            data_fold_root = '%s/fold%d' % (self.network_data_root, fold)
            te_indices = np.loadtxt('%s/test.indices' % data_fold_root, dtype='int')
            
            # Read results from InnerCrossVal
            yte_fname = '%s/fold%d/%s/yte' % (inner_cv_resdir, fold, subdir_name)
            self.true_labels[te_indices] = np.loadtxt(yte_fname, dtype='int')

            pred_values_fname = '%s/fold%d/%s/predValues' % (inner_cv_resdir, fold, subdir_name)
            self.pred_values[te_indices] = np.loadtxt(pred_values_fname)

            features_list_fname = '%s/fold%d/%s/featuresList' % (inner_cv_resdir, fold, subdir_name)
            self.features_list.append(np.loadtxt(features_list_fname, dtype='int'))

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
        return skm.roc_auc_score(self.true_labels, self.pred_values)
        

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
                fov_list.append(-np.log10(st.fisher_exact(contingency, alternative='greater')[1]))
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
            AUC
            Fisher overlaps (per fold)
            Consistency index (per fold).

        fov.pdf:
            Box plot of Fisher overlaps.

        cix.pdf:
            Box plot of consistencies.        
        """

        # Open results file for writing
        res_fname = '%s/results.txt' % results_dir
        with open(res_fname, 'w') as f:
            # Write number of selected features
            f.write("Number of features selected per fold:\t")
            f.write("%s\n" % " ".join(["%d" % len(x) for x in self.features_list]))

            # Write AUC
            f.write("AUC:\t%.2f\n" % self.compute_auc())

            # Write the stability (Fisher overlap)
            fov_list = self.compute_fisher_overlap()
            f.write("Stability (Fisher overlap):\t")
            f.write("%s\n" % ["%.2e" % x for x in fov_list])

            # Write the stability (consistency index)
            cix_list = self.compute_consistency()
            f.write("Stability (Consistency Index):\t")
            f.write("%s\n" % ["%.2e" % x for x in cix_list])
            f.close()

        # Plot the stability (Fisher overlap)
        fov_fname = '%s/fov.pdf' % results_dir
        plt.figure()
        plt.boxplot(fov_list, 0, 'gD')
        plt.title('Fisher overlap')
        plt.ylabel('-log10(p-value)')
        plt.savefig(fov_fname, bbox_inches='tight')

        # Plot the stability (consistency index)
        cix_fname = '%s/cix.pdf' % results_dir
        plt.figure()
        plt.boxplot(cix_list, 0, 'gD')
        plt.title('Consistency Index')
        plt.savefig(cix_fname, bbox_inches='tight')

        
def main():
    """ Run a cross-validation experiment on sample-specific co-expression networks.

    Example
    -------
        $ python OuterCrossVal.py ACES/ outputs/U133A_combat_DMFS lioness results/U133A_combat_DMFS/lioness -o 5 -k 5 -m 1000

    Files created
    -------------
    <results_dir>/results.txt
        - number of selected features per fold
        - final AUC
        - pairwise Fisher overlaps between sets of selected features
        - pairwise consistencies between sets of selected features
    """
    parser = argparse.ArgumentParser(description="Cross-validate sample-specific co-expression networks",
                                     add_help=True)
    parser.add_argument("aces_data_path", help="Path to the folder containing the ACES data")
    parser.add_argument("network_data_path", help="Path to the folder containing the network data")
    parser.add_argument("network_type", help="Type of co-expression networks")
    parser.add_argument("results_dir", help="Folder where to store results")
    parser.add_argument("-o", "--num_outer_folds", help="Number of outer cross-validation folds",
                        type=int)
    parser.add_argument("-k", "--num_inner_folds", help="Number of inner cross-validation folds",
                        type=int)
    parser.add_argument("-m", "--max_nr_feats", help="Maximum number of selected features",
                        type=int)
    parser.add_argument("-n", "--nodes", action='store_true', default=False,
                        help="Work with node weights rather than edge weights")
    parser.add_argument("-e", "--enet", action='store_true', default=False,
                        help="Only run elastic net")
    parser.add_argument("-s", "--sfan",
                        help='Path to sfan code (then automatically use sfan + l2 logistic regression)')
    args = parser.parse_args()

    try:
        assert args.network_type in ['lioness', 'regline']
    except AssertionError:
        sys.stderr.write("network_type should be one of 'lioness', 'regline'.\n")
        sys.stderr.write("Aborting.\n")
        sys.exit(-1)

    # Get the total number of samples
    num_samples = 0
    for fold_nr in range(args.num_outer_folds):
        with open('%s/fold%d/test.indices' % (args.netowrk_data_path, fold_nr)) as f:
            num_samples += len(f.readlines())
            f.close()

    # Create results dir if it does not exist
    if not os.path.isdir(args.results_dir):
        sys.stdout.write("Creating %s\n" % args.results_dir)
        try: 
            os.makedirs(args.results_dir)
        except OSError:
            if not os.path.isdir(args.results_dir):
                raise

    if args.sfan:
        # ========= Sfan =========
        # Initialize OuterCrossVal
        ocv = OuterCrossVal.OuterCrossVal(args.aces_data_path, args.network_data_path, 
                                          args.network_type, num_samples,
                                          args.num_inner_folds, args.num_outer_folds, 
                                          max_nr_feats=args.max_nr_feats,
                                          use_nodes=True, use_sfan=True, sfan_path=args.sfan)

        # Baseline using only connected features
        # Run the experiment
        ocv.run_outer_l2_logreg()

        # Create results dir if it does not exist
        results_dir = '%s/nosel' % args.results_dir
        if not os.path.isdir(results_dir):
            sys.stdout.write("Creating %s\n" % results_dir)
            try: 
                os.makedirs(results_dir)
            except OSError:
                if not os.path.isdir(results_dir):
                    raise

        # Write results
        ocv.write_results(results_dir)

        # Use sfan to select features
        # Run the experiment
        ocv.run_outer_sfan()

        print "Number of features:\t", [len(x) for x in ocv.features_list]
        print "AUC:\t", ocv.compute_auc()

        # Write results
        ocv.write_results(args.results_dir)
        # ========= End sfan =========

    else:
        # Initialize OuterCrossVal
        ocv = OuterCrossVal(args.aces_data_path, args.network_data_path, args.network_type, num_samples,
                            args.num_inner_folds, args.num_outer_folds, args.max_nr_feats, args.nodes)

        # ========= l1 regularization =========
        if not args.enet:
            # Run the experiment
            ocv.run_outer_l1_logreg()

            print "Number of features:\t", [len(x) for x in ocv.features_list]
            print "AUC:\t", ocv.compute_auc()

            ocv.write_results(args.results_dir)
        # ========= End l1 regularization =========


        # ========= l1/l2 regularization =========
        else:
            # Run the experiment
            ocv.run_outer_enet_logreg()

            # Create results dir if it does not exist
            results_dir = '%s/enet' % args.results_dir
            if not os.path.isdir(results_dir):
                sys.stdout.write("Creating %s\n" % results_dir)
                try: 
                    os.makedirs(results_dir)
                except OSError:
                    if not os.path.isdir(results_dir):
                        raise

            print "Number of features:\t", [len(x) for x in ocv.features_list]
            print "AUC:\t", ocv.compute_auc()

            ocv.write_results(results_dir)
        # ========= End l1/l2 regularization =========

    
    

if __name__ == "__main__":
    main()
        
