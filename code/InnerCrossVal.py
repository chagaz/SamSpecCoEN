# @Author 
# Chloe-Agathe Azencott
# chloe-agathe.azencott@mines-paristech.fr
# April 2016

import argparse
import h5py
import numpy as np
import os
import sys

from sklearn import linear_model as sklm 
from sklearn import  metrics as skm

class InnerCrossVal(object):
    """ Manage the inner cross-validation loop for learning on sample-specific co-expression networks.

    Attributes
    ----------
    self.x_tr: (numEdges, numTrainingSamples) array
        Edge weights for the samples of the training data.
    self.x_te: (numEdges, num TestingSamples) array
        Edge weights for the samples of the test data.
    self.y_tr: (numTrainingSamples, ) array
        Labels (0/1) of the training samples.
    self.y_te: (numTestSamples, ) array
        Labels (0/1) of the test samples.
    self.nr_folds: int
        Number of folds for the inner cross-validation loop.
    self.max_nr_feats: int
        Maximum number of features to return.
        Default value=400, as in [Staiger et al.]

    Optional attributes
    -------------------

    Reference
    ---------
    Staiger, C., Cadot, S., Gyoerffy, B., Wessels, L.F.A., and Klau, G.W. (2013).
    Current composite-feature classification methods do not outperform simple single-genes
    classifiers in breast cancer prognosis. Front Genet 4.  
    """
    def __init__(self, aces_data_root, data_fold_root, network_type, nr_folds,
                 max_nr_feats=400, use_nodes=False):
        """
        Parameters
        ----------
        aces_data_root: path
            Path to folder containing ACES data
        data_fold_root: path
            Path to folder containing data for the fold.
        network_type: string
            Type of network to work with
            Correspond to a folder in data_fold_root
            Possible value: 'lioness', 'regline'
        nr_folds: int
            Number of (inner) cross-validation folds.
        max_nr_feats: int
            Maximum number of features to return.
        use_nodes: bool
            Whether to use node weights rather than edge weights as features.
           (This does not make use of the network information.)
        """
        try:
            assert network_type in ['lioness', 'regline']
        except AssertionError:
            sys.stderr.write("network_type should be one of 'lioness', 'regline'.\n")
            sys.stderr.write("Aborting.\n")
            sys.exit(-1)

        if use_nodes:
            print "Using node weights as features"
            # Read ACES data
            sys.path.append(aces_data_root)
            from datatypes.ExpressionDataset import HDF5GroupToExpressionDataset
            f = h5py.File("%s/experiments/data/U133A_combat.h5" % aces_data_root)
            aces_data = HDF5GroupToExpressionDataset(f['U133A_combat_RFS'], checkNormalise=False)
            f.close()

            # Get train/test indices for fold
            tr_indices = np.loadtxt('%s/train.indices' % data_fold_root, dtype='int')
            te_indices = np.loadtxt('%s/test.indices' % data_fold_root, dtype='int')

            # Get Xtr, Xte
            self.x_tr = aces_data.expressionData[tr_indices, :]
            self.x_te = aces_data.expressionData[te_indices, :]
        else:
            print "Using edge weights as features"
            x_tr_f = '%s/%s/edge_weights.gz' % (data_fold_root, network_type)
            self.x_tr = np.loadtxt(x_tr_f).transpose()

            x_te_f = '%s/%s/edge_weights_te.gz' % (data_fold_root, network_type)
            self.x_te = np.loadtxt(x_te_f).transpose()

        # Normalize data (according to training data)
        x_mean = np.mean(self.x_tr, axis=0)
        x_stdv = np.std(self.x_tr, axis=0, ddof=1)

        self.x_tr = (self.x_tr - x_mean) / x_stdv
        self.x_te = (self.x_te - x_mean) / x_stdv

        # Labels
        self.y_tr = np.loadtxt('%s/train.labels' % data_fold_root, dtype='int')
        self.y_te = np.loadtxt('%s/test.labels' % data_fold_root, dtype='int')

        self.nr_folds = nr_folds
        self.max_nr_feats = max_nr_feats
        

    def run_inner_l1_logreg(self, reg_params=[10.**k for k in range(-3, 3)]):
        """ Run the inner loop, using an l1-regularized logistic regression.
        
        Parameters
        ----------
        reg_params: list
            Range of lambda values to try out.
            Default: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        
        Returns
        -------
        pred_values: (numTestSamples, ) array
            Probability estimates for test samples, in the same order as self.y_tr.
        features: list
            List of indices of the selected features.
        """
        # Get the optimal value of the regularization parameter by inner cross-validation
        best_reg_param = self.cv_inner_l1_logreg(reg_params)

        # Return the predictions and selected features
        return self.train_pred_inner_l1_logreg(best_reg_param)     

        
    def run_inner_l1_logreg_write(self, resdir, reg_params=[10.**k for k in range(-3, 3)], ):
        """ Run the inner loop, using an l1-regularized logistic regression.
        Save outputs to files.
        
        Parameters
        ----------
        resdir: path
            Path to dir where to save outputs
        reg_params: list
            Range of lambda values to try out.
            Default: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        
        Returns
        -------
        pred_values: (num_test_samples, ) array
            Probability estimates for test samples, in the same order as self.y_tr.
        features: list
            List of indices of the selected features.

        Write files
        -----------
        yte:
            Contains self.y_te
        pred_values:
            Contains predictions
        featuresList:
            Contains selected features
        """
        # Get the optimal value of the regularization parameter by inner cross-validation
        best_reg_param = self.cv_inner_l1_logreg(reg_params)

        # Get the predictions and selected features
        [pred_values, features_list] = self.train_pred_inner_l1_logreg(best_reg_param)

        # Save to files
        yte_fname = '%s/yte' % resdir
        np.savetxt(yte_fname, self.y_te, fmt='%d')
        
        pred_values_fname = '%s/predValues' % resdir
        np.savetxt(pred_values_fname, pred_values)
        
        features_list_fname = '%s/featuresList' % resdir
        np.savetxt(features_list_fname, features_list, fmt='%d')            
        

        
    def cv_inner_l1_logreg(self, reg_params=[10.**k for k in range(-3, 3)]):
        """ Compute the inner cross-validation loop to determine the best regularization parameter
        for an l1-regularized logistic regression.
        
        Parameters
        ----------
        reg_params: list
            Range of lambda values to try out.
            Default: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        
        Returns
        -------
        best_reg_param: float
            Optimal value of the regularization parameter.
        """
        # Initialize logistic regression cross-validation classifier
        cv_clf = sklm.LogisticRegressionCV(Cs=reg_params, penalty='l1', solver='liblinear',
                                           cv=self.nr_folds,
                                           class_weight='balanced', scoring='roc_auc')

        # Fit to training data
        cv_clf.fit(self.x_tr, self.y_tr)

        # Quality of fit?
        y_tr_pred = cv_clf.predict_proba(self.x_tr)
        y_tr_pred = y_tr_pred[:, cv_clf.classes_.tolist().index(1)]
        print "\tTraining AUC:\t", skm.roc_auc_score(self.y_tr, y_tr_pred)

        # Get optimal value of the regularization parameter.
        # If there are multiple equivalent values, return the first one.
        # Note: Small C = more regularization.
        best_reg_param = cv_clf.C_[0]
        print "\tall top C:\t", cv_clf.C_
        print "\tbest C:\t", best_reg_param
        return best_reg_param

        
    def train_pred_inner_l1_logreg(self, best_reg_param):
        """ Train an l1-regularized logistic regression (with optimal parameter)
        on the train set, predict on the test set.
        
        Parameters
        ----------
        best_reg_param: float
            Optimal value of the regularization parameter.

        Returns
        -------
        pred_values: (num_test_samples, ) array
            Probability estimates for test samples, in the same order as trueLabels.
        features: list
            List of indices of the selected features.
        """
        # Initialize logistic regression classifier
        classif = sklm.LogisticRegression(C=best_reg_param, penalty='l1', solver='liblinear',
                                          class_weight='balanced')
        
        # Train on the training set
        classif.fit(self.x_tr, self.y_tr)

        # Predict on the test set
        pred_values = classif.predict_proba(self.x_te)

        # Only get the probability estimates for the positive class
        pred_values = pred_values[:, classif.classes_.tolist().index(1)]

        # Quality of fit
        print "\tTest AUC:\t", skm.roc_auc_score(self.y_te, pred_values)

        # Get selected features
        # If there are less than self.max_nr_feats, these are the non-zero coefficients
        features = np.where(classif.coef_[0])[0]
        if len(features) > self.max_nr_feats:
            # Prune the coefficients with lowest values
            features = np.argsort(classif.coef_[0])[-self.max_nr_feats:]

        print "\tNumber of selected features:\t", len(features)

        return pred_values, features


def main():
    """ Run an inner cross-validation on sample-specific co-expression networks.
    Save results to file.

    Example
    -------
        $ python InnerCrossVal.py ACES outputs/U133A_combat_RFS/subtype_stratified/repeat0/fold0 lioness outputs/U133A_combat_RFS/subtype_stratified/repeat0/results/lioness/fold0 -k 5 -m 1000 

    Files created
    -------------
    <results_dir>/yte
        True test labels.

    <results_dir>/predValues
        Predictions for test samples.

    <results_dir>/featuresList
        Selected features.
    """
    parser = argparse.ArgumentParser(description="Inner CV of sample-specific co-expression networks",
                                     add_help=True)
    parser.add_argument("aces_data_path", help="Path to the folder containing the ACES data")
    parser.add_argument("network_data_path", help="Path to the folder containing the network data")
    parser.add_argument("network_type", help="Type of co-expression networks")
    parser.add_argument("results_dir", help="Folder where to store results")
    parser.add_argument("-k", "--num_inner_folds", help="Number of inner cross-validation folds",
                        type=int)
    parser.add_argument("-m", "--max_nr_feats", help="Maximum number of selected features",
                        type=int)
    parser.add_argument("-n", "--nodes", action='store_true', default=False,
                        help="Work with node weights rather than edge weights")
    args = parser.parse_args()

    try:
        assert args.network_type in ['lioness', 'regline']
    except AssertionError:
        sys.stderr.write("network_type should be one of 'lioness', 'regline'.\n")
        sys.stderr.write("Aborting.\n")
        sys.exit(-1)

    # Initialize InnerCrossVal
    icv = InnerCrossVal(args.aces_data_path, args.network_data_path, args.network_type, 
                        args.num_inner_folds, args.max_nr_feats, args.nodes)

    # Create results dir if it does not exist
    if not os.path.isdir(args.results_dir):
        sys.stdout.write("Creating %s\n" % args.results_dir)
        try: 
            os.makedirs(args.results_dir)
        except OSError:
            if not os.path.isdir(args.results_dir):
                raise

    # Run the inner cross-validation
    icv.run_inner_l1_logreg_write(args.results_dir,
                                  reg_params=[2.**k for k in range(-7, -1)])



if __name__ == "__main__":
    main()
                