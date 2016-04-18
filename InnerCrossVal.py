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


class InnerCrossVal(object):
    """ Manage the inner cross-validation loop for learning on sample-specific co-expression networks.

    Attributes
    ----------
    self.Xtr: (numEdges, numTrainingSamples) array
        Edge weights for the samples of the training data.
    self.Xte: (numEdges, num TestingSamples) array
        Edge weights for the samples of the test data.
    self.Ytr: (numTrainingSamples, ) array
        Labels (0/1) of the training samples.
    self.Yte: (numTestSamples, ) array
        Labels (0/1) of the test samples.
    self.nrFolds: int
        Number of folds for the inner cross-validation loop.
    self.maxNrFeats: int
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
    def __init__(self, dataFoldRoot, networkType, nrFolds, maxNrFeats=400):
        """
        Parameters
        ----------
        dataFoldRoot: path
            Path to folder containing data for the fold.
        networkType: string
            Type of network to work with
            Correspond to a folder in dataFoldRoot
            Possible value: 'lioness', 'regline'
        nrFolds: int
            Number of (inner) cross-validation folds.
        maxNrFeats: int
            Maximum number of features to return.
        """
        try:
            assert networkType in ['lioness', 'regline']
        except AssertionError:
            sys.stderr.write("networkType should be one of 'lioness', 'regline'.\n")
            sys.stderr.write("Aborting.\n")
            sys.exit(-1)
    
        self.Xtr = np.loadtxt('%s/%s/edge_weights.gz' % (dataFoldRoot, networkType)).transpose()
        self.Xte = np.loadtxt('%s/%s/edge_weights_te.gz' % (dataFoldRoot, networkType)).transpose()
        
        self.Ytr = np.loadtxt('%s/train.labels' % dataFoldRoot, dtype='int')
        self.Yte = np.loadtxt('%s/test.labels' % dataFoldRoot, dtype='int')

        self.nrFolds = nrFolds
        self.maxNrFeats = maxNrFeats
        

    def runInnerL1LogReg(self, regParams=[10.**k for k in range(-3, 3)]):
        """ Run the inner loop, using an l1-regularized logistic regression.
        
        Parameters
        ----------
        regParams: list
            Range of lambda values to try out.
            Default: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        
        Returns
        -------
        predValues: (numTestSamples, ) array
            Probability estimates for test samples, in the same order as self.Ytr.
        features: list
            List of indices of the selected features.
        """
        # Get the optimal value of the regularization parameter by inner cross-validation
        bestRegParam = self.cvInnerL1LogReg(regParams)

        # Return the predictions and selected features
        return self.trainPredInnerL1LogReg(bestRegParam)     

        
    def runInnerL1LogReg_write(self, resdir, regParams=[10.**k for k in range(-3, 3)], ):
        """ Run the inner loop, using an l1-regularized logistic regression.
        Save outputs to files.
        
        Parameters
        ----------
        resdir: path
            Path to dir where to save outputs
        regParams: list
            Range of lambda values to try out.
            Default: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        
        Returns
        -------
        predValues: (numTestSamples, ) array
            Probability estimates for test samples, in the same order as self.Ytr.
        features: list
            List of indices of the selected features.

        Write files
        -----------
        yte:
            Contains self.Yte
        predValues:
            Contains predictions
        featuresList:
            Contains selected features
        """
        # Get the optimal value of the regularization parameter by inner cross-validation
        bestRegParam = self.cvInnerL1LogReg(regParams)

        # Get the predictions and selected features
        [predValues, featuresList] = self.trainPredInnerL1LogReg(bestRegParam)

        # Save to files
        yte_fname = '%s/yte' % resdir
        np.savetxt(yte_fname, self.Yte, fmt='%d')
        
        predValues_fname = '%s/predValues' % resdir
        np.savetxt(predValues_fname, predValues)
        
        featuresList_fname = '%s/featuresList' % resdir
        np.savetxt(featuresList_fname, featuresList, fmt='%d')            
        

        
    def cvInnerL1LogReg(self, regParams=[10.**k for k in range(-3, 3)]):
        """ Compute the inner cross-validation loop to determine the best regularization parameter
        for an l1-regularized logistic regression.
        
        Parameters
        ----------
        regParams: list
            Range of lambda values to try out.
            Default: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        
        Returns
        -------
        bestRegParam: float
            Optimal value of the regularization parameter.
        """
        # Initialize logistic regression cross-validation classifier
        cvClassif = sklm.LogisticRegressionCV(Cs=regParams, penalty='l1', solver='liblinear',
                                              class_weight='balanced', scoring='roc_auc')

        # Fit to training data
        cvClassif.fit(self.Xtr, self.Ytr)

        # Get optimal value of the regularization parameter.
        # If there are multiple equivalent values, return the first one.
        # Note: Small C = more regularization.
        bestRegParam = cvClassif.C_[0]
        return bestRegParam

        
    def trainPredInnerL1LogReg(self, bestRegParam):
        """ Train an l1-regularized logistic regression (with optimal parameter)
        on the train set, predict on the test set.
        
        Parameters
        ----------
        bestRegParam: float
            Optimal value of the regularization parameter.

        Returns
        -------
        predValues: (numTestSamples, ) array
            Probability estimates for test samples, in the same order as trueLabels.
        features: list
            List of indices of the selected features.
        """
        # Initialize logistic regression classifier
        classif = sklm.LogisticRegression(C=bestRegParam, penalty='l1', solver='liblinear',
                                          class_weight='balanced')
        
        # Train on the training set
        classif.fit(self.Xtr, self.Ytr)

        # Predict on the test set
        predValues = classif.predict_proba(self.Xte)

        # Only get the probability estimates for the positive class
        predValues = predValues[:, classif.classes_.tolist().index(1)]

        # Get selected features
        # If there are less than self.maxNrFeats, these are the non-zero coefficients
        features = np.where(classif.coef_[0])[0]
        if len(features) > self.maxNrFeats:
            # Prune the coefficients with lowest values
            features = np.argsort(classif.coef_[0])[-self.maxNrFeats:]

        return predValues, features


def main():
    """ Run an inner cross-validation on sample-specific co-expression networks.
    Save results to file.

    Example
    -------
        $ python InnerCrossVal.py outputs/U133A_combat_RFS/subtype_stratified/repeat0/fold0 lioness outputs/U133A_combat_RFS/subtype_stratified/repeat0/results/lioness/fold0 -k 5 -m 400 

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
    parser.add_argument("data_path", help="Path to the folder containing the data")
    parser.add_argument("network_type", help="Type of co-expression networks")
    parser.add_argument("results_dir", help="Folder where to store results")
    parser.add_argument("-k", "--num_inner_folds", help="Number of inner cross-validation folds",
                        type=int)
    parser.add_argument("-m", "--max_nr_feats", help="Maximum number of selected features",
                        type=int)
    args = parser.parse_args()

    try:
        assert args.network_type in ['lioness', 'regline']
    except AssertionError:
        sys.stderr.write("network_type should be one of 'lioness', 'regline'.\n")
        sys.stderr.write("Aborting.\n")
        sys.exit(-1)

    # Initialize InnerCrossVal
    icv = InnerCrossVal(args.data_path, args.network_type, 
                        args.num_inner_folds, args.max_nr_feats)

    # Create results dir if it does not exist
    if not os.path.isdir(args.results_dir):
        sys.stdout.write("Creating %s\n" % args.results_dir)
        try: 
            os.makedirs(args.results_dir)
        except OSError:
            if not os.path.isdir(args.results_dir):
                raise

    # Run the inner cross-validation
    icv.runInnerL1LogReg_write(args.results_dir)



if __name__ == "__main__":
    main()
                