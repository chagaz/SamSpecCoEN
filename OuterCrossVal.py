# @Author 
# Chloe-Agathe Azencott
# chloe-agathe.azencott@mines-paristech.fr
# January 2016

import argparse
import h5py
# import matplotlib.pyplot as plt
# import memory_profiler # call program with flag -m memory_profiler
import numpy as np
import os
import sys
# import timeit

# sys.path.append('ACES')
# from datatypes.ExpressionDataset import HDF5GroupToExpressionDataset, MakeRandomFoldMap

import InnerCrossVal

# orange_color = '#d66000'
# blue_color = '#005599'

class OuterCrossVal(object):
    """ Manage the outer cross-validation loop for learning on sample-specific co-expression networks.

    Attributes
    ----------
    self.nrInnerFolds: int
        Number of folds for the inner cross-validation loop.
    self.nrOuterFolds: int
        Number of folds for the outer cross-validation loop.
    self.maxNrFeats: int
        Maximum number of features to return.
        Default value=400, as in [Staiger et al.]
    self.trueLabels: (numSamples, ) array
        True labels for all samples.
    self.predLabels: (numSamples, ) array
        Predicted labels for all samples, in the same order as trueLabels.
    self.featuresList: list of list
        List of list of indices of the selected features.

    Optional attributes
    -------------------

    Reference
    ---------
    Staiger, C., Cadot, S., Györffy, B., Wessels, L.F.A., and Klau, G.W. (2013).
    Current composite-feature classification methods do not outperform simple single-genes
    classifiers in breast cancer prognosis. Front Genet 4.  
    """
    def __init__(self, dataRoot, networkType, numSamples, nrInnerFolds, nrOuterFolds, maxNrFeats=400):
        """
        Parameters
        ----------
        dataRoot: path
            Path to folder containing data for all folds.
        networkType: string
            Type of network to work with
            Correspond to a folder in dataFoldRoot
            Possible value: 'lioness', 'linreg'
        nrInnerFolds: int
            Number of folds for the inner cross-validation loop.
        maxNrFeats: int
            Maximum number of features to return.
            Default value=400.
        """
        self.trueLabels = np.ones((numSamples, ))
        self.predLabels = np.ones((numSamples, ))
        self.featuresList = []
        

    def runOuterLasso(self):
        """ Run the outer loop of the experiment, for the Lasso algorithm

        Parameters
        ----------


        
        Updated attributes
        ------------------
        trueLabels: (numSamples, ) array
            True labels for all samples.
        predLabels: (numSamples, ) array
            Predicted labels for all samples, in the same order as trueLabels.
        featuresList: list of list
            List of list of indices of the selected features.
        """
        for fold in range(self.nrOuterFolds):
            sys.stdout.write("Working on fold number %d\n" % fold)

            # Read the test indices
            teIndices = np.loadtxt('%s/test.indices' % dataFoldRoot, dtype='int')
            
            # Create an InnerCrossVal
            dataFoldRoot = '%s/%d' % (self.dataRoot, fold)            
            icv = InnerCrossVal.InnerCrossVal(dataFoldRoot, self.networkType,
                                              self.nrInnerFolds, self.maxNrFeats)

            # Get predictions and selected features for the inner loop
            [predLabelsFold, featuresFold] = icv.runInnerLasso()

            # Update self.trueLabels, self.predLabels, self.featuresList
            self.trueLabels[teIndices] = icv.Yte
            self.predLabels[teIndices] = predLabelsFold
            self.featuresList.append(featuresFold)


    def computeAUC(self):
        """ Compute the AUC of the experiment.

        Parameters
        ----------

        Returns
        -------
        auc: float
           Area under the ROC curve for the experiment.
        """


    def computeFisherOverlap(self):
        """ Compute the Fisher overlap between the sets of selected features.

        Parameters
        ----------

        Returns
        -------
        fov: float
           Fisher overlap between the sets of selected features for the experiment.
        """


def main():
    """ Run a cross-validation experiment on sample-specific co-expression networks.

    Example:
        $ python OuterCrossVal.py outputs/U133A_combat_RFS lioness -o 5 -k 5 -m 400
    """
    parser = argparse.ArgumentParser(description="Cross-validate sample-specific co-expression networks",
                                     add_help=True)
    parser.add_argument("data_path", help="Path to the folder containing the data")
    parser.add_argument("network_type", help="Type of co-expression networks")
    parser.add_argument("-o", "--num_outer_folds", help="Number of outer cross-validation folds",
                        type=int)
    parser.add_argument("-k", "--num_inner_folds", help="Number of inner cross-validation folds",
                        type=int)
    parser.add_argument("-m", "--max_nr_feats", help="Maximum number of selected features",
                        type=int)
    args = parser.parse_args()

    try:
        assert args.network_type in ['lioness', 'linreg']
    except AssertionError:
        sys.stderr.write("network_type should be one of 'lioness', 'linreg'.\n")
        sys.stderr.write("Aborting.\n")
        sys.exit(-1)

    # Initialize OuterCrossVal
    ocv = OuterCrossVal(args.data_path, args.network_type, numSamples,
                        args.num_inner_folds, args.num_outer_folds, max_nr_feats)
    # Run the experiment
    ocv.runOuterLasso()

    # Get the AUC
    print "AUC:", ocv.computeAUC()

    # Get the stability (Fisher overlap)
    print "Stability (Fisher overlap):", ocv.computeFisherOverlap()


if __name__ == "__main__":
    main()
        