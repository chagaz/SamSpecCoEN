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

# import utils

# orange_color = '#d66000'
# blue_color = '#005599'

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
    Staiger, C., Cadot, S., Györffy, B., Wessels, L.F.A., and Klau, G.W. (2013).
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
            Possible value: 'lioness', 'linreg'
        """
        try:
            assert networkType in ['lioness', 'linreg']
        except AssertionError:
            sys.stderr.write("networkType should be one of 'lioness', 'linerg'.\n")
            sys.stderr.write("Aborting.\n")
            sys.exit(-1)
    
        self.Xtr = np.loadtxt('%s/%s/edge_weights.gz' % (dataFoldRoot, networkType))
        self.Xte = np.loadtxt('%s/%s/edge_weights_te.gz' % (dataFoldRoot, networkType))
        
        self.Ytr = np.loadtxt('%s/train.labels' % dataFoldRoot, dtype='int')
        self.Yte = np.loadtxt('%s/test.labels' % dataFoldRoot, dtype='int')

        self.nrFolds = nrFolds
        self.maxNrFeats = maxNrFeats
        

    def runInnerLasso(self):
        """ Run the inner loop, using the Lasso algorithm.
        
        Returns
        -------
        predLabels: (numTestSamples, ) array
            Predicted labels for test samples, in the same order as self.Ytr.
        features: list
            List of indices of the selected features.
        """
        # Get the optimal value of the lambda parameter by inner cross-validation
        bestLambda = self.cvInnerLasso()

        # Return the predictions and selected features
        return self.trainPredInnerLasso(bestLambda)     

        
    def cvInnerLasso(self, lambdaRange=[10.**k for k in range(-3, 3)]):
        """ Compute the inner cross-validation loop to determine the best lambda parameter for Lasso.
        
        Parameters
        ----------
        lambdaRange: list
            Range of lambda values to try out.
            Default: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

        Returns
        -------
        bestLambda: float
            Optimal value of the lambda parameter.
        """

        
    def trainPredInnerLasso(self, bestLambda):
        """ Train Lasso (with optimal parameter) on the train set, predict on the test set.
        
        Parameters
        ----------
        bestLambda: float
            Optimal value of the lambda parameter.

        Returns
        -------
        predLabels: (numTestSamples, ) array
            Predicted labels for test samples, in the same order as trueLabels.
        features: list
            List of indices of the selected features.
        """        