# @Author 
# Chloe-Agathe Azencott
# chloe-agathe.azencott@mines-paristech.fr
# January 2016

import argparse
import h5py
import matplotlib.pyplot as plt
# import memory_profiler # call program with flag -m memory_profiler
import numpy as np
import os
import scipy.stats as st
import sys
# import timeit

from sklearn import metrics as skm

# sys.path.append('ACES')
# from datatypes.ExpressionDataset import HDF5GroupToExpressionDataset, MakeRandomFoldMap

import InnerCrossVal

# orange_color = '#d66000'
# blue_color = '#005599'

class OuterCrossVal(object):
    """ Manage the outer cross-validation loop for learning on sample-specific co-expression networks.

    Attributes
    ----------
    self.dataRoot: path
        Path to folder containing data for all folds.
    self.networkType: string
        Type of network to work with
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
    self.predValues: (numSamples, ) array
        Probability estimates for all samples, in the same order as trueLabels
    self.featuresList: list of list
        List of list of indices of the selected features.
    self.numEdges: float
        Total number of features (=edges)

    Reference
    ---------
    Staiger, C., Cadot, S., Gyoerffy, B., Wessels, L.F.A., and Klau, G.W. (2013).
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
        self.predValues = np.ones((numSamples, ))
        self.featuresList = []
        
        self.nrOuterFolds = nrOuterFolds
        self.nrInnerFolds = nrInnerFolds
        self.maxNrFeats = maxNrFeats

        self.dataRoot = dataRoot
        self.networkType = networkType

        # Read the number of edges
        self.numEdges = np.loadtxt("%s/edges.gz" % self.dataRoot).shape[0]
        
        
    def runOuterL1LogReg(self):
        """ Run the outer loop of the experiment, for an l1-regularized logistic regression.

        Updated attributes
        ------------------
        trueLabels: (numSamples, ) array
            True labels for all samples.
        predLabels: (numSamples, ) array
            Predicted labels for all samples, in the same order as trueLabels.
        predValues: (numSamples, ) array
            Probability estimates for all samples, in the same order as trueLabels.
        featuresList: list of list
            List of list of indices of the selected features.
        """
        for fold in range(self.nrOuterFolds):
            sys.stdout.write("Working on fold number %d\n" % fold)

            # Read the test indices
            dataFoldRoot = '%s/%d' % (self.dataRoot, fold)
            teIndices = np.loadtxt('%s/test.indices' % dataFoldRoot, dtype='int')
            
            # Create an InnerCrossVal
            icv = InnerCrossVal.InnerCrossVal(dataFoldRoot, self.networkType,
                                              self.nrInnerFolds, self.maxNrFeats)

            # Get predictions and selected features for the inner loop
            [predValuesFold, featuresFold] = icv.runInnerL1LogReg()

            # Update self.trueLabels, self.predLabels, self.featuresList
            self.trueLabels[teIndices] = icv.Yte
            self.predValues[teIndices] = predValuesFold
            self.featuresList.append(featuresFold)

        # Convert probability estimates in labels
        self.predLabels = np.array(self.predValues > 0, dtype='int')
            

    def computeAUC(self):
        """ Compute the AUC of the experiment.

        Returns
        -------
        auc: float
           Area under the ROC curve for the experiment.
        """
        return skm.roc_auc_score(self.trueLabels, self.predValues)
        

    def computeFisherOverlap(self):
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
        allFeatures = set(range(self.numEdges))
        fovList = []
        for setIdx1 in range(len(self.featuresList)):
            featureSet1 = set(self.featuresList[setIdx1].tolist())
            for setIdx2 in range(setIdx1+1, len(self.featuresList)):
                featureSet2 = set(self.featuresList[setIdx2].tolist())
                contingency = [[len(featureSet1.intersection(featureSet2)),
                                len(featureSet2.difference(featureSet1))],
                               [len(featureSet1.difference(featureSet2)),
                                len(allFeatures.difference(featureSet1.union(featureSet2)))]]
                fovList.append(-np.log10(st.fisher_exact(contingency, alternative='greater')[1]))
        return fovList


    def computeConsistency(self):
        """ Compute the pairwise consistency indices between the sets of selected features.

        Returns
        -------
        cixList: list
            List of pairwise consistency indices between the sets of selected features.

        Reference
        ---------
        Kuncheva, L.I. (2007).
        A Stability Index for Feature Selection. AIA, pp. 390--395.
        """
        cixList = []
        for setIdx1 in range(len(self.featuresList)):
            featureSet1 = set(self.featuresList[setIdx1])
            for setIdx2 in range(setIdx1+1, len(self.featuresList)):
                featureSet2 = set(self.featuresList[setIdx2])
                observed = float(len(featureSet1.intersection(featureSet2)))
                expected = len(featureSet1) * len(featureSet2) / float(self.numEdges)
                maxposbl = float(min(len(featureSet1), len(featureSet2)))
                if expected == maxposbl:
                    cixList.append(0.)
                else:
                    cixList.append((observed - expected) / (maxposbl - expected))
        return cixList
        

def main():
    """ Run a cross-validation experiment on sample-specific co-expression networks.

    Example:
        $ python OuterCrossVal.py outputs/U133A_combat_DMFS lioness -o 5 -k 5 -m 400
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

    # Get the total number of samples
    numSamples = 0
    for foldNr in range(args.num_outer_folds):
        with open('%s/%d/test.indices' % (args.data_path, foldNr)) as f:
            numSamples += len(f.readlines())
            f.close()

    # Initialize OuterCrossVal
    ocv = OuterCrossVal(args.data_path, args.network_type, numSamples,
                        args.num_inner_folds, args.num_outer_folds, args.max_nr_feats)
    # Run the experiment
    ocv.runOuterL1LogReg()

    # Print number of selected features
    print "Number of features selected per fold: ", [len(x) for x in ocv.featuresList]
    
    # Get the AUC
    print "AUC:", ocv.computeAUC()

    # Get the stability (Fisher overlap)
    fovList = ocv.computeFisherOverlap()
    print "Stability (Fisher overlap):", fovList 
    plt.figure()
    plt.boxplot(fovList, 0, 'gD')
    plt.title('Fisher overlap')
    plt.ylabel('-log10(p-value)')
    plt.show()

    # Get the stability (consistency index)
    cixList = ocv.computeConsistency()
    print "Stability (Consistency Index):", cixList
    plt.figure()
    plt.boxplot(cixList, 0, 'gD')
    plt.title('Consistency Index')
    plt.show()
    
    

if __name__ == "__main__":
    main()
        