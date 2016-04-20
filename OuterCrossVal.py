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
            Possible value: 'lioness', 'regline'
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
        self.numEdges = np.loadtxt("%s/fold0/edges.gz" % self.dataRoot).shape[0]
        
        
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
            

    def readOuterL1LogReg(self, innerCV_resdir):
        """ Read the results of the outer loop of the experiment,
        for an l1-regularized logistic regression.

        Parameters
        ----------
        innerCV_resdir: path
            Path to outputs of InnerCrossVal for each fold

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
            sys.stdout.write("Reading results for fold number %d\n" % fold)

            # Read the test indices
            dataFoldRoot = '%s/fold%d' % (self.dataRoot, fold)
            teIndices = np.loadtxt('%s/test.indices' % dataFoldRoot, dtype='int')
            
            # Read results from InnerCrossVal
            yte_fname = '%s/fold%d/yte' % (innerCV_resdir, fold)
            self.trueLabels[teIndices] = np.loadtxt(yte_fname, dtype='int')

            predValues_fname = '%s/fold%d/predValues' % (innerCV_resdir, fold)
            self.predValues[teIndices] = np.loadtxt(predValues_fname)

            featuresList_fname = '%s/fold%d/featuresList' % (innerCV_resdir, fold)
            self.featuresList.append(np.loadtxt(featuresList_fname, dtype='int'))

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

    Example
    -------
        $ python OuterCrossVal.py outputs/U133A_combat_DMFS lioness results/U133A_combat_DMFS/lioness -o 5 -k 5 -m 400 

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
    parser.add_argument("data_path", help="Path to the folder containing the data")
    parser.add_argument("network_type", help="Type of co-expression networks")
    parser.add_argument("results_dir", help="Folder where to store results")
    parser.add_argument("-o", "--num_outer_folds", help="Number of outer cross-validation folds",
                        type=int)
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

    # Get the total number of samples
    numSamples = 0
    for foldNr in range(args.num_outer_folds):
        with open('%s/fold%d/test.indices' % (args.data_path, foldNr)) as f:
            numSamples += len(f.readlines())
            f.close()

    # Initialize OuterCrossVal
    ocv = OuterCrossVal(args.data_path, args.network_type, numSamples,
                        args.num_inner_folds, args.num_outer_folds, args.max_nr_feats)
    # Run the experiment
    ocv.runOuterL1LogReg()

    # Create results dir if it does not exist
    if not os.path.isdir(args.results_dir):
        sys.stdout.write("Creating %s\n" % args.results_dir)
        try: 
            os.makedirs(args.results_dir)
        except OSError:
            if not os.path.isdir(args.results_dir):
                raise

    # Open results file
    res_fname = '%s/results.txt' % args.results_dir
    with open(res_fname, 'w') as f:
    
        # Write number of selected features
        f.write("Number of features selected per fold:\t")
        f.write("%s\n" % " ".join(["%d" % len(x) for x in ocv.featuresList]))
    
        # Write AUC
        f.write("AUC:\t%.2f\n" % ocv.computeAUC())

        # Write the stability (Fisher overlap)
        fovList = ocv.computeFisherOverlap()
        f.write("Stability (Fisher overlap):\t")
        f.write("%s\n" % ["%.2e" % x for x in fovList])

        # Write the stability (consistency index)
        cixList = ocv.computeConsistency()
        f.write("Stability (Consistency Index):\t")
        f.write("%s\n" % ["%.2e" % x for x in cixList])

        f.close()


    # Plot the stability (Fisher overlap)
    fov_fname = '%s/fov.pdf' % args.results_dir
    plt.figure()
    plt.boxplot(fovList, 0, 'gD')
    plt.title('Fisher overlap')
    plt.ylabel('-log10(p-value)')
    plt.savefig(fov_fname, bbox_inches='tight')

    # Plot the stability (consistency index)
    cix_fname = '%s/cix.pdf' % args.results_dir
    plt.figure()
    plt.boxplot(cixList, 0, 'gD')
    plt.title('Consistency Index')
    plt.savefig(cix_fname, bbox_inches='tight')
    
    

if __name__ == "__main__":
    main()
        
