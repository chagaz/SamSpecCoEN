# @Author 
# Chloe-Agathe Azencott
# chloe-agathe.azencott@mines-paristech.fr
# February 2016

import argparse
import h5py
import numpy as np
import os
import sys

from sklearn import cross_validation as skcv 


import CoExpressionNetwork

numFolds = 10


def main():
    """ Build sample-specific co-expression networks for one repeat of a
    10-fold "sub-type stratified cross-validation" experiment on the RFS data.

    The data will be stored under
        outputs/U133A_combat_RFS/subtype_stratified/repeat<repeat idx>
    with the following structure:
        edges.gz: 
            Gzipped file containing the list of edges of the co-expression networks.
            Each line is an undirected edge, formatted as:
                <index of gene 1> <index of gene 2>
            By convention, the index of gene 1 is smaller than that of gene 2.
        For k=1..numFolds:
            <k>/global_connectivity.png
                Regression plot of log10(p(connectivities)) against log10(connectivities)
                for the global network.
            <k>/train.indices
                List of indices of the training set (one per line).
            <k>/train.labels
                List of (0/1) labels of the training set (one per line).
            <k>/test.indices
                List of indices of the test set (one per line).
            <k>/test.labels
                List of (0/1) labels of the test set (one per line).
            <k>/lioness/edge_weights.gz:
                gzipped file containing the (self.numSamples, numEdges) array
                describing the edge weights of the LIONESS co-expression networks
                for the training samples.
            <k>/lioness/edge_weights_te.gz:
                gzipped file containing the (self.numSamples, numEdges) array
                describing the edge weights of the LIONESS co-expression networks
                for the test samples.
            <k>/regline/edge_weights.gz:
                gzipped file containing the (self.numSamples, numEdges) array
                describing the edge weights of the Regline co-expression networks
                for the training samples.
            <k>/regline/edge_weights_te.gz:
                gzipped file containing the (self.numSamples, numEdges) array
                describing the edge weights of the Regline co-expression networks
                for the test samples.
    Example:
        $ python setUpSubTypeStratifiedCV.py 0
    
    Reference
    ---------
    Allahyar, A., and Ridder, J. de (2015).
    FERAL: network-based classifier with application to breast cancer outcome prediction.
    Bioinformatics 31, i311--i319.
    """
    parser = argparse.ArgumentParser(description="Build sample-specific co-expression networks" + \
                                     "for a 10-fold sub-type stratified CV on the RFS data",
                                     add_help=True)
    parser.add_argument("repeat", help="Index of the repeat", type=int)
    args = parser.parse_args()

    outDir = 'outputs/U133A_combat_RFS/subtype_stratified/repeat%d' % args.repeat
    
    # Create outDir if it does not exist
    if not os.path.isdir(outDir):
        sys.stdout.write("Creating %s\n" % outDir)
        try: 
            os.makedirs(args.outDir)
        except OSError:
            if not os.path.isdir(args.outDir):
                raise

    # Get expression data, sample labels.
    # Do not normalize the data while loading it (so as not to use test data for normalization).
    f = h5py.File("ACES/experiments/data/U133A_combat.h5")
    expressionData = np.array(f['U133A_combat_RFS']['ExpressionData'])
    sampleLabels = np.array(f['U133A_combat_RFS']['PatientClassLabels'])
    f.close()
    
    # Create the data split:
    skf = skcv.StratifiedKFold(sampleLabels, n_folds=numFolds,
                               shuffle=True, random_state=args.repeat)

    for foldNr, (trIndices, teIndices) in enumerate(skf):
        # Create output directory
        foldDir = "%s/fold%d" % (outDir, foldNr)
        try: 
            os.makedirs(foldDir)
        except OSError:
            if not os.path.isdir(foldDir):
                raise
                
        # Save train indices to file
        trIndicesF = '%s/train.indices' % foldDir
        np.savetxt(trIndicesF, trIndices, fmt='%d')
        sys.stdout.write("Wrote training indices for fold %d to %s\n" % (foldNr, trIndicesF))

        # Save test indices to file
        teIndicesF = '%s/test.indices' % foldDir
        np.savetxt(teIndicesF, teIndices, fmt='%d')
        sys.stdout.write("Wrote test indices for fold %d to %s\n" % (foldNr, teIndicesF))

        # Save train labels to file
        trLabelsF = '%s/train.labels' % foldDir
        np.savetxt(trLabelsF, np.array(sampleLabels[trIndices], dtype='int'),
                   fmt='%d')
        sys.stdout.write("Wrote training labels for fold %d to %s\n" % (foldNr, trLabelsF))

        # Save test labels to file
        teLabelsF = '%s/test.labels' % foldDir
        np.savetxt(teLabelsF, np.array(sampeLabels[teIndices], dtype='int'),
                   fmt='%d')
        sys.stdout.write("Wrote test labels for fold %d to %s\n" % (foldNr, teLabelsF))

        # Create the networks
        CoExpressionNetwork.run_whole_data(expressionData, sampleLabels, foldDir,
                                           trIndices=trIndices, teIndices=teIndices)



if __name__ == "__main__":
    main()

