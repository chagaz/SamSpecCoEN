# @Author 
# Chloe-Agathe Azencott
# chloe-agathe.azencott@mines-paristech.fr
# April 2016

import argparse
import h5py
import numpy as np
import os
import sys

from sklearn import cross_validation as skcv 

DATA_DIR = "/share/data40T/chloe/SamSpecCoEN"

#import CoExpressionNetwork

numFolds = 10


def main():
    """ Create train/test indices for one repeat of a subtype-stratified CV
    on the RFS data.

    Meant to be run on the cluster.
    
    The data will be stored under
        $DATA_DIR/outputs/U133A_combat_RFS/subtype_stratified/repeat<repeat idx>
    with the following structure:
        For k=1..numFolds:
            <k>/train.indices
                List of indices of the training set (one per line).
            <k>/train.labels
                List of (0/1) labels of the training set (one per line).
            <k>/test.indices
                List of indices of the test set (one per line).
            <k>/test.labels
                List of (0/1) labels of the test set (one per line).
    Example:
        $ python setUpSubTypeStratifiedCV_writeIndices.py 0
    
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

    outDir = '%s/outputs/U133A_combat_RFS/subtype_stratified/repeat%d' % (DATA_DIR, args.repeat)
    
    # Create outDir if it does not exist
    if not os.path.isdir(outDir):
        sys.stdout.write("Creating %s\n" % outDir)
        try: 
            os.makedirs(outDir)
        except OSError:
            if not os.path.isdir(outDir):
                raise

    # Get expression data, sample labels.
    # Do not normalize the data while loading it (so as not to use test data for normalization).
    f = h5py.File("%s/ACES/experiments/data/U133A_combat.h5" % DATA_DIR)
    #expressionData = np.array(f['U133A_combat_RFS']['ExpressionData'])
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
        np.savetxt(teLabelsF, np.array(sampleLabels[teIndices], dtype='int'),
                   fmt='%d')
        sys.stdout.write("Wrote test labels for fold %d to %s\n" % (foldNr, teLabelsF))

        # # Create the networks
        # CoExpressionNetwork.run_whole_data(expressionData, sampleLabels, foldDir,
        #                                    trIndices=trIndices, teIndices=teIndices)



if __name__ == "__main__":
    main()

