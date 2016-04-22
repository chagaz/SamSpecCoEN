# @Author 
# Chloe-Agathe Azencott
# chloe-agathe.azencott@mines-paristech.fr
# April 2016

import argparse
import h5py
import numpy as np
import os
import sys

#from sklearn import cross_validation as skcv 

DATA_DIR = "/share/data40T/chloe/SamSpecCoEN"

import CoExpressionNetwork

numFolds = 10


def main():
    """ Create sample-specific co-expression networks for one fold and one repeat
    of a subtype-stratified CV on the RFS data.

    Meant to be run on the cluster.
    
    The data will be stored under
        $DATA_DIR/outputs/U133A_combat_RFS/subtype_stratified/repeat<repeat idx>
    with the following structure:
        edges.gz: 
            Gzipped file containing the list of edges of the co-expression networks.
            Each line is an undirected edge, formatted as:
                <index of gene 1> <index of gene 2>
            By convention, the index of gene 1 is smaller than that of gene 2.
        For k=1..numFolds:
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
        $ python setUpSubTypeStratifiedCV_computeNetworks.py 0 0
    
    Reference
    ---------
    Allahyar, A., and Ridder, J. de (2015).
    FERAL: network-based classifier with application to breast cancer outcome prediction.
    Bioinformatics 31, i311--i319.
    """
    parser = argparse.ArgumentParser(description="Build sample-specific co-expression networks" + \
                                     "for a 10-fold sub-type stratified CV on the RFS data",
                                     add_help=True)
    parser.add_argument("fold", help="Index of the fold", type=int)
    parser.add_argument("repeat", help="Index of the repeat", type=int)
    args = parser.parse_args()

    outDir = '%s/outputs/U133A_combat_RFS/subtype_stratified/repeat%d' % (DATA_DIR, args.repeat)

    # Get expression data, sample labels.
    # Do not normalize the data while loading it (so as not to use test data for normalization).
    f = h5py.File("%s/ACES/experiments/data/U133A_combat.h5" % DATA_DIR)
    expressionData = np.array(f['U133A_combat_RFS']['ExpressionData'])
    sampleLabels = np.array(f['U133A_combat_RFS']['PatientClassLabels'])
    f.close()
    
    foldNr = args.fold
    # Output directory
    foldDir = "%s/fold%d" % (outDir, foldNr)

    # Read train indices from file
    trIndicesF = '%s/train.indices' % foldDir
    trIndices = np.loadtxt(trIndicesF, dtype=int)
    sys.stdout.write("Read training indices for fold %d from %s\n" % (foldNr, trIndicesF))

    # Read test indices from file
    teIndicesF = '%s/test.indices' % foldDir
    teIndices = np.loadtxt(teIndicesF, dtype=int)
    sys.stdout.write("Read training indices for fold %d from %s\n" % (foldNr, teIndicesF))
    print teIndices
    print teIndices.shape

    # Create networks
    CoExpressionNetwork.run_whole_data(expressionData, sampleLabels, foldDir,
                                       trIndices=trIndices, teIndices=teIndices)



if __name__ == "__main__":
    main()

