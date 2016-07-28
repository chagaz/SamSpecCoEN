# @Author 
# Chloe-Agathe Azencott
# chloe-agathe.azencott@mines-paristech.fr
# July 2016

import argparse
import h5py
import numpy as np
import os
import sys

DATA_DIR = "/share/data40T/chloe/SamSpecCoEN"

import CoExpressionNetwork

numFolds = 10


def main():
    """ Create sample-specific LIONESS co-expression networks for one fold and one repeat
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

    out_dir = '%s/outputs/U133A_combat_RFS/subtype_stratified/repeat%d' % (DATA_DIR, args.repeat)

    # Get expression data, sample labels.
    # Do not normalize the data while loading it (so as not to use test data for normalization).
    f = h5py.File("%s/ACES/experiments/data/U133A_combat.h5" % DATA_DIR)
    expression_data = np.array(f['U133A_combat_RFS']['ExpressionData'])
    sample_labels = np.array(f['U133A_combat_RFS']['PatientClassLabels'])
    f.close()

    fold_nr = args.fold
    # Output directory
    fold_dir = "%s/fold%d" % (out_dir, fold_nr)

    # Read train indices from file
    tr_indices_f = '%s/train.indices' % fold_dir
    tr_indices = np.loadtxt(tr_indices_f, dtype=int)
    sys.stdout.write("Read training indices for fold %d from %s\n" % (fold_nr, tr_indices_f))

    # Read test indices from file
    te_indices_f = '%s/test.indices' % fold_dir
    te_indices = np.loadtxt(te_indices_f, dtype=int)
    sys.stdout.write("Read training indices for fold %d from %s\n" % (fold_nr, te_indices_f))
    print te_indices
    print te_indices.shape

    # Create networks
    CoExpressionNetwork.run_whole_data_lioness(expression_data, sample_labels, fold_dir,
                                               tr_indices=tr_indices, te_indices=te_indices)



if __name__ == "__main__":
    main()

