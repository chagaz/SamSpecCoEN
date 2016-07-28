# @Author 
# Chloe-Agathe Azencott
# chloe-agathe.azencott@mines-paristech.fr
# July 2016

import argparse
import h5py
import numpy as np
import os
import sys

import CoExpressionNetwork

def main():
    """ Create sample-specific co-expression networks for one fold and one repeat
    of a cross-validation for which fold indices have already been computed.

    The data will be stored under
        <data_dir>/repeat<repeat idx>
    with the following structure:
        edges.gz: 
            Gzipped file containing the list of edges of the co-expression networks.
            Each line is an undirected edge, formatted as:
                <index of gene 1> <index of gene 2>
            By convention, the index of gene 1 is smaller than that of gene 2.
        For k=0..(numFolds-1):
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

    Parameters
    ----------
    aces_dir: path
        Path to the ACES folder.
    data_dir: path
        Path to the folder containing fold indices (under <data_dir>/repeat<repeat_idx>/fold<fold_idx>).
    fold: int
        Fold index.
    repeat: int
        Repeat index.

    Example
    -------
        $ python setUpSubTypeStratifiedCV_computeNetworks.py ACES ArrayExpress/postproc/ outputs/U133A_combat_RFS/subtype_stratified 0 0
    
    Reference
    ---------
    Allahyar, A., and Ridder, J. de (2015).
    FERAL: network-based classifier with application to breast cancer outcome prediction.
    Bioinformatics 31, i311--i319.
    """
    parser = argparse.ArgumentParser(description="Build sample-specific co-expression networks" + \
                                     "for a 10-fold sub-type stratified CV on the RFS data",
                                     add_help=True)
    parser.add_argument("aces_dir", help="Path to ACES data")
    parser.add_argument("mtab_dir", help="Path to MTAB data")
    parser.add_argument("data_dir", help="Path to the fold indices")
    parser.add_argument("fold", help="Index of the fold", type=int)
    parser.add_argument("repeat", help="Index of the repeat", type=int)
    args = parser.parse_args()

    out_dir = '%s/repeat%d' % (args.data_dir, args.repeat)

    # Get expression data, sample labels.
    # Do not normalize the data while loading it (so as not to use test data for normalization).
    f = h5py.File("%s/experiments/data/U133A_combat.h5" % args.aces_dir)
    expression_data = np.array(f['U133A_combat_RFS']['ExpressionData'])
    sample_labels = np.array(f['U133A_combat_RFS']['PatientClassLabels'])
    f.close()
    
    # Get reference expression data.
    f = h5py.File("%s/MTAB-62.h5" % args.refc_dir)
    refc_expression_data = np.array(f['MTAB-62']['ExpressionData'])
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
    CoExpressionNetwork.run_whole_data(expression_data, refc_expression_data,
                                       sample_labels, fold_dir,
                                       tr_indices=tr_indices, te_indices=te_indices)



if __name__ == "__main__":
    main()

