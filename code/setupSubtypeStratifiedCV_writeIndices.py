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


def main():
    """ Create train/test indices for one repeat of a subtype-stratified cross-validation experiment
    on the ACES data.

    The data will be stored under
        <data_dir>/outputs/U133A_combat_<dataset_name>/subtype_stratified/repeat<repeat idx>
    with the following structure:
        For k=1..num_folds:
            <k>/train.indices
                List of indices of the training set (one per line).
            <k>/train.labels
                List of (0/1) labels of the training set (one per line).
            <k>/test.indices
                List of indices of the test set (one per line).
            <k>/test.labels
                List of (0/1) labels of the test set (one per line).
    Example:
        $ python setUpSubTypeStratifiedCV_writeIndices.py /share/data40T/chloe/SamSpecCoEN RFS 10 0
    
    Reference
    ---------
    Allahyar, A., and Ridder, J. de (2015).
    FERAL: network-based classifier with application to breast cancer outcome prediction.
    Bioinformatics 31, i311--i319.
    """
    parser = argparse.ArgumentParser(description="Build the train and test indices" + \
                                     "for a sub-type stratified cross-validation on the RFS data",
                                     add_help=True)
    parser.add_argument("data_dir", help="Data directory")
    parser.add_argument("dataset_name", help="Dataset name")
    parser.add_argument("num_folds", help="Number of folds", type=int)
    parser.add_argument("repeat", help="Index of the repeat", type=int)
    args = parser.parse_args()

    try:
        assert args.dataset_name in ['DMFS', 'RFS', 'SOS']
    except AssertionError:
        sys.stderr.write("dataset_name should be one of 'DMFS', 'RFS', 'SOS'.\n")
        sys.stderr.write("Aborting.\n")
        sys.exit(-1)
    
    out_dir = '%s/outputs/U133A_combat_%s/subtype_stratified/repeat%d' % (args.data_dir,
                                                                          args.dataset_name,
                                                                          args.repeat)
    # Create out_dir if it does not exist
    if not os.path.isdir(out_dir):
        sys.stdout.write("Creating %s\n" % out_dir)
        try: 
            os.makedirs(out_dir)
        except OSError:
            if not os.path.isdir(out_dir):
                raise

    # Get expression data, sample labels.
    f = h5py.File("%s/ACES/experiments/data/U133A_combat.h5" % args.data_dir)
    sample_labels = np.array(f['U133A_combat_%s' % args.dataset_name]['PatientClassLabels'])
    f.close()
    
    # Create the data split:
    skf = skcv.StratifiedKFold(sample_labels, n_folds=num_folds,
                               shuffle=True, random_state=args.repeat)

    for fold_nr, (tr_indices, te_indices) in enumerate(skf):
        # Create output directory
        fold_dir = "%s/fold%d" % (out_dir, fold_nr)
        try: 
            os.makedirs(fold_dir)
        except OSError:
            if not os.path.isdir(fold_dir):
                raise
                
        # Save train indices to file
        tr_indices_f = '%s/train.indices' % fold_dir
        np.savetxt(tr_indices_f, tr_indices, fmt='%d')
        sys.stdout.write("Wrote training indices for fold %d to %s\n" % (fold_nr, tr_indices_f))

        # Save test indices to file
        te_indices_f = '%s/test.indices' % fold_dir
        np.savetxt(te_indices_f, te_indices, fmt='%d')
        sys.stdout.write("Wrote test indices for fold %d to %s\n" % (fold_nr, te_indices_f))

        # Save train labels to file
        tr_labels_f = '%s/train.labels' % fold_dir
        np.savetxt(tr_labels_f, np.array(sample_labels[tr_indices], dtype='int'),
                   fmt='%d')
        sys.stdout.write("Wrote training labels for fold %d to %s\n" % (fold_nr, tr_labels_f))

        # Save test labels to file
        te_labels_f = '%s/test.labels' % fold_dir
        np.savetxt(te_labels_f, np.array(sample_labels[te_indices], dtype='int'),
                   fmt='%d')
        sys.stdout.write("Wrote test labels for fold %d to %s\n" % (fold_nr, te_labels_f))


if __name__ == "__main__":
    main()

