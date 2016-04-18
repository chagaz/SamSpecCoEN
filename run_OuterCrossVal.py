# @Author 
# Chloe-Agathe Azencott
# chloe-agathe.azencott@mines-paristech.fr
# April 2016

import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats as st
import sys

from sklearn import metrics as skm
orange_color = '#d66000'
blue_color = '#005599'

import OuterCrossVal

def main():
    """ Perform a cross-validation experiment on sample-specific co-expression networks
    in the case where inner cross-validation results are available.
    
    Example
    -------
        $ python run_OuterCrossVal.py outputs/U133A_combat_RFS/subtype_stratified/repeat0 lioness outputs/U133A_combat_RFS/subtype_stratified/repeat0/results/lioness -o 10 -k 5 -m 400 

    
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
    parser.add_argument("results_dir", help="Folder where inner cross-validation results are stored")
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
    
    # Read outputs from inner cross-validation experiments
    ocv.readOuterL1LogReg()

    # Open results file for writing
    res_fname = '%s/results.txt' % args.results_dir
    with open(res_fname, 'r') as f:
    
        # Write number of selected features
        f.write("Number of features selected per fold:\t")
        f.write("%s\n" % " ".join["%d" % len(x) for x in ocv.featuresList])
    
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
        