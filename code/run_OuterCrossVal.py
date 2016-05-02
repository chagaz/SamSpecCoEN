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

import OuterCrossVal

def main():
    """ Perform a cross-validation experiment on sample-specific co-expression networks
    in the case where inner cross-validation results are available.
    
    Example
    -------
        $ python run_OuterCrossVal.py ACES/experiments/data/ \
         outputs/U133A_combat_RFS/subtype_stratified/repeat0 lioness \
         outputs/U133A_combat_RFS/subtype_stratified/repeat0/results/lioness -o 10 -k 5 -m 1000

        $ python run_OuterCrossVal.py ACES/experiments/data/ \
          outputs/U133A_combat_RFS/subtype_stratified/repeat0 lioness \
          outputs/U133A_combat_RFS/subtype_stratified/repeat0/results/sfan -o 10 -k 5 -m 1000 \
          -s ../../sfan/code 
    
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
    parser.add_argument("aces_data_path", help="Path to the folder containing the ACES data")
    parser.add_argument("network_data_path", help="Path to the folder containing the network data")
    parser.add_argument("network_type", help="Type of co-expression networks")
    parser.add_argument("results_dir", help="Folder where inner cross-validation results are stored")
    parser.add_argument("-o", "--num_outer_folds", help="Number of outer cross-validation folds",
                        type=int)
    parser.add_argument("-k", "--num_inner_folds", help="Number of inner cross-validation folds",
                        type=int)
    parser.add_argument("-m", "--max_nr_feats", help="Maximum number of selected features",
                        type=int)
    parser.add_argument("-n", "--nodes", action='store_true', default=False,
                        help="Work with node weights rather than edge weights")
    parser.add_argument("-s", "--sfan",
                        help='Path to sfan code (then automatically use sfan + l2 logistic regression)')
    parser.add_argument("-e", "--enet", action='store_true', default=False,
                        help="Only run elastic net")
    args = parser.parse_args()

    try:
        assert args.network_type in ['lioness', 'regline']
    except AssertionError:
        sys.stderr.write("network_type should be one of 'lioness', 'regline'.\n")
        sys.stderr.write("Aborting.\n")
        sys.exit(-1)

    # Get the total number of samples
    num_samples = 0
    for fold_nr in range(args.num_outer_folds):
        with open('%s/fold%d/test.indices' % (args.network_data_path, fold_nr)) as f:
            num_samples += len(f.readlines())
            f.close()
    print "%d samples" % num_samples

    if args.sfan:
        # ========= Sfan =========
        # Initialize OuterCrossVal
        ocv = OuterCrossVal.OuterCrossVal(args.aces_data_path, args.network_data_path, 
                                          args.network_type, num_samples,
                                          args.num_inner_folds, args.num_outer_folds, 
                                          max_nr_feats=args.max_nr_feats,
                                          use_nodes=True, use_sfan=True, sfan_path=args.sfan)
        
        # Baseline using only connected features
        results_dir = '%s/nosel' % args.results_dir
        # Read outputs from inner cross-validation experiments 
        ocv.read_inner_results(results_dir)

        # Write results
        ocv.write_results(results_dir)

        #  Use sfan to select features
        # Read outputs from inner cross-validation experiments
        ocv.read_inner_results(args.results_dir)

        # Write results
        ocv.write_results(args.results_dir)
        # ========= End sfan =========

    else:
        # Initialize OuterCrossVal
        ocv = OuterCrossVal.OuterCrossVal(args.aces_data_path, args.network_data_path, 
                                          args.network_type, num_samples,
                                          args.num_inner_folds, args.num_outer_folds, 
                                          max_nr_feats=args.max_nr_feats,
                                          use_nodes=args.nodes)
        
        # ========= l1-regularized logistic regression =========\
        if not args.enet:
            # Read outputs from inner cross-validation experiments
            ocv.read_inner_results(args.results_dir)

            # Write results
            ocv.write_results(args.results_dir)
        # ========= End l1-regularized logistic regression =========

        
        # ========= l1/l2-regularized logistic regression =========
        results_dir = 'args.results_dir/enet'
        # Read outputs from inner cross-validation experiments
        ocv.read_inner_results(results_dir)

        # Write results
        ocv.write_results(results_dir)
        # ========= End l1/l2-regularized logistic regression =========

    

if __name__ == "__main__":
    main()
        
