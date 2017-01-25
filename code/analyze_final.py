# @Author 
# Chloe-Agathe Azencott
# chloe-agathe.azencott@mines-paristech.fr

import argparse
# import h5py

# import matplotlib # in a non-interactive environment
# matplotlib.use('Agg') # in a non-interactive environment
# import matplotlib.pyplot as plt

# import warnings
# warnings.filterwarnings("error", category=RuntimeWarning)

# import numpy as np
import os
# import scipy.stats as st
import sys

# from sklearn import metrics 

import OuterCrossVal


def main():
    """ Analyze the features that were selected across a fraction of folds in a CV experiment.
    
    Example
    -------
        $ python analyze_final.py ../ACES ../outputs/U133A_combat_RFS \
         ../outputs/U133A_combat_RFS/subtype_stratified/repeat0  \
         regline -o 10 -k 5 -m 1000 -t 10
    
    Files created
    -------------
    <results_dir>/final_selection_genes.txt
        list of names of selected genes + number of edges they belong to
    <results_dir>/final_selection_results.txt
        - cross-validated predictivity (ridge regression) of selected features
    """
    parser = argparse.ArgumentParser(description="Analyze the selected features",
                                     add_help=True)
    parser.add_argument("aces_data_path", help="Folder containing the ACES data")
    parser.add_argument("network_path", help="Folder containing network skeleton and weights")
    parser.add_argument("innercv_path", help="Folder containing the inner cross-validation results")
    parser.add_argument("network_type", help="Type of co-expression network")
    parser.add_argument("-o", "--num_outer_folds", help="Number of outer cross-validation folds",
                        type=int)
    parser.add_argument("-k", "--num_inner_folds", help="Number of inner cross-validation folds",
                        type=int)
    parser.add_argument("-m", "--max_nr_feats", help="Maximum number of selected features",
                        type=int)
    parser.add_argument("-t", "--threshold", help="Number of folds a feature must appear in to be selected",
                        type=int)
    parser.add_argument("-n", "--nodes", action='store_true', default=False,
                        help="Work with node weights rather than edge weights")
    parser.add_argument("-s", "--sfan",
                        help='Path to sfan code (then automatically use sfan + l2 logistic regression)')
    parser.add_argument("-e", "--enet", action='store_true', default=False,
                        help="Run elastic net instead of lasso.")
    args = parser.parse_args()

    try:
        assert args.network_type in OuterCrossVal.network_types
    except AssertionError:
        sys.stderr.write("network_type should be one of ")
        sys.stderr.write(",".join([" '%s'" % nt for nt in network_types]))
        sys.stderr.write(".\n")
        sys.stderr.write("Aborting.\n")
        sys.exit(-1)

    # Get the total number of samples
    num_samples = 0
    for fold_nr in range(args.num_outer_folds):
        with open('%s/fold%d/test.indices' % (args.innercv_path, fold_nr)) as f:
            num_samples += len(f.readlines())
            f.close()
    print "%d samples" % num_samples

    if args.sfan:
        # ========= Sfan =========
        # Initialize OuterCrossVal
        ocv = OuterCrossVal. OuterCrossVal(args.aces_data_path, args.network_path,
                                           args.innercv_path, 
                                           args.network_type, num_samples,
                                           args.num_inner_folds, args.num_outer_folds, 
                                           max_nr_feats=args.max_nr_feats,
                                           use_nodes=True, use_sfan=True, sfan_path=args.sfan)
        
        # Baseline using only connected features
        # Read outputs from inner cross-validation experiments 
        ocv.read_inner_results('nosel')

        # Write results
        results_dir = '%s/repeat0/results/nodes/sfan/nosel' % args.innercv_path
        # Create results dir if it does not exist
        if not os.path.isdir(results_dir):
            sys.stdout.write("Creating %s\n" % results_dir)
            try: 
                os.makedirs(results_dir)
            except OSError:
                if not os.path.isdir(results_dir):
                    raise
        
        ocv.final_analysis(results_dir, args.threshold)

                    
        #  Use sfan to select features
        # Read outputs from inner cross-validation experiments
        ocv.read_inner_results()

        # Write results
        results_dir = '%s/results/nodes/sfan' % args.innercv_path
        # Create results dir if it does not exist
        if not os.path.isdir(results_dir):
            sys.stdout.write("Creating %s\n" % results_dir)
            try: 
                os.makedirs(results_dir)
            except OSError:
                if not os.path.isdir(results_dir):
                    raise
        ocv.final_analysis(results_dir, args.threshold)
        # ========= End sfan =========

    else:
        # Initialize OuterCrossVal
        ocv = OuterCrossVal.OuterCrossVal(args.aces_data_path, args.network_path,
                                          args.innercv_path, 
                                          args.network_type, num_samples,
                                          args.num_inner_folds, args.num_outer_folds, 
                                          max_nr_feats=args.max_nr_feats,
                                          use_nodes=args.nodes, use_enet=args.enet)
        
        # ========= l1-regularized logistic regression =========\
        if not args.enet:
            # Read outputs from inner cross-validation experiments
            ocv.read_inner_results()

            # Write results
            if args.nodes:
                results_dir = '%s/results/nodes' % args.innercv_path
            else:
                results_dir = '%s/results/%s' % (args.innercv_path, args.network_type)
            # Create results dir if it does not exist
            if not os.path.isdir(results_dir):
                sys.stdout.write("Creating %s\n" % results_dir)
                try: 
                    os.makedirs(results_dir)
                except OSError:
                    if not os.path.isdir(results_dir):
                        raise
            ocv.final_analysis(results_dir, args.threshold)
        # ========= End l1-regularized logistic regression =========

        
        # ========= l1/l2-regularized logistic regression =========
        else:
            # Read outputs from inner cross-validation experiments
            ocv.read_inner_results()

            # Write results
            if args.nodes:
                results_dir = '%s/results/nodes/enet' % args.innercv_path
            else:
                results_dir = '%s/results/%s/enet' % (args.innercv_path, args.network_type)
            # Create results dir if it does not exist
            if not os.path.isdir(results_dir):
                sys.stdout.write("Creating %s\n" % results_dir)
                try: 
                    os.makedirs(results_dir)
                except OSError:
                    if not os.path.isdir(results_dir):
                        raise
            ocv.final_analysis(results_dir, args.threshold)
        # ========= End l1/l2-regularized logistic regression =========

            
if __name__ == "__main__":
    main()