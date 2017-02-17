# @Author 
# Chloe-Agathe Azencott
# chloe-agathe.azencott@mines-paristech.fr

import argparse
import os
import sys

import OuterCrossVal


def main():
    """ Analyze the features that were selected across a fraction of folds in a CV experiment.
    
    Example
    -------
        $ python analyze_final.py ../ACES ../outputs/U133A_combat_RFS \
         ../outputs/U133A_combat_RFS/subtype_stratified/repeat0  \
         -t regline -o 10 -k 5 -m 1000 -t 10
    
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
    parser.add_argument("-t", "--network_type",
                        help="Type of co-expression networks")
    parser.add_argument("-o", "--num_outer_folds", help="Number of outer cross-validation folds",
                        type=int)
    parser.add_argument("-k", "--num_inner_folds", help="Number of inner cross-validation folds",
                        type=int)
    parser.add_argument("-m", "--max_nr_feats", help="Maximum number of selected features",
                        type=int)
    parser.add_argument("-u", "--threshold", help="Number of folds a feature must appear in to be selected",
                        type=int)
    parser.add_argument("-n", "--nodes", action='store_true', default=False,
                        help="Work with node weights rather than edge weights")
    parser.add_argument("-c", "--cnodes", action='store_true', default=False,
                        help="Work with *connected* node weights rather than edge weights")
    parser.add_argument("-s", "--sfan",
                        help='Path to sfan code (then automatically use sfan + l2 logistic regression)')
    parser.add_argument("-e", "--enet", action='store_true', default=False,
                        help="Run elastic net instead of lasso.")
    args = parser.parse_args()

    #========= Sanity checks =========
    if args.network_type:
        try:
            assert args.network_type in OuterCrossVal.network_types
        except AssertionError:
            sys.stderr.write("network_type should be one of ")
            sys.stderr.write(",".join([" '%s'" % nt for nt in network_types]))
            sys.stderr.write(".\n")
            sys.stderr.write("Aborting.\n")
            sys.exit(-1)

    if args.network_type and (args.cnodes or args.nodes or args.sfan):
        sys.stderr.write("network_type and nodes, cnodes or sfan are incompatible\n")
        sys.stderr.write("Aborting.\n")
        sys.exit(-1)

    if args.sfan and (args.cnodes or args.nodes):
        sys.stderr.write("sfan and (c)nodes are incompatible\n")
        sys.stderr.write("Aborting.\n")
        sys.exit(-1)

    if args.cnodes and args.nodes:
        sys.stderr.write("nodes and cnodes are incompatible\n")
        sys.stderr.write("Aborting.\n")
        sys.exit(-1)
    #========= End sanity checks =========

    # Get the total number of samples
    num_samples = 0
    for fold_nr in range(args.num_outer_folds):
        with open('%s/fold%d/test.indices' % (args.innercv_path, fold_nr)) as f:
            num_samples += len(f.readlines())
            f.close()
    print "%d samples" % num_samples

    # Get results dir
    if args.cnodes:
        results_dir = "%s/results/cnodes" % args.innercv_path
    elif args.nodes:
        results_dir = "%s/results/nodes" % args.innercv_path
    elif args.sfan:
        results_dir = "%s/results/sfan" % args.innercv_path
    else:
        results_dir = "%s/results/%s" % (args.innercv_path, args.network_type)
    if args.enet:
        results_dir = "%s/enet" % results_dir

    # Create results dir if it does not exist
    if not os.path.isdir(results_dir):
        sys.stdout.write("Creating %s\n" % results_dir)
        try: 
            os.makedirs(results_dir)
        except OSError:
            if not os.path.isdir(results_dir):
                raise
                
    # Read results
    if args.sfan:
        use_sfan=True
    else:
        use_sfan=False
    ocv = OuterCrossVal.OuterCrossVal(args.aces_data_path, args.network_path, args.innercv_path, 
                                      args.network_type, num_samples,
                                      args.num_inner_folds, args.num_outer_folds, 
                                      max_nr_feats=args.max_nr_feats,
                                      use_nodes=args.nodes, use_cnodes=args.cnodes,
                                      use_sfan=use_sfan, sfan_path=args.sfan,
                                      use_enet=args.enet)
    ocv.read_inner_results()

    # Run final analysis
    ocv.final_analysis(results_dir, args.threshold)

            
if __name__ == "__main__":
    main()