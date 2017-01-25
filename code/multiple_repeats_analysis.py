# @Author 
# Chloe-Agathe Azencott
# chloe-agathe.azencott@mines-paristech.fr

import argparse
import h5py

# import warnings
# warnings.filterwarnings("error", category=RuntimeWarning)

import gzip
import numpy as np
import os
import sys

from sklearn import metrics, linear_model, model_selection

import OuterCrossVal


def get_selected_genes(min_number_folds, aces_data, features_list, use_nodes, network_root):
    """ Get the genes that have been selected in a given number of folds.

    Parameters
    ----------
    min_number_folds: int
        Number of folds in which a feature must appear to be selected.
    aces_data: datatypes.ExpressionDataset.ExpressionDataset
        Data in ACES format, read using HDF5GroupToExpression_dataset.
    features_list: list
        List of selected features, for all repeats and folds.
    use_nodes: bool
        Whether to use node weights rather than edge weights as features.
    network_root: path
        Path to folder containing network skeleton and edges.

    Output
    ------
    selected_features_list: list
        indices of selected features.

    selected_genes_dict: dict
        keys: selected genes
        values: number of selected edges that gene belongs to;
                0 if nodes-based selection.
    """
    ### Create list of indices of features that are selected.
    selected_features_dict = {} # feat_idx:number_of_times_selected
    for feature_set in features_list:
        for feat_idx in feature_set:
            if not selected_features_dict.has_key(feat_idx):
                selected_features_dict[feat_idx] = 1
            else:
                selected_features_dict[feat_idx] += 1

    selected_features_list = []

    for feat_idx, number_of_times_selected in selected_features_dict.iteritems():
        try:
            assert number_of_times_selected <= len(features_list)
        except AssertionError:
            print feat_idx, number_of_times_selected
            sys.stderr.write("Error in computing the number of times a feature was selected.\n")
            sys.stderr.write("Aborting.\n")
            sys.exit(-1)
        if number_of_times_selected >= min_number_folds:
            selected_features_list.append(feat_idx)
    selected_features_list.sort()

    print "%d features in the final selection." % len(selected_features_list)

    ### Map features to gene names
    if use_nodes:
        ## Features are genes
        aces_gene_names = aces_data.geneLabels
        selected_genes_dict = {aces_gene_names[ix]:0 for ix in selected_features_list}
    else:
        ## Features are edges
        selected_genes_dict = {}
        edges_f = '%s/edges_entrez.gz' % network_root
        edges_entrez_list = []
        with gzip.open(edges_f) as f:
            edges_entrez_list = [[line.split()[0], line.split()[1]] for line in f.readlines()]
            f.close()
        edges_entrez_list = [edges_entrez_list[i] for i in selected_features_list]
        for edge in edges_entrez_list:
            for g in edge:
                if not selected_genes_dict.has_key(g):
                    selected_genes_dict[g] = 1
                else:
                    selected_genes_dict[g] += 1
        print "\t This corresponds to %d distinct genes" % len(selected_genes_dict.keys())
    return selected_features_list, selected_genes_dict


def final_cv_score(selected_features_list, aces_data, use_nodes, network_root, network_type):
    """ Get the cross-validated predictivity of the selected features

    Parameters
    ----------
    selected_features_list: list
        indices of selected features.
    aces_data: datatypes.ExpressionDataset.ExpressionDataset
        Data in ACES format, read using HDF5GroupToExpression_dataset.
    use_nodes: bool
        Whether to use node weights rather than edge weights as features.
    network_root: path
        Path to folder containing network skeleton and edges.
    network_type: string
        Type of network to work with.

    Output
    ------
    cv_scores: float
        cross-validated AUCs of a logistic regression using only
        the selected features.

    # cv_score_ridge: float
    #     cross-validated AUC of a rige-regularized logistic regression
    #     using only the selected features.            
    """
    # Get data
    y_data = aces_data.patientClassLabels
    if use_nodes:
        print "Using node weights as features"
        x_data = aces_data.expressionData[:, selected_features_list]
    else:
        print "Using edge weights as features"
        x_f = '%s/%s/edge_weights.gz' % (network_root, network_type)
        x_data = np.loadtxt(x_f).transpose()[:, selected_features_list]

    # Initialize logistic regression cross-validation classifier
    cv_clf = linear_model.LogisticRegression(C=1e6, class_weight='balanced')

    # 10-fold cross-validation
    cv_folds = model_selection.KFold(n_splits=10).split(y_data)
    pred = np.zeros(y_data.shape)
    for tr, te in cv_folds:
        Xtr = x_data[tr, :]
        ytr = y_data[tr]
        Xte = x_data[te, :]

        # Fit classifier
        cv_clf.fit(Xtr, ytr)

        # Predict probabilities (of belonging to +1 class) on test data
        yte_pred = cv_clf.predict_proba(Xte)
        pred[te] = yte_pred[:, np.nonzero(cv_clf.classes_ == 1)[0][0]]

    return metrics.roc_auc_score(y_data, pred)

    

def final_analysis(features_list, results_dir, threshold, aces_data_path, use_nodes,
                   network_root, network_type):
    """ Create results files.

    Parameter
    ---------
    features_list: list
        List of selected features, for all repeats and folds.
    results_dir: path
        Where to save results.
    threshold: int
        Number of folds a feature must appear in to be selected.
    aces_data_path: path
        Path to folder containing ACES data.
    use_nodes: bool
        Whether to use node weights rather than edge weights as features.
    network_root: path
        Path to folder containing network skeleton and edges.
    network_type: string
        Type of network to work with.

    Created files
    -------------
    <results_dir>/final_selection_genes.txt
        list of names of selected genes + number of edges they belong to
    <results_dir>/final_selection_results.txt
        - cross-validated predictivity (ridge regression) of selected features
    """
    # Read ACES data
    sys.path.append(aces_data_path)
    from datatypes.ExpressionDataset import HDF5GroupToExpressionDataset
    f = h5py.File("%s/experiments/data/U133A_combat.h5" % aces_data_path)
    aces_data = HDF5GroupToExpressionDataset(f['U133A_combat_RFS'],
                                             checkNormalise=False)
    f.close()

    # Get final selection
    selected_features_list, selected_genes_dict = get_selected_genes(threshold, aces_data,
                                                                     features_list, use_nodes,
                                                                     network_root) 
    sel_fname = '%s/final_selection_genes.txt' % results_dir
    with open(sel_fname, 'w') as f:
        for g, v in selected_genes_dict.iteritems():
            f.write("%s %d\n" % (g, v))
        f.close()

    # Cross-validated predictivity of selected features
    cv_scores = final_cv_score(selected_features_list, aces_data, use_nodes,
                               network_root, network_type)
    res_fname = '%s/final_selection_results.txt' % results_dir
    with open(res_fname, 'w') as f:
        # Write
        f.write("Number of features used:\t%d\n" % len(selected_features_list))

        # Write AUC
        f.write("Logistic regression AUC:\t%.2f\n" % cv_scores)

        # f.write("Ridge logistic regression AUC:\t%.2f\n" % cv_score_ridge)
        f.close()
    


def main():
    """ Analyze the features that were selected across a fraction of folds 
    in a repeated CV experiment.
    
    Example
    -------
        $ python multiple_repeats_analysis.py ../ACES ../outputs/U133A_combat_RFS/KEGG_edges1210 \
         ../outputs/U133A_combat_RFS/subtype_stratified/KEGG_edges1210  \
         regline -o 10 -r 5 -t 50
    
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
    parser.add_argument("repeat_path", help="Folder containing the results for each repeat")
    parser.add_argument("network_type", help="Type of co-expression network")
    parser.add_argument("-o", "--num_outer_folds", help="Number of outer cross-validation folds",
                        type=int)
    parser.add_argument("-r", "--num_repeats", help="Number of repeats",
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

    try:
        assert args.threshold <= (args.num_repeats * args.num_outer_folds)
    except AssertionError:
        sys.stderr.write("threshold should be smaller than or equal to")
        sys.stderr.write("the product of num_repeats and num_outer_folds.\n")
        sys.stderr.write("Aborting.\n")
        sys.exit(-1)

    # Get the total number of samples
    num_samples = 0
    for fold_nr in range(args.num_outer_folds):
        with open('%s/repeat0/fold%d/test.indices' % (args.repeat_path, fold_nr)) as f:
            num_samples += len(f.readlines())
            f.close()
    print "%d samples" % num_samples

    # for using OuterCrossVal with some dummy non-optional variables
    dummy_str = "" 
    dummy_int = 0
    
    if args.sfan:
        # ========= Baseline using only connected features =========
        # Determine name of results dir
        results_dir = '%s/results/nodes/sfan/nosel' % args.repeat_path
        # Create results dir if it does not exist
        if not os.path.isdir(results_dir):
            sys.stdout.write("Creating %s\n" % results_dir)
            try: 
                os.makedirs(results_dir)
            except OSError:
                if not os.path.isdir(results_dir):
                    raise

        # Read features selected in inner cross-validation experiments
        features_list = []
        for repeat_idx in range(args.num_repeats):
            innercv_path = '%s/repeat$d' % (args.repeat_path, repeat_idx)
            ocv = OuterCrossVal.OuterCrossVal(args.aces_data_path, args.network_path,
                                              innercv_path,
                                              args.network_type, num_samples,
                                              dummy_int, args.num_outer_folds, 
                                              use_nodes=True, use_sfan=True, sfan_path=args.sfan)
            ocv.read_inner_results('nosel')
            features_list.extend(ocv.features_list)
                        
        # Analyze and write results
        final_analysis(features_list, results_dir, args.threshold, args.aces_data_path,
                       True, args.network_path, args.network_type)
        # ========= End baseline using only connected features =========

        
        # ========= Sfan  =========                    
        # Determine name of results dir
        results_dir = '%s/results/nodes/sfan' % args.repeat_path
        # Create results dir if it does not exist
        if not os.path.isdir(results_dir):
            sys.stdout.write("Creating %s\n" % results_dir)
            try: 
                os.makedirs(results_dir)
            except OSError:
                if not os.path.isdir(results_dir):
                    raise

        # Read features selected in inner cross-validation experiments
        features_list = []
        for repeat_idx in range(args.num_repeats):
            innercv_path = '%s/repeat$d' % (args.repeat_path, repeat_idx)
            ocv = OuterCrossVal.OuterCrossVal(args.aces_data_path, args.network_path,
                                              innercv_path,
                                              args.network_type, num_samples,
                                              dummy_int, args.num_outer_folds, 
                                              use_nodes=True, use_sfan=True, sfan_path=args.sfan)
            ocv.read_inner_results('nosel')
            features_list.extend(ocv.features_list)

        # Analyze and write results
        final_analysis(features_list, results_dir, args.threshold, args.aces_data_path,
                       True, args.network_path, args.network_type)
        # ========= End sfan =========

    else:
        # ========= Regularized logistic regression =========
        # Determine name of results dir
        if args.enet:
            if args.nodes:
                results_dir = '%s/results/nodes/enet' % args.repeat_path
            else:
                results_dir = '%s/results/%s/enet' % (args.repeat_path, args.network_type)
        else:
            if args.nodes:
                results_dir = '%s/results/nodes' % args.repeat_path
            else:
                results_dir = '%s/results/%s' % (args.repeat_path, args.network_type)

        # Create results dir if it does not exist
        if not os.path.isdir(results_dir):
            sys.stdout.write("Creating %s\n" % results_dir)
            try: 
                os.makedirs(results_dir)
            except OSError:
                if not os.path.isdir(results_dir):
                    raise

        # Read features selected in inner cross-validation experiments
        features_list = []
        for repeat_idx in range(args.num_repeats):
            innercv_path = '%s/repeat%d' % (args.repeat_path, repeat_idx)
            ocv = OuterCrossVal.OuterCrossVal(args.aces_data_path, args.network_path,
                                              innercv_path,
                                              args.network_type, num_samples,
                                              dummy_int, args.num_outer_folds, 
                                              use_nodes=args.nodes, use_enet=args.enet)
            ocv.read_inner_results()
            features_list.extend(ocv.features_list)

                        
        # Analyze and write results
        final_analysis(features_list, results_dir, args.threshold, args.aces_data_path,
                       args.nodes, args.network_path, args.network_type)
        # ========= End regularized logistic regression =========
            
if __name__ == "__main__":
    main()