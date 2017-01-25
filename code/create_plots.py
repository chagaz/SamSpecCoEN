# @Author 
# Chloe-Agathe Azencott
# chloe-agathe.azencott@mines-paristech.fr
# April 2016

import argparse
import itertools
import os
import string
import sys

import numpy as np

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
orange_color = '#d66000'
blue_color = '#005599'

reg_list = ['lasso', 'enet'] 
xp_list = ['nodes', 'regline', 'euclthr', 'sum']


def plot_numf(results_dir, figure_path, num_repeats=10):
    """ Plot cross-validated number of selected features,
    averaged over multiple repeats.

    Parameters
    ----------
    results_dir: path
        Path to the directory containing the results.

    figure_path: path
        Path to the file where to save the plot.

    num_repeats: (int, None)
        Number of repeated experiments.        
    """
    numf_dict = {}
    for reg in reg_list:
        numf_dict[reg] = {}
        for xp in xp_list:
            numf_dict[reg][xp] = []
    
    # read number of selected features
    for repeat_idx in range(num_repeats):
        for xp in xp_list:
            # Lasso experiments
            results_f = '%s/repeat%d/results/%s/results.txt' % (results_dir,
                                                                repeat_idx, xp)
            with open(results_f, 'r') as f:
                for line in f:
                    ls = line.split('\t')
                    if ls[0] == "Number of features selected per fold:":
                        numf_list = [int(x) for x in ls[1].split()]
                        break
                f.close()
            numf_dict['lasso'][xp].extend(numf_list)

            # Enet experiments
            results_f = '%s/repeat%d/results/%s/enet/results.txt' % (results_dir,
                                                                     repeat_idx, xp)
            with open(results_f, 'r') as f:
                for line in f:
                    ls = line.split('\t')
                    if ls[0] == "Number of features selected per fold:":
                        numf_list = [int(x) for x in ls[1].split()]
                        break
                f.close()
            numf_dict['enet'][xp].extend(numf_list)

            
    numf_data = np.array([numf_dict[reg][xp] for (reg, xp) in itertools.product(reg_list, xp_list)])
    numf_data = np.transpose(numf_data)
    num_xp = numf_data.shape[1]
    
    # Plot
    plt.figure(figsize=(10, 5))

    bp = plt.boxplot(numf_data)
    plt.setp(bp['boxes'], color=blue_color)
    plt.setp(bp['whiskers'], color=blue_color)
    plt.setp(bp['fliers'], color=blue_color)
    plt.setp(bp['medians'], color=orange_color)

    plt.title('Cross-validated logistic regression over %d repeats' % num_repeats)
    plt.ylabel('Number of features')
    # labels = ('Nodes (l1)', 'Cnodes (l2)', 'Sfan', 'Edges (Lioness)', 'Edges (Regline)')
    labels = ('%s (%s)' % (reg, xp) for (reg, xp) in itertools.product(reg_list, xp_list))
    plt.xticks(range(1, (num_xp + 1)), labels, rotation=35)    

    means = np.mean(numf_data, axis=0)
    maxes = np.max(numf_data, axis=0)
    # plt.text(0.80, (maxes[0]+90), '%.0f / 12750' % means[0], fontsize='14')
    # plt.text(1.65, (maxes[1]+90), '%.0f / 12750' % means[1], fontsize='14')
    # plt.text(2.65, (maxes[2]+90), '%.0f / 12750' % means[2], fontsize='14')
    # plt.text(3.70, (maxes[3]+90), '%.0f / 45000' % means[3], fontsize='14')
    # plt.text(4.70,(maxes[4]+90), '%.0f / 45000' % means[4], fontsize='14')    
    
    plt.savefig(figure_path, bbox_inches='tight')
    print "Saved number of features to %s" % figure_path


def plot_cixs(results_dir, figure_path, num_repeats=10):
    """ Plot cross-validated consistency indices, averaged over multiple repeats.

    Parameters
    ----------
    results_dir: path
        Path to the directory containing the results.

    figure_path: path
        Path to the file where to save the plot.

    num_repeats: (int, None)
        Number of repeated experiments.        
    """
    cix_dict = {}
    for reg in reg_list:
        cix_dict[reg] = {}
        for xp in xp_list:
            cix_dict[reg][xp] = []
    
    # Read consistency indices
    for repeat_idx in range(num_repeats):
        for xp in xp_list:
            # Lasso experiments
            results_f = '%s/repeat%d/results/%s/results.txt' % (results_dir,
                                                                repeat_idx, xp)
            with open(results_f, 'r') as f:
                for line in f:
                    ls = line.split('\t')
                    if ls[0] == "Stability (Consistency Index):":
                        cix_list = [float(string.lstrip(string.rstrip(x, ",']\n"), ",' [")) \
                                    for x in ls[-1].split(",")]
                        break
                f.close()
            cix_dict['lasso'][xp].extend(cix_list)

            # Enet experiments
            results_f = '%s/repeat%d/results/%s/enet/results.txt' % (results_dir,
                                                                     repeat_idx, xp)
            with open(results_f, 'r') as f:
                for line in f:
                    ls = line.split('\t')
                    if ls[0] == "Stability (Consistency Index):":
                        cix_list = [float(string.lstrip(string.rstrip(x, ",']\n"), ",' [")) \
                                    for x in ls[-1].split(",")]
                        break
                f.close()
            cix_dict['enet'][xp].extend(cix_list)
            
    cix_data = np.array([cix_dict[reg][xp] for (reg, xp) in itertools.product(reg_list, xp_list)])
    cix_data = np.transpose(cix_data)
    num_xp = cix_data.shape[1]

    # Plot
    plt.figure(figsize=(10, 5))

    bp = plt.boxplot(cix_data)
    plt.setp(bp['boxes'], color=blue_color)
    plt.setp(bp['whiskers'], color=blue_color)
    plt.setp(bp['fliers'], color=blue_color)
    plt.setp(bp['medians'], color=orange_color)

    plt.title('Cross-validated logistic regression over %d repeats' % num_repeats)
    plt.ylabel('Consistency Index')
    labels = ('%s (%s)' % (reg, xp) for (reg, xp) in itertools.product(reg_list, xp_list))
    plt.xticks(range(1, (num_xp + 1)), labels, rotation=35)    
    plt.ylim(-0.1, 1.1)
            
    plt.savefig(figure_path, bbox_inches='tight')
    print "Saved consistency indices to %s" % figure_path


def plot_fovs(results_dir, figure_path, num_repeats=10):
    """ Plot cross-validated Fisher overlaps, averaged over multiple repeats.

    Parameters
    ----------
    results_dir: path
        Path to the directory containing the results.

    figure_path: path
        Path to the file where to save the plot.

    num_repeats: (int, None)
        Number of repeated experiments.        
    """
    fov_dict = {}
    for reg in reg_list:
        fov_dict[reg] = {}
        for xp in xp_list:
            fov_dict[reg][xp] = []
    
    # read Fisher overlaps
    for repeat_idx in range(num_repeats):
        for xp in xp_list:
            # Lasso experiments
            results_f = '%s/repeat%d/results/%s/results.txt' % (results_dir,
                                                                repeat_idx, xp)
            with open(results_f, 'r') as f:
                for line in f:
                    ls = line.split('\t')
                    if ls[0] == "Stability (Fisher overlap):":
                        fov_list = [float(string.lstrip(string.rstrip(x, ",']\n"), ",' [")) \
                                    for x in ls[-1].split(",")]
                        break
                f.close()
            fov_dict['lasso'][xp].extend(fov_list)

            # Enet experiments
            results_f = '%s/repeat%d/results/%s/enet/results.txt' % (results_dir,
                                                                     repeat_idx, xp)
            with open(results_f, 'r') as f:
                for line in f:
                    ls = line.split('\t')
                    if ls[0] == "Stability (Fisher overlap):":
                        fov_list = [float(string.lstrip(string.rstrip(x, ",']\n"), ",' [")) \
                                    for x in ls[-1].split(",")]
                        break
                f.close()
            fov_dict['enet'][xp].extend(fov_list)
            
    fov_data = np.array([fov_dict[reg][xp] for (reg, xp) in itertools.product(reg_list, xp_list)])
    fov_data = np.transpose(fov_data)
    num_xp = fov_data.shape[1]

    # Replace np.inf with twice the max (excluding inf)
    fov_data[np.isinf(fov_data)] = 2*np.max(fov_data[np.isfinite(fov_data)])
    
    # Plot
    plt.figure(figsize=(10, 5))

    bp = plt.boxplot(fov_data)
    plt.setp(bp['boxes'], color=blue_color)
    plt.setp(bp['whiskers'], color=blue_color)
    plt.setp(bp['fliers'], color=blue_color)
    plt.setp(bp['medians'], color=orange_color)

    plt.title('Cross-validated logistic regression over %d repeats' % num_repeats)
    plt.ylabel('Fisher Overlap')
    labels = ('%s (%s)' % (reg, xp) for (reg, xp) in itertools.product(reg_list, xp_list))
    plt.xticks(range(1, (num_xp + 1)), labels, rotation=35)    

    plt.savefig(figure_path, bbox_inches='tight')
    print "Saved Fisher overlaps to %s" % figure_path

    
    
def plot_aucs(results_dir, figure_path, num_repeats=10):
    """ Plot cross-validated AUCs, averaged over multiple repeats.

    Parameters
    ----------
    results_dir: path
        Path to the directory containing the results.

    figure_path: path
        Path to the file where to save the plot.

    num_repeats: (int, None)
        Number of repeated experiments.        
    """
    auc_dict = {}
    for reg in reg_list:
        auc_dict[reg] = {}
        for xp in xp_list:
            auc_dict[reg][xp] = []    

    # read AUCs
    for repeat_idx in range(num_repeats):
        for xp in xp_list:
            # Lasso experiments
            results_f = '%s/repeat%d/results/%s/results.txt' % (results_dir,
                                                                repeat_idx, xp)
            with open(results_f, 'r') as f:
                for line in f:
                    ls = line.split()
                    if ls[0] == "AUC:":
                        auc = float(ls[1])
                        break
                f.close()
            auc_dict['lasso'][xp].append(auc)

            # Enet experiments
            results_f = '%s/repeat%d/results/%s/enet/results.txt' % (results_dir,
                                                                     repeat_idx, xp)
            with open(results_f, 'r') as f:
                for line in f:
                    ls = line.split()
                    if ls[0] == "AUC:":
                        auc = float(ls[1])
                        break
                f.close()
            auc_dict['enet'][xp].append(auc)
            
    auc_data = np.array([auc_dict[reg][xp] for (reg, xp) in itertools.product(reg_list, xp_list)])
    auc_data = np.transpose(auc_data)
    num_xp = auc_data.shape[1]

    # Plot AUCs
    plt.figure(figsize=(10, 5))

    bp = plt.boxplot(auc_data)
    plt.setp(bp['boxes'], color=blue_color)
    plt.setp(bp['whiskers'], color=blue_color)
    plt.setp(bp['fliers'], color=blue_color)
    plt.setp(bp['medians'], color=orange_color)

    plt.plot([0, num_xp], [0.74, 0.74], 
             ls='--', color='#666666')
    plt.text(0.75, 0.745, 'FERAL (paper)', color='#666666')

    plt.title('Cross-validated logistic regression over %d repeats' % num_repeats)
    plt.ylabel('AUC')
    labels = ('%s (%s)' % (reg, xp) for (reg, xp) in itertools.product(reg_list, xp_list))
    plt.xticks(range(1, (num_xp + 1)), labels, rotation=35)    
    # plt.ylim(0.63, 0.78)

    plt.savefig(figure_path, bbox_inches='tight')
    print "Saved AUCs to %s" % figure_path

    
def main():
    """ Plot results from cross-validation experiments.

    Example:
        $ python create_plots.py ../outputs/U133A_combat_RFS/KEGG_edges1210/subtype_stratified -r 5
    """
    parser = argparse.ArgumentParser(description="Plot results from CV experiments",
                                     add_help=True)
    parser.add_argument("results_dir", help="Path to results")
    parser.add_argument("-r", "--num_repeats", help="Number of repeats",
                        type=int, default=10)
    args = parser.parse_args()

    # Plots are stored under <results_dir>/results
    # Create folder if it does not exist
    plot_dir = '%s/results' % args.results_dir
    if not os.path.isdir(plot_dir):
        sys.stdout.write("Creating %s\n" % results_dir)
        try: 
            os.makedirs(plot_dir)
        except OSError:
            if not os.path.isdir(plot_dir):
                raise
    
    # AUC
    figure_path = '%s/subtype_stratified_auc.pdf' % plot_dir
    plot_aucs(args.results_dir, figure_path, num_repeats=args.num_repeats)

    # Number of features
    figure_path = '%s/subtype_stratified_numf.pdf' % plot_dir
    plot_numf(args.results_dir, figure_path, num_repeats=args.num_repeats)
    
    # Consistency index
    figure_path = '%s/subtype_stratified_cix.pdf' % plot_dir
    plot_cixs(args.results_dir, figure_path, num_repeats=args.num_repeats)
    
    # Fisher overlap
    figure_path = '%s/subtype_stratified_fov.pdf' % plot_dir
    plot_fovs(args.results_dir, figure_path, num_repeats=args.num_repeats)
    

if __name__ == "__main__":
    main()


