# @Author 
# Chloe-Agathe Azencott
# chloe-agathe.azencott@mines-paristech.fr
# August 2016

import argparse
import copy
import gzip
import h5py
import numpy as np
import os
import sys

sys.path.append('../ACES')
from datatypes.ExpressionDataset import HDF5GroupToExpressionDataset, MakeRandomFoldMap

from CoExpressionNetwork import CoExpressionNetwork
import utils

THRESHOLD = 0.5


def overlap(edges_fname_1, edges_fname_2):
    """ Compute the overlap between two lists of network edges (saved in .gz files).

    Parameters:
    -----------
    edges_fname_1: file name
        Path to first (.gz) list of network edges
    edges_fname_2: file name
        Path to second (.gz) list of network edges

    Return:
    -------
    tanimoto: float
        Normalized number of shared edges.
    """
    # Read first network into dictionary
    edges_dict_1 = {} # index_1:[index_2]
    num_edges_1 = 0
    with gzip.open(edges_fname_1) as f:
        for line in f:
            num_edges_1 += 1
            ls = [int(x) for x in line.split()]
            if not edges_dict_1.has_key(ls[0]):
                edges_dict_1[ls[0]] = [ls[1]]
            else:
                edges_dict_1[ls[0]].append(ls[1])
        f.close()

    # Read second network
    num_edges_in_both = 0 # count edges in both networks
    num_edges_in_oone = 0 # count edges in only one of the two networks
    num_edges_2 = 0
    with gzip.open(edges_fname_2) as f:
        for line in f:
            num_edges_2 += 1
            ls = [int(x) for x in line.split()]
            try:
                edges_dict_1[ls[0]].remove(ls[1])
                num_edges_in_both += 1
            except (KeyError, ValueError):
                # edge2 not in network1
                num_edges_in_oone += 1              
        f.close()

    num_edges_in_oone += np.sum([len(elist) for elist in edges_dict_1.values()])
    print "in both:", num_edges_in_both,
    print "\tin oone:", num_edges_in_oone,
    print "\tin 1:", num_edges_1,
    print "\tin 2:", num_edges_2
    # print "\t%d ?= %d " % ((num_edges_in_both * 2 + num_edges_in_oone), (num_edges_1 + num_edges_2))

    return float(num_edges_in_both) / (float(num_edges_in_both) + float(num_edges_in_oone))
                
                

def main():
    """ Evaluate the stability of the procedure chosen to build co-expression network skeletons:
    - Repeat num_repeats time the procedure (i.e. sampling different 50% of the data)
    - Compute the overlap of the networks obtained.

    Example:
        $ python evaluate_network_stability.py RFS ../outputs/U133A_combat_RFS/repeats -k 10 
    """
    parser = argparse.ArgumentParser(description="Evaluate the stability of co-expression networks",
                                     add_help=True)
    parser.add_argument("dataset_name", help="Dataset name")
    parser.add_argument("out_dir", help="Where to store generated networks")
    parser.add_argument("-k", "--num_repeats", help="Number of repeats", type=int)
    args = parser.parse_args()

    try:
        assert args.dataset_name in ['DMFS', 'RFS', 'SOS']
    except AssertionError:
        sys.stderr.write("dataset_name should be one of 'DMFS', 'RFS', 'SOS'.\n")
        sys.stderr.write("Aborting.\n")
        sys.exit(-1)
    
    # Create out_dir if it does not exist
    if not os.path.isdir(args.out_dir):
        sys.stdout.write("Creating %s\n" % args.out_dir)
        try: 
            os.makedirs(args.out_dir)
        except OSError:
            if not os.path.isdir(args.out_dir):
                raise

    # Get expression data, sample labels.
    # Do not normalize the data while loading it (so as not to use test data for normalization).
    f = h5py.File("../ACES/experiments/data/U133A_combat.h5")
    aces_data = HDF5GroupToExpressionDataset(f['U133A_combat_%s' % args.dataset_name],
                                             checkNormalise=False)
    f.close()


    if False:
        ### To do once before everything (compute full network) ###
        sys.stdout.write("Computing network skeleton (full)\n")
        out_path = "%s/edges.gz" % (args.out_dir)

        num_samples, num_genes = aces_data.expressionData.shape
        global_network = np.abs(np.corrcoef(np.transpose(aces_data.expressionData)))
        global_network = np.where((global_network > THRESHOLD), global_network, 0)
        global_network[np.tril_indices(num_genes)] = 0
        sys.stdout.write("A global network of %d edges was constructed.\n" % \
                         np.count_nonzero(global_network))

        # List non-zero indices (i.e edges)
        edges = np.nonzero(global_network)
        edges = np.array([edges[0], edges[1]], dtype='int').transpose()

        # Save edges to file
        np.savetxt(out_path, edges, fmt='%d')
        sys.stdout.write("Co-expression network edges saved to %s\n" % out_path)
        del edges
        ######

        
    if False:
    # Create the skeletons according to the procedure
        for repeat in range(args.num_repeats):
            sys.stdout.write("Computing network skeleton (repeat %d)\n" % repeat)
            out_path = "%s/edges_%d.gz" % (args.out_dir, repeat)

            # Initialize a CoExpressionNetwork object
            co_expression_net = CoExpressionNetwork(aces_data.expressionData,
                                                    aces_data.patientClassLabels)

            # Create the network skeleton
            co_expression_net.create_global_network(THRESHOLD, out_path)
        
    # Compare the skeletons
    # tanimotos = []
    # for idx1 in range(args.num_repeats):
    #     out_path_1 = "%s/edges_%d.gz" % (args.out_dir, idx1)
    #     for idx2 in range(idx1+1, args.num_repeats):
    #         out_path_2 = "%s/edges_%d.gz" % (args.out_dir, idx2)
    #         tanimotos.append(overlap(out_path_1, out_path_2))
    # print tanimotos
    # print "Average tanimoto:", np.mean(tanimotos)

    partial_not_full = [] # proportion of edges of partial network not in full network
    full_not_partial = [] # proportion of edges of full network not in partial network

    # Read full network into dictionary
    edges_fname = "%s/edges.gz" % (args.out_dir)
    edges_dict = {} # index_1:[index_2]
    num_edges = 0
    with gzip.open(edges_fname) as f:
        for line in f:
            num_edges += 1
            ls = [int(x) for x in line.split()]
            if not edges_dict.has_key(ls[0]):
                edges_dict[ls[0]] = [ls[1]]
            else:
                edges_dict[ls[0]].append(ls[1])
        f.close()

        
    # Compare networks built on partial data
    for idx2 in range(args.num_repeats):
        edges_fname_2 = "%s/edges_%d.gz" % (args.out_dir, idx2)
        # Read second network
        pnf = 0 # count edges in partial not full
        num_edges_2 = 0

        ref_edges_dict = copy.deepcopy(edges_dict)
        
        with gzip.open(edges_fname_2) as f:
            for line in f:
                num_edges_2 += 1
                ls = [int(x) for x in line.split()]
                try:
                    ref_edges_dict[ls[0]].remove(ls[1])
                except (KeyError, ValueError):
                    # edge2 not in network1
                    pnf += 1              
            f.close()
        pnf = float(pnf) / num_edges_2

        nfp = float(np.sum([len(elist) for elist in ref_edges_dict.values()])) / num_edges
        
        print "pnf: %.3f" %  pnf,
        print "nfp: %.3f" % nfp

        partial_not_full.append(pnf)
        full_not_partial.append(nfp)

    print "spurious edges from partial data:\t", np.mean(partial_not_full)
    print "edges undetected from partial data:\t", np.mean(full_not_partial)


        
            
if __name__ == "__main__":
    main()