# @Author 
# Chloe-Agathe Azencott
# chloe-agathe.azencott@mines-paristech.fr
# October 2016

import argparse
import h5py
import matplotlib.pyplot as plt
# import memory_profiler # call program with flag -m memory_profiler
import numpy as np
import os
import sys
import timeit

sys.path.append('../ACES/')
sys.path.append('/share/data40T/chloe/SamSpecCoEN/ACES/')
from datatypes.ExpressionDataset import HDF5GroupToExpressionDataset, MakeRandomFoldMap

import utils

THRESHOLD = 0.75 # threshold for correlation values

plot_regline = False # turn to True to visualize the cloud of points (gene 1 vs gene 2)

orange_color = '#d66000'
blue_color = '#005599'

class CoExpressionNetwork(object):
    """ Create and manipulate sample-specific co-expression networks.

    Attributes
    ----------
    num_genes: int
        Number of genes in the data.
    num_samples: int
        Number of samples in the data.
        If there are training indices, this is the number of TRAINING samples.
    expression_data: (num_samples, num_genes) array
        Array of gene expression data.
        If there is training data, this is ONLY the training data.
    refc_data: (num_refc_samples, num_genes) array
        Array of reference gene expression data.
    gene_names: (num_genes, ) list
        List of gene names (same order as the data).
    sample_labels: (num_samples, ) array
        Labels.
        If there is training data, this is ONLY the training data.

    Optional attributes
    ------------------
    ntwk_skeleton: (num_genes, num_genes) array, optional
        Upper-triangular adjacency matrix of the global network.
    num_edges: {int, None}, optional
        Number of edges of the global network.
    edges: {(num_edges, 2) array, None}, optional
        List of edges of the global network.
        Each row is an undirected edge, formatted as:
            <index of gene 1> <index of gene 2>
        By convention, the index of gene 1 is smaller than that of gene 2.
    """
    def __init__(self, expression_data, sample_labels, refc_data, gene_names):
        """
        Parameters
        ----------
        expression_data: (self.num_samples, self.num_genes) array
            Array of gene expression data.
        sample_labels: (self.num_samples, ) array
            1D array of sample labels.
        refc_expression_data: (self.num_refc_samples, self.num_genes) array
            Array of reference gene expression data.
        """
        self.expression_data = expression_data#[:, :800] # TODO Only for testing!
        self.sample_labels = sample_labels
        self.refc_data = refc_data
        self.gene_names = list(gene_names)

        self.num_samples, self.num_genes = self.expression_data.shape

        # Check the reference and actual data do match 
        self.num_refc_samples, num_genes = self.refc_data.shape
        try:
            assert num_genes == self.num_genes
        except AssertionError:
            sys.stderr.write("Number of genes between reference and data does not match.\n")
            sys.stderr.write("Aborting.\n")
            sys.exit(-1)        

        self.ntwk_skeleton = None
        self.edges = None
        self.num_edges = None
        
        
    def normalize_expression_data(self):
        """ Normalize self.expression_data so that each gene has a mean of 0
        and a standard deviation of 1.

        Modified attributes
        -------------------
        self.expression_data:
            self.expression_data is replaced with its normalized version.
        self.refc_data:
            self.refc_data is replaced with its normalized version.
        """
        # Normalize refc data
        x_mean = np.mean(self.refc_data, axis=0)
        x_stdv = np.std(self.refc_data, axis=0, ddof=1)
        self.refc_data -= x_mean
        self.refc_data /= x_stdv        

        x_mean = np.mean(self.expression_data, axis=0)
        x_stdv = np.std(self.expression_data, axis=0, ddof=1)
        self.expression_data -= x_mean
        self.expression_data /= x_stdv



    def read_ntwk_skeleton(self, ppi_path, out_path):
        """ Read network skeleton.

         Parameters
        ----------
        ppi_path: path
            Path to PPI data (in sif format) to use.
        out_path: path
            Where to store the edges of the global network.
        
        Modified attributes
        -------------------
        self.ntwk_skeleton: (self.num_genes, self.num_genes) array
            Upper-triangular adjacency matrix of the global network.
        self.num_edges: int
            Number of edges of the network.
        self.edges: (self.num_edges, 2) array
            List of edges of the network.
        
        Created files
        -------------
        out_path/edges.gz: 
            Gzipped file containing the list of edges of the PPI network.
            Each line is an undirected edge, formatted as:
                <index of gene 1> <index of gene 2>
            By convention, the index of gene 1 is smaller than that of gene 2.
        """
        
        # Load set of edges
        edges_set = set([]) # (gene_idx_1, gene_idx_2)
        # gene_idx_1 < gene_idx_2
        # idx in aces_gene_names, starting at 0
        with open(ppi_path) as f:
            for line in f:
                ls = line.split()
                gene_name_1 = 'Entrez_%s' % ls[0]
                gene_name_2 = 'Entrez_%s' % ls[2]
                # Exclude self edges
                if gene_name_1 == gene_name_2:
                    continue 
                try:
                    gene_idx_1 = self.gene_names.index(gene_name_1)  
                    gene_idx_2 = self.gene_names.index(gene_name_2)
                except ValueError:
                    continue
                if gene_idx_1 < gene_idx_2:
                    e = (gene_idx_1, gene_idx_2)
                else:
                    e = (gene_idx_2, gene_idx_1)
                edges_set.add(e)
        f.close()  
        edges_list = np.array(list(edges_set))
        self.num_edges = len(edges_list)

        # Restrict CoExpressionNetwork data to the genes that are in the network
        genes_in_network = list(set(np.array([list(e) for e in edges_list]).flatten()))
        genes_in_network.sort()
        self.expression_data = self.expression_data[:, genes_in_network]
        self.num_genes = len(genes_in_network)

        # Create self.edges
        # that is, edges in format [idx_1, idx_2]
        # where idx_1, idx_2 are indices in the new self.expression_data
        # Currently, in edges_list, idx_1, idx_2 are indices in self.gene_names
        self.edges = np.array([[genes_in_network.index(e[0]),
                               genes_in_network.index(e[1])] for e in edges_list])
        
        # Create self.ntwk_skeleton (upper triangular) from edges_list
        self.ntwk_skeleton = np.zeros((self.num_genes, self.num_genes))
        for e in self.edges:
            if e[0] < e[1]:
                self.ntwk_skeleton[e[0], e[1]] = 1
            else:
                self.ntwk_skeleton[e[1], e[0]] = 1
        
        sys.stdout.write("A network skeleton of %d edges was processed.\n" % \
                         np.count_nonzero(self.ntwk_skeleton))

        # Save edges to file
        edges_f = '%s/edges.gz' % out_path
        np.savetxt(edges_f, self.edges, fmt='%d')
        sys.stdout.write("Network skeleton edges saved to %s\n" % edges_f)
        
            
    def create_sam_spec_regline(self, regline_path):
        """ Create sample-specific co-expression networks,
        using the REGLINE approach.

        The weight of the (x0, y0) edge is computed as
        the distance from (x0, y0) to the regression line between x and y.

        Use reference data to compute the line.
        
        Parameters
        ----------
        regline_path: dirname
            Name of the directory where to write the files describing the co-expression networks.
        
        Created files
        -------------
        edge_weights.gz:
            gzipped file containing the (self.num_edges, self.num_samples) array
            describing the edge weights for each sample.
        """
        try:
            assert isinstance(self.ntwk_skeleton, np.ndarray)
        except AssertionError:
            sys.stderr.write("The network skeleton has not been read yet.\n")
            sys.stderr.write("Call read_ntwk_skeleton first.\n")
            sys.stderr.write("Aborting.\n")
            sys.exit(-1)

        weights_f = '%s/edge_weights.gz' % regline_path

        # Compute sample-specific edge weights
        weights = np.ones((self.num_edges, self.num_samples))
        for (edge_idx, e) in enumerate(self.edges):
            if plot_regline:
                print "edge", e[0], e[1]
            # Fit regression line to reference data
            n = self.refc_data.shape[0]
            reg_w = np.linalg.lstsq(np.array([ self.refc_data[:, e[0]],
                                               np.ones(n) ]).transpose(),
                                    self.refc_data[:, e[1]])[0]
            if plot_regline:
                print reg_w[0], reg_w[1]

            # Compute distances from samples to line:
            weights[edge_idx, :] = np.abs(reg_w[0]*self.expression_data[:, e[0]] - \
                                          self.expression_data[:, e[1]] + reg_w[1]) / \
                np.sqrt(reg_w[0]**2+1)

            if plot_regline:
                print "max weight", np.max(weights[edge_idx, :])
                print np.where(weights[edge_idx, :]==np.max(weights[edge_idx, :]))
                s_idx = np.where(weights[edge_idx, :]==np.max(weights[edge_idx, :]))[0][0]
                print "s_idx", s_idx
                print "distance", weights[edge_idx, s_idx]

                plt.figure(figsize=(5, 5))
                plt.scatter(self.expression_data[:, e[0]], self.expression_data[:, e[1]],
                            marker="+")
                plt.plot([-5, 5], [(reg_w[0]*(-5) + reg_w[1]),
                                   (reg_w[0]*(5) + reg_w[1])])
            
                x0 = self.expression_data[s_idx, e[0]]
                y0 = self.expression_data[s_idx, e[1]]
                plt.scatter(x0, y0, color='orange', marker="o")                        
                a = (reg_w[0] * (y0 - reg_w[1]) + x0)/(1 + reg_w[0]**2)
                b = (reg_w[1] + reg_w[0] * (x0 + reg_w[0] * y0))/(1 + reg_w[0]**2)
                plt.plot([x0, a], [y0, b], ls='-', color='orange')
                plt.text(0.5*(x0+a), 0.5*(y0+b), 'd', color='orange')

                print "x0", x0, "\ty0", y0, "\ta", a, "\tb", b

                plt.axis([-5, 5, -5, 5])
                plt.show()
                sys.exit(0)

        # Save edge weights to file
        np.savetxt(weights_f, weights, fmt='%.5f')
        sys.stdout.write("Regline edge weights saved to %s\n" % weights_f)

            
    def create_sam_spec_sum(self, sum_path):
        """ Create sample-specific co-expression networks,
        using the SUM approach.

        The weight of the (x0, y0) edge is computed as
        the sum of x0 and y0.
        
        Parameters
        ----------
        sum_path: dirname
            Name of the directory where to write the files describing the co-expression networks.
        
        Created files
        -------------
        edge_weights.gz:
            gzipped file containing the (self.num_edges, self.num_samples) array
            describing the edge weights for each sample.
        """
        try:
            assert isinstance(self.ntwk_skeleton, np.ndarray)
        except AssertionError:
            sys.stderr.write("The network skeleton has not been read yet.\n")
            sys.stderr.write("Call read_ntwk_skeleton first.\n")
            sys.stderr.write("Aborting.\n")
            sys.exit(-1)

        weights_f = '%s/edge_weights.gz' % sum_path

        # Compute edge weights
        weights = np.ones((self.num_edges, self.num_samples))

        for (edge_idx, e) in enumerate(self.edges):
            # Compute sum of node weights:
            weights[edge_idx, :] = e[0] + e[1]
            
        # Save edge weights to file
        np.savetxt(weights_f, weights, fmt='%.5f')
        sys.stdout.write("SUM edge weights saved to %s\n" % weights_f)

            
    def create_sam_spec_euclide(self, euclide_path):
        """ Create sample-specific co-expression networks,
        using the EUCLIDE approach.

        The weight of the (x0, y0) edge is computed as
        the euclidean distance between x0 and y0.
        
        Parameters
        ----------
        euclide_path: dirname
            Name of the directory where to write the files describing the co-expression networks.
        
        Created files
        -------------
        edge_weights.gz:
            gzipped file containing the (self.num_edges, self.num_samples) array
            describing the edge weights for each sample.
        """
        try:
            assert isinstance(self.ntwk_skeleton, np.ndarray)
        except AssertionError:
            sys.stderr.write("The network skeleton has not been read yet.\n")
            sys.stderr.write("Call read_ntwk_skeleton first.\n")
            sys.stderr.write("Aborting.\n")
            sys.exit(-1)

        weights_f = '%s/edge_weights.gz' % euclide_path

        # Compute edge weights
        weights = np.ones((self.num_edges, self.num_samples))

        for (edge_idx, e) in enumerate(self.edges):
            # Compute euclide distance between node weights:
            weights[edge_idx, :] = 0.5 * np.sqrt(e[0]**2 + e[1]**2)
            
        # Save edge weights to file
        np.savetxt(weights_f, weights, fmt='%.5f')
        sys.stdout.write("EUCLIDE edge weights saved to %s\n" % weights_f)

            
    def create_sam_spec_euclthr(self, euclthr_path):
        """ Create sample-specific co-expression networks,
        using the EUCLTHR approach.

        The weight of the (x0, y0) edge is computed as
        the euclidean distance between x0 and y0 if both x0 and y0 > 0
        and 0 otherwise.
        
        Parameters
        ----------
        euclthr_path: dirname
            Name of the directory where to write the files describing the co-expression networks.
        
        Created files
        -------------
        edge_weights.gz:
            gzipped file containing the (self.num_edges, self.num_samples) array
            describing the edge weights for each sample.
        """
        try:
            assert isinstance(self.ntwk_skeleton, np.ndarray)
        except AssertionError:
            sys.stderr.write("The network skeleton has not been read yet.\n")
            sys.stderr.write("Call read_ntwk_skeleton first.\n")
            sys.stderr.write("Aborting.\n")
            sys.exit(-1)

        weights_f = '%s/edge_weights.gz' % euclthr_path

        # Compute edge weights
        weights = np.ones((self.num_edges, self.num_samples))

        for (edge_idx, e) in enumerate(self.edges):
            # Compute euclide distance between node weights:
            weights[edge_idx, :] = np.where((e[0]>0) * (e[1]>0), 0.5 * np.sqrt(e[0]**2 + e[1]**2), 0)
            
        # Save edge weights to file
        np.savetxt(weights_f, weights, fmt='%.5f')
        sys.stdout.write("EUCLTHR edge weights saved to %s\n" % weights_f)

            

def run_whole_data(expression_data, sample_labels, gene_names,
                   ppi_path, reference_data, out_dir):
    """ Compute sample-specific edge weights co-expression networks.

    Parameters
    ----------
    expression_data: (num_samples, num_genes) array
        Array of gene expression data.
        If there is training data, this is ONLY the training data.
    reference_data: (num_refc_samples, num_genes) array
        Array of reference gene expression data.
    gene_names: (num_genes, ) list
        List of gene names (same order as the data).
    ppi_path: path
        Path of the .sif file containing the PPI network to use.
    sample_labels: (num_samples, ) array
        Labels for the whole data.
    out_dir: path
        Path of the repository where to store the generated networks.

    Created files
    -------------
    out_dir/edges.gz: 
        Gzipped file containing the list of edges of the co-expression networks.
        Each line is an undirected edge, formatted as:
            <index of gene 1> <index of gene 2>
        By convention, the index of gene 1 is smaller than that of gene 2.
    out_dir/regline/edge_weights.gz:
        gzipped file containing the (self.num_edges, self.num_samples) array
        describing the edge weights of the Regline co-expression networks
        for each sample.
    out_dir/sum/edge_weights.gz:
        gzipped file containing the (self.num_edges, self.num_samples) array
        describing the edge weights of the 'sum' co-expression networks
        for each sample.
    out_dir/euclide/edge_weights.gz:
        gzipped file containing the (self.num_edges, self.num_samples) array
        describing the edge weights of the 'euclide' co-expression networks
        for each sample.
    out_dir/euclthr/edge_weights.gz:
        gzipped file containing the (self.num_edges, self.num_samples) array
        describing the edge weights of the 'euclide_th' co-expression networks
        for each sample.

    """
    # Create CoExpressionNetwork instance
    sys.stdout.write("Computing networks on whole data\n")
    co_expression_net = CoExpressionNetwork(expression_data, sample_labels,
                                            reference_data, gene_names)

    # Normalize the data
    co_expression_net.normalize_expression_data()

    # Read network skeleton
    co_expression_net.read_ntwk_skeleton(ppi_path, out_dir)
    
    # Create repertory in which to store co-expression networks (REGLINE)
    regline_path = "%s/regline" % out_dir
    try: 
        os.makedirs(regline_path)
    except OSError:
        if not os.path.isdir(regline_path):
            raise
    # Compute and store edge weights (REGLINE)
    co_expression_net.create_sam_spec_regline(regline_path)

    
    # Create repertory in which to store co-expression networks (SUM)
    sum_path = "%s/sum" % out_dir
    try: 
        os.makedirs(sum_path)
    except OSError:
        if not os.path.isdir(sum_path):
            raise
    # Compute and store edge weights (SUM)
    co_expression_net.create_sam_spec_sum(sum_path)

    
    # Create repertory in which to store co-expression networks (EUCLIDE)
    euclide_path = "%s/euclide" % out_dir
    try: 
        os.makedirs(euclide_path)
    except OSError:
        if not os.path.isdir(euclide_path):
            raise
    # Compute and store edge weights (EUCLIDE)
    co_expression_net.create_sam_spec_euclide(euclide_path)

    
    # Create repertory in which to store co-expression networks (EUCLTHR)
    euclthr_path = "%s/euclthr" % out_dir
    try: 
        os.makedirs(euclthr_path)
    except OSError:
        if not os.path.isdir(euclthr_path):
            raise
    # Compute and store edge weights (EUCLTHR)
    co_expression_net.create_sam_spec_euclthr(euclthr_path)
    
    
def run_whole_data_aces(aces_data, ppi_path, refc_data, out_dir):
    """ Build sample-specific co-expression networks, from data in ACES format.

    If tr_indices is not None, use tr_indices and te_indices to determine train/test samples
    for normalization and network weights parameters.
    Otherwise, build on the entire given dataset.

    Parameters
    ----------
    aces_data: datatypes.ExpressionDataset.ExpressionDataset
        Data in ACES format, read using HDF5GroupToExpression_dataset.
    ppi_path: path
        Path of the .sif file containing the PPI network to use.
    refc_data: (num_refc_samples, num_genes) array
        Reference gene expression data.    
    out_dir: path
        Path of the repository where to store the generated networks.

    Created files
    -------------
    out_dir/edges.gz: 
        Gzipped file containing the list of edges of the co-expression networks.
        Each line is an undirected edge, formatted as:
            <index of gene 1> <index of gene 2>
        By convention, the index of gene 1 is smaller than that of gene 2.
    out_dir/global_connectivity.png
        Regression plot of log10(p(connectivities)) against log10(connectivities)
        for the global network.
    out_dir/regline/edge_weights.gz:
        gzipped file containing the (self.num_edges, self.num_samples) array
        describing the edge weights of the Regline co-expression networks
        for each sample (training samples only if self.tr_indices)
    """
    run_whole_data(aces_data.expressionData, aces_data.patientClassLabels,
                   aces_data.geneLabels, ppi_path, refc_data, out_dir)
                   


def main():
    """ Build sample-specific co-expression networks.

    Example:
        $ python CoExpressionNetwork.py RFS ../ACES/experiments/data/KEGG_edges1210.sif ../ArrayExpress/postproc/MTAB-62.h5 ../outputs/U133A_combat_RFS 
    """
    parser = argparse.ArgumentParser(description="Build sample-specific co-expression networks",
                                     add_help=True)
    parser.add_argument("dataset_name",
                        help="Dataset name")
    parser.add_argument("ppi",
                        help="PPI network")
    parser.add_argument("refc_data",
                        help="Reference data for network construction")
    parser.add_argument("out_dir",
                        help="Where to store generated edge weights")
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
    f = h5py.File("../ACES/experiments/data/U133A_combat.h5", "r")
    aces_data = HDF5GroupToExpressionDataset(f['U133A_combat_%s' % args.dataset_name],
                                            checkNormalise=False)
    f.close()

    # Reference expression data
    f = h5py.File(args.refc_data, "r")
    refc_data = HDF5GroupToExpressionDataset(f['MTAB-62'], checkNormalise=False)
    f.close()

    # Reorder reference data so that genes map those in ACES data
    aces_gene_names = aces_data.geneLabels
    refc_gene_names = refc_data.geneLabels
    refc_gene_names = list(refc_gene_names)
    refc_gene_names_dict = dict([(a, ix) for ix, a in enumerate(refc_gene_names)]) # name:idx
    reordered_genes = [refc_gene_names_dict[a] for a in aces_gene_names]
    refc_reordered = np.array(refc_data.expressionData)
    for ix in range(refc_data.expressionData.shape[1]):
        refc_reordered[:, ix] = refc_data.expressionData[:, reordered_genes[ix]]

    # Compute network edges
    run_whole_data_aces(aces_data, args.ppi, refc_reordered, args.out_dir)
    

if __name__ == "__main__":
    main()
