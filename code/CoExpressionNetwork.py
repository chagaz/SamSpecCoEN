# @Author 
# Chloe-Agathe Azencott
# chloe-agathe.azencott@mines-paristech.fr
# July 2016

import argparse
import h5py
import matplotlib.pyplot as plt
#import memory_profiler # call program with flag -m memory_profiler
import numpy as np
import os
import sys
import timeit

sys.path.append('../ACES')
from datatypes.ExpressionDataset import HDF5GroupToExpressionDataset, MakeRandomFoldMap

import utils

orange_color = '#d66000'
blue_color = '#005599'

THRESHOLD = 0.5

class CoExpressionNetwork(object):
    """ Create and manipulate sample-specific co-expression networks.

    Attributes
    ----------
    num_genes: int
        Number of genes in the data.
    num_samples: int
        Number of samples in the data.
        If there are training indices, this is the number of TRAINING samples.
    expression_data: {(num_samples, num_genes) array, (num_tr_samples, num_genes) array}
        Array of gene expression data.
        If there is training data, this is ONLY the training data.
    sample_labels: (num_samples, ) array
        Labels.
        If there is training data, this is ONLY the training data.

    Optional attributes
    ------------------
    refc_data: (num_refc_samples, num_genes) array
        Array of reference gene expression data.
    num_refc_samples: int
        Number of samples in the reference data.
    num_te_samples: {int, None}, optional
        Number of samples used for testing.    
    tr_indices: {(num_tr_samples, ) array, None}, optional
        List of indices to be used for training (if any).
    te_indices: {(num_te_samples, ) array, None}, optional
        List of indices to be used for testing (if any).
    te_expression_data: {(num_te_samples, num_genes) array, None}, optional
        Array of gene expression data for testing.
    te_sample_labels: {(num_te_samples, ) array, None}, optional
        Labels of the genes for testing.
    global_network: (num_genes, num_genes) array, optional
        Upper-triangular adjacency matrix of the global network.
    num_edges: {int, None}, optional
        Number of edges of the global network.
    edges: {(num_edges, 2) array, None}, optional
        List of edges of the global network.
        Each row is an undirected edge, formatted as:
            <index of gene 1> <index of gene 2>
        By convention, the index of gene 1 is smaller than that of gene 2.
    """
    def __init__(self, expression_data, sample_labels,
                 tr_indices=None, te_indices=None, refc_data=None):
        """
        Parameters
        ----------
        expression_data: (self.num_samples, self.num_genes) array
            Array of gene expression data.
        sample_labels: (self.num_samples, ) array
            1D array of sample labels.
        tr_indices: {(self.num_tr_samples, ) array, None}, optional
            1D array of training indices, if any.
        te_indices: {(self.num_te_samples, ) array, None}, optional
            1D array of test indices, if any.
        refc_expression_data: (self.num_refc_samples, self.num_genes) array
            Array of reference gene expression data.
        """
        self.expression_data = expression_data#[:, :800] # TODO Only for testing!
        self.sample_labels = sample_labels
        self.tr_indices = tr_indices
        self.te_indices = te_indices
        self.refc_data = refc_data

        # It does not make sense to have test indices but not train indices
        # Only train indices can be used to work on a subset of the data.
        if not isinstance(self.tr_indices, np.ndarray):
            try:
                assert not isinstance(self.te_indices, np.ndarray)
            except AssertionError:
                sys.stderr.write("You can't have train indices without test indices.\n")
                sys.stderr.write("Aborting.\n")
                sys.exit(-1)
        
        if isinstance(self.tr_indices, np.ndarray):
            if isinstance(self.te_indices, np.ndarray):
                self.num_te_samples = self.te_indices.shape[0]
                self.te_expression_data = self.expression_data[self.te_indices, :]
                self.te_sample_labels = self.sample_labels[self.te_indices]
            self.expression_data = self.expression_data[self.tr_indices, :]
            self.sample_labels = self.sample_labels[self.tr_indices]

        self.num_samples, self.num_genes = self.expression_data.shape
        if isinstance(self.refc_data, np.ndarray):
            self.num_refc_samples, num_genes = self.refc_data.shape
            try:
                assert num_genes == self.num_genes
            except AssertionError:
                sys.stderr.write("Number of genes between reference and data does not match.\n")
                sys.stderr.write("Aborting.\n")
                sys.exit(-1)

        self.global_network = None
        self.edges = None
        self.num_edges = None
        
    #@profile
    def create_global_network(self, threshold, out_path):
        """ Create the global co-expression network **from the reference data**

        This is done by computing, on the reference data, Pearson's correlation over the whole population
        and then thresholding: an edge is kept if its weight (i.e. Pearson's correlation between the
        expression of the 2 genes) is greater than the threshold in this network

        Parameters
        ----------
        threshold: float
            Thresholding value for including an edge in the network.
        out_path: path
            Where to store the edges of the global network

        Modified attributes
        -------------------
        self.global_network: (self.num_genes, self.num_genes) array
            Upper-triangular adjacency matrix of the global network.
        self.num_edges: int
            Number of edges of the global network.
        self.edges: (self.num_edges, 2) array
            List of edges of the global network.

        Created files
        -------------
        out_path: 
            Gzipped file containing the list of edges of the co-expression networks.
            Each line is an undirected edge, formatted as:
                <index of gene 1> <index of gene 2>
            By convention, the index of gene 1 is smaller than that of gene 2.
        """
        # Compute absolute value of Pearson's correlations between genes
        # self.global_network = np.abs(np.corrcoef(np.transpose(self.refc_data)))
        self.global_network = np.abs(np.corrcoef(np.transpose(self.expression_data)))

        # Threshold the network
        self.global_network = np.where((self.global_network > threshold),
                                       self.global_network, 0)
        
        # Only keep the upper triangular matrix (it's symmetric)
        self.global_network[np.tril_indices(self.num_genes)] = 0
        
        sys.stdout.write("A global network of %d edges was constructed.\n" % \
                         np.count_nonzero(self.global_network))

        # List non-zero indices (i.e edges)
        self.edges = np.nonzero(self.global_network)
        self.edges = np.array([self.edges[0], self.edges[1]], dtype='int').transpose()
        self.num_edges = self.edges.shape[0]
        
        # Save edges to file
        np.savetxt(out_path, self.edges, fmt='%d')
        sys.stdout.write("Co-expression network edges saved to %s\n" % out_path)
        
        
    def check_scale_free(self, plot_path=None):
        """ Compute the scale-free criteria (Zhang et al., 2005) for the global network.

        Denoting by k the connectivities of the nodes (number of neighbors),
        the authors recommend that the network be approximately scale-free, i.e.
        (1) the network should have high mean connectivity 
        (2) the slope of the regression line between log10(p(k)) and log10(k) should be close to -1
        (3) the coefficient of determination R2 between log10(p(k)) and log10(k) should be > 0.8

        Parameters
        ----------
        plot_path: {filename, None}, optional
            File where to save the regression plot of log10(freq(connectivities)) against
            log10(connectivities).

        Returns
        -------
        ave_conn: float
            Average connectivity of the nodes of self.global_network.
        slope: float
            Slope of the regression line of log10(freq(connectivities)) against
            log10(connectivities). This should ideally be close to -1.
        r2: float
            Coefficient of determination of the linear regression of log10(freq(connectivities)) against
            log10(connectivities). This should ideally be greater than 0.8.

        
        Displays
        --------
            The regression plot of log10(freq(connectivities)) against log10(connectivities).
            The values relationship should appear linear.

        Reference
        ---------
        B. Zhang and S. Horvath. (2005).
        A General Framework for Weighted Gene Co-Expression Network Analysis.
        Statistical Applications in Genetics and Molecular Biology 4.
        """
        try:
            assert isinstance(self.global_network, np.ndarray)
        except AssertionError:
            sys.stderr.write("The global network has not been computed yet.\n")
            sys.stderr.write("Call create_global_network first.\n")
            sys.stderr.write("Aborting.\n")
            sys.exit(-1)

        connectivities = np.round(np.sum(self.global_network + self.global_network.transpose(), axis=1)) 

        # Compute number of nodes connected (at least one neighbor)
        nnodes = np.shape(np.nonzero(connectivities))[1]
        print nnodes, "genes in the network"
        
        # Compute mean network connectivity
        #ave_conn = np.mean(connectivities)
        ave_conn = np.sum(connectivities) / nnodes

        # Compute log10(connectivites)
        k_values = np.unique(connectivities)
        if k_values[0] == 0:
            k_values = k_values[1:]
        l_k_values = np.log10(k_values)
        
        # Compute log10(freq(connectivities))
        connectivities_bin_count = np.bincount(np.round(connectivities).astype(int))[1:]
        l_freq_k = np.log10(connectivities_bin_count[np.where(connectivities_bin_count > 0)[0]])

        # Fit regression line
        k_w, residuals = np.linalg.lstsq(np.array([ l_k_values, np.ones(len(k_values)) ]).T,
                                         l_freq_k)[:2]
        print k_w
        print residuals
        slope = k_w[0]

        # Compute r2 from residuals
        r2 = 1 - residuals[0] / (l_freq_k.size * l_freq_k.var())
        
        # Regression plot
        fig = plt.figure()
        plt.plot(l_k_values, l_freq_k, "+", color=blue_color)
        plt.plot(l_k_values, k_w[0]*l_k_values + k_w[1], '-', color=orange_color)
        if plot_path:
            fig.savefig(plot_path)
            sys.stdout.write("Scale-free check regression plot saved to %s\n" % plot_path)
        sys.stdout.write("Check visually that the relationship between log10(connectivities) " +\
                         " and log10(freq(connectivities)) is approximately linear\n")
        plt.show()
        
        return ave_conn, slope, r2

        
    def normalize_expression_data(self):
        """ Normalize self.expression_data so that each gene has a mean of 0
        and a standard deviation of 1.
        Normalize reference expression data as well.

        Modified attributes
        -------------------
        self.expression_data:
            self.expression_data is replaced with its normalized version.
        self.expression_data:
            self.refc_data is replaced with its normalized version.
        self.te_expression_data:
            If te_indices, self.te_expression_data is  replaced with its normalized version,
            using the normalization parameters computed on self.trExpression_data.       
        """

        x_mean = np.mean(self.refc_data, axis=0)
        x_stdv = np.std(self.refc_data, axis=0, ddof=1)
        self.refc_data -= x_mean
        self.refc_data /= x_stdv

        self.expression_data -= x_mean
        self.expression_data /= x_stdv
        if isinstance(self.te_indices, np.ndarray):
            # also normalize self.te_expression_data
            print self.te_expression_data.shape
            print x_mean.shape
            self.te_expression_data -= x_mean
            self.te_expression_data /= x_stdv

            
    def create_sam_spec_lioness(self, lioness_path):
        """ Create sample-specific co-expression networks,
        using the LIONESS approach.

        The contribution of x0, y0 to the (x, y) edge is computed as
        num_samples / (num_samples - 1) * (x0 - x_mean)/x_stdv * (y0 - y_mean)/y_stdv
        under the assumption that the number of samples is large
        and that (x0 - x_mean^0) << num_samples
        where x_mean^0 is the average expression of gene x, excluding x0.
        
        Parameters
        ----------
        lioness_path: dirname
            Name of the directory where to write the files describing the co-expression networks.
        
        Created files
        -------------
        edge_weights.gz:
            gzipped file containing the (self.num_edges, self.num_samples) array
            describing the edge weights for each sample (training samples only if self.tr_indices)
        edge_weights_te.gz:
            If te_indices, gzipped file containing the (self.num_edges, self.num_te_samples) array
            describing the edge weights for each test sample.

        Reference
        ---------
        M. L. Kuijjer, M. Tung, G. Yuan, J. Quackenbush and K. Glass (2015).
        Estimating sample-specific regulatory networks.
        arXiv:1505.06440 [q-Bio].
        """
        try:
            assert isinstance(self.global_network, np.ndarray)
        except AssertionError:
            sys.stderr.write("The global network has not been computed yet.\n")
            sys.stderr.write("Call create_global_network first.\n")
            sys.stderr.write("Aborting.\n")
            sys.exit(-1)

        weights_f = '%s/edge_weights.gz' % lioness_path
        if isinstance(self.te_indices, np.ndarray):
            weights_te_f = '%s/edge_weights_te.gz' % lioness_path


        # Assume expression data has been normalized: x_mean = 0 and x_stdv = 1.
        # Compute edge weights
        weights = np.array([np.float(self.num_samples)/np.float(self.num_samples - 1) *\
                            self.expression_data[:, e[0]] * self.expression_data[:, e[1]] \
                            for e in self.edges])
        
        # Save edge weights to file
        np.savetxt(weights_f, weights, fmt='%.5f')
        sys.stdout.write("Lioness edge weights saved to %s\n" % weights_f)

        if isinstance(self.te_indices, np.ndarray):
            # Assume expression data has been normalized: x_mean = 0 and x_stdv = 1.
            # Compute edge weights
            weights = np.array([np.float(self.num_samples)/np.float(self.num_samples - 1) *\
                                self.te_expression_data[:, e[0]] * self.te_expression_data[:, e[1]] \
                                for e in self.edges])
            # Save weight edges to file
            np.savetxt(weights_te_f, weights, fmt='%.5f')
            sys.stdout.write("Lioness edge weights (test data) saved to %s\n" % weights_te_f)
        
        
            
    def create_sam_spec_reglref(self, reglref_path):
        """ Create sample-specific co-expression networks, using the REGLREF approach.

        The weight of the (x0, y0) edge is computed as
        the distance from (x0, y0) to the *reference* regression line between x and y.
        
        Parameters
        ----------
        reglref_path: dirname
            Name of the directory where to write the files describing the co-expression networks.
        
        Created files
        -------------
        reglref_edge_weights.gz:
            gzipped file containing the (self.num_edges, self.num_samples) array
            describing the edge weights for each sample.
        """
        try:
            assert isinstance(self.global_network, np.ndarray)
        except AssertionError:
            sys.stderr.write("The global network has not been computed yet.\n")
            sys.stderr.write("Call create_global_network first.\n")
            sys.stderr.write("Aborting.\n")
            sys.exit(-1)

        weights_f = '%s/reglref_edge_weights.gz' % reglref_path

        # Compute edge weights
        weights = np.ones((self.num_edges, self.num_samples))

        for (edgeIdx, e) in enumerate(self.edges):
            # Fit reference regression line
            reg_w = np.linalg.lstsq(np.array([self.refc_data[:, e[0]],
                                               np.ones(self.num_refc_samples) ]).transpose(),
                                    self.refc_data[:, e[1]])[0]

            # Compute distance from sample to line:
            weights[edgeIdx, :] = np.abs(reg_w[0]*self.expression_data[:, e[0]] - \
                                          self.expression_data[:, e[1]] + reg_w[1]) / \
                np.sqrt(reg_w[0]**2+1)

        # Save edge weights to file
        np.savetxt(weights_f, weights, fmt='%.5f')
        sys.stdout.write("Reglref edge weights saved to %s\n" % weights_f)

            
def run_whole_data(expression_data, refc_expression_data, sample_labels, out_dir):
    """ Build sample-specific REGLREF co-expression networks.

    Parameters
    ----------
    expression_data: (num_samples, num_genes) array
        Array of gene expression data.
    refc_expression_data: (num_refc_samples, num_genes) array
        Array of reference gene expression data.
    sample_labels: (num_samples, ) array
        Labels for the whole data.
    out_dir: path
        Path of the repository where to store the generated networks.

    Created files
    -------------
    out_dir/reglref_edges.gz: 
        Gzipped file containing the list of edges of the co-expression networks.
        Each line is an undirected edge, formatted as:
            <index of gene 1> <index of gene 2>
        By convention, the index of gene 1 is smaller than that of gene 2.
    out_dir/reglref_connectivity.png
        Regression plot of log10(p(connectivities)) against log10(connectivities)
        for the global REGLREF network.
    out_dir/reglref_edge_weights.gz:
        gzipped file containing the (self.num_edges, self.num_samples) array
        describing the edge weights of the REGLREF co-expression networks
        for each sample.
    """
    # Create CoExpressionNetwork instance
    sys.stdout.write("Computing REGLREF network on whole data\n")
    co_expression_net = CoExpressionNetwork(expression_data, sample_labels,
                                            refc_data=refc_expression_data)

    # Normalize the data
    co_expression_net.normalize_expression_data()

    # Create global network
    co_expression_net.create_global_network(THRESHOLD, ("%s/reglref_edges.gz" % out_dir))

    # Check whether the scale-free assumptions are verified
    scalefree_path = '%s/reglref_connectivity.png' % out_dir
    ave_conn, slope, r2 = co_expression_net.check_scale_free(scalefree_path)
    print "Average connectivity: ", ave_conn
    print "Slope (should be close to -1): ", slope
    print "R2 (should be larger than 0.8)", r2
     
    # Build and store co-expression network (REGLREF)
    co_expression_net.create_sam_spec_reglref(out_dir)
    # # (Uncomment to) Time the creation of REGLREF co-expression networks
    # exec_time = timeit.timeit(utils.wrapper(co_expression_net.create_sam_spec_reglref, out_dir),
    #                          number=10)
    # sys.stdout.write("REGLREF network created in %.2f seconds (averaged over 10 repeats)\n" % \
    #                  (exec_time/10))



def run_whole_data_aces(aces_data, refc_data, out_dir):
    """ Build sample-specific REGLREF co-expression networks, from data in ACES format.

    Parameters
    ----------
    aces_data: datatypes.ExpressionDataset.ExpressionDataset
        Data in ACES format, read using HDF5GroupToExpressionDataset.
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
    out_dir/reglref_edge_weights.gz:
        gzipped file containing the (self.num_edges, self.num_samples) array
        describing the edge weights of the REGLREF co-expression networks
        for each sample.
    """
    run_whole_data(aces_data.expressionData, refc_data,
                   aces_data.patientClassLabels, out_dir)


def run_whole_data_lioness(expression_data, sample_labels, out_dir,
                           tr_indices=None, te_indices=None):
    """ Build sample-specific LIONESS co-expression networks.

    If tr_indices is not None, use tr_indices and te_indices to determine train/test samples
    for normalization and network weights parameters.
    Otherwise, build on the entire given dataset.

    Parameters
    ----------
    expression_data: (num_samples, num_genes) array
        Array of gene expression data.
        If there is training data, this is ONLY the training data.
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
    out_dir/global_connectivity.png
        Regression plot of log10(p(connectivities)) against log10(connectivities)
        for the global network.
    out_dir/lioness/edge_weights.gz:
        gzipped file containing the (self.num_edges, self.num_samples) array
        describing the edge weights of the LIONESS co-expression networks
        for each sample (training samples only if self.tr_indices)

    Created files (optional)
    -----------------------
    out_dir/lioness/edge_weights_te.gz:
        gzipped file containing the (self.num_edges, self.num_te_samples) array
        describing the edge weights of the LIONESS co-expression networks
        for each test sample.
    """
    # Create CoExpressionNetwork instance
    if not isinstance(tr_indices, np.ndarray):
        sys.stdout.write("Computing networks on whole data\n")
        co_expression_net = CoExpressionNetwork(expression_data, sample_labels)
    else:
        sys.stdout.write("Computing networks on train / test data\n")
        co_expression_net = CoExpressionNetwork(expression_data, sample_labels,
                                                tr_indices=tr_indices, te_indices=te_indices)

    # Normalize the data
    co_expression_net.normalize_expression_data()

    # Create global network
    co_expression_net.create_global_network(THRESHOLD, ("%s/edges.gz" % out_dir))

    # Check whether the scale-free assumptions are verified
    scalefree_path = '%s/global_connectivity.png' % out_dir
    ave_conn, slope, r2 = co_expression_net.check_scale_free(scalefree_path)
    print "Average connectivity: ", ave_conn
    print "Slope (should be close to -1): ", slope
    print "R2 (should be larger than 0.8)", r2
     
    ###   LIONESS   ###
    # Create repertory in which to store co-expression networks (LIONESS)
    lioness_path = "%s/lioness" % out_dir
    try: 
        os.makedirs(lioness_path)
    except OSError:
        if not os.path.isdir(lioness_path):
            raise
    # Build and store co-expression networks (LIONESS)
    co_expression_net.create_sam_spec_lioness(lioness_path)
    # # (Uncomment to) Time the creation of LIONESS co-expression networks
    # exec_time = timeit.timeit(utils.wrapper(co_expression_net.create_sam_spec_lioness, lioness_path),
    #                          number=10)
    # sys.stdout.write("LIONESS network created in %.2f seconds (averaged over 10 repeats)\n" % \
    #                  (exec_time/10))

    
def run_whole_data_aces_lioness(aces_data, out_dir, tr_indices=None, te_indices=None):
    """ Build sample-specific LIONESS co-expression networks, from data in ACES format.

    If tr_indices is not None, use tr_indices and te_indices to determine train/test samples
    for normalization and network weights parameters.
    Otherwise, build on the entire given dataset.

    Parameters
    ----------
    aces_data: datatypes.ExpressionDataset.ExpressionDataset
        Data in ACES format, read using HDF5GroupToExpressionDataset.
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
    out_dir/lioness/edge_weights.gz:
        gzipped file containing the (self.num_edges, self.num_samples) array
        describing the edge weights of the LIONESS co-expression networks
        for each sample (training samples only if self.tr_indices)

    Created files (optional)
    -----------------------
    out_dir/lioness/edge_weights_te.gz:
        gzipped file containing the (self.num_edges, self.num_te_samples) array
        describing the edge weights of the LIONESS co-expression networks
        for each test sample.
    """
    run_whole_data(aces_data.expressionData, aces_data.patientClassLabels,
                   out_dir, tr_indices, te_indices)


def run_crossval_data_aces_lioness(aces_data, out_dir, num_folds):
    """ Build sample-specific LIONESS co-expression networks, in a cross-validation setting.

    Parameters
    ----------
    aces_data: datatypes.ExpressionDataset.ExpressionDataset
        Data in ACES format, read using HDF5GroupToExpressionDataset.
    out_dir: path
        Path of the repository where to store the generated networks.
    num_folds: int
        Number of cross-validation folds.

    Created files
    -------------
    out_dir/edges.gz: 
        Gzipped file containing the list of edges of the co-expression networks.
        Each line is an undirected edge, formatted as:
            <index of gene 1> <index of gene 2>
        By convention, the index of gene 1 is smaller than that of gene 2.
    For k=1..num_folds:
        out_dir/<k>/global_connectivity.png
            Regression plot of log10(p(connectivities)) against log10(connectivities)
            for the global network.
        out_dir/<k>/train.indices
            List of indices of the training set (one per line).
        out_dir/<k>/train.labels
            List of (0/1) labels of the training set (one per line).
        out_dir/<k>/test.indices
            List of indices of the test set (one per line).
        out_dir/<k>/test.labels
            List of (0/1) labels of the test set (one per line).
        out_dir/<k>/lioness/edge_weights.gz:
            gzipped file containing the (self.num_samples, num_edges) array
            describing the edge weights of the LIONESS co-expression networks
            for the training samples.
        out_dir/<k>/lioness/edge_weights_te.gz:
            gzipped file containing the (self.num_samples, num_edges) array
            describing the edge weights of the LIONESS co-expression networks
            for the test samples.
    """
    # Split the data
    fold_map = MakeRandomFoldMap(aces_data, num_folds, 1)

    for fold_nr in range(num_folds):
        # Get training and test indices
        tr_indices = np.where(np.array(fold_map.foldAssignments)-1 != fold_nr)[0]
        te_indices = np.where(np.array(fold_map.foldAssignments)-1 == fold_nr)[0]

        # Create output directory
        fold_dir = "%s/%s" % (out_dir, fold_nr)
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
        np.savetxt(tr_labels_f, np.array(aces_data.patientClassLabels[tr_indices], dtype='int'),
                   fmt='%d')
        sys.stdout.write("Wrote training labels for fold %d to %s\n" % (fold_nr, tr_labels_f))

        # Save test labels to file
        te_labels_f = '%s/test.labels' % fold_dir
        np.savetxt(te_labels_f, np.array(aces_data.patientClassLabels[te_indices], dtype='int'),
                   fmt='%d')
        sys.stdout.write("Wrote test labels for fold %d to %s\n" % (fold_nr, te_labels_f))

        # Create the networks
        run_whole_data_lioness(aces_data.expressionData,
                               aces_data.patientClassLabels, fold_dir,
                               tr_indices=tr_indices, te_indices=te_indices)

        
def main():
    """ Build sample-specific co-expression networks.

    Example:
        $ python CoExpressionNetwork.py DMFS ../ArrayExpress/postproc ../outputs/U133A_combat_DMFS -k 5 
    """
    parser = argparse.ArgumentParser(description="Build sample-specific co-expression networks",
                                     add_help=True)
    parser.add_argument("dataset_name", help="Dataset name")
    parser.add_argument("refc_dir", help="Where the reference data is stored")
    parser.add_argument("out_dir", help="Where to store generated networks")
    parser.add_argument("-k", "--num_folds", help="Number of cross-validation folds", type=int)
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

    # Get reference expression data
    # Do not normalize the data while loading it (so as not compute the normalization ourselves).
    f = h5py.File("%s/MTAB-62.h5" % args.refc_dir)
    refc_data = HDF5GroupToExpressionDataset(f['MTAB-62'], checkNormalise=False)
    f.close()
    
    # ###   LIONESS   ###
    # if not args.num_folds:
    #     # Compute networks on whole data 
    #     run_whole_data_aces_lioness(aces_data, args.out_dir)
    # else:
    #     # Create cross-validation folds and compute networks on them
    #     run_crossval_data_aces_lioness(aces_data, args.out_dir, args.num_folds)


    ###   REGLREF   ###
    # Reorder reference data so that genes map those in ACES data
    aces_gene_names = aces_data.geneLabels
    refc_gene_names = refc_data.geneLabels
    refc_gene_names = list(refc_gene_names)
    refc_gene_names_dict = dict([(a, ix) for ix, a in enumerate(refc_gene_names)]) # name:idx
    reordered_genes = [refc_gene_names_dict[a] for a in aces_gene_names]
    refc_reordered = np.array(refc_data.expressionData)
    for ix in range(refc_data.expressionData.shape[1]):
        refc_reordered[:, ix] = refc_data.expressionData[:, reordered_genes[ix]]
        
    run_whole_data_aces(aces_data, refc_reordered, args.out_dir)
    

if __name__ == "__main__":
    main()




    