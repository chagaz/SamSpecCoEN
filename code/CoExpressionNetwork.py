# @Author 
# Chloe-Agathe Azencott
# chloe-agathe.azencott@mines-paristech.fr
# January 2016

import argparse
import h5py
import matplotlib.pyplot as plt
import memory_profiler # call program with flag -m memory_profiler
import numpy as np
import os
import sys
import timeit

sys.path.append('ACES')
from datatypes.ExpressionDataset import HDF5GroupToExpressionDataset, MakeRandomFoldMap

import utils

orange_color = '#d66000'
blue_color = '#005599'

class CoExpressionNetwork(object):
    """ Create and manipulate sample-specific co-expression networks.

    Attributes
    ----------
    numGenes: int
        Number of genes in the data.
    numSamples: int
        Number of samples in the data.
        If there are training indices, this is the number of TRAINING samples.
    expressionData: {(numSamples, numGenes) array, (numTrSamples, numGenes) array}
        Array of gene expression data.
        If there is training data, this is ONLY the training data.
    sampleLabels: (numSamples, ) array
        Labels.
        If there is training data, this is ONLY the training data.

    Optional attributes
    ------------------
    numTeSamples: {int, None}, optional
        Number of samples used for testing.    
    trIndices: {(numTrSamples, ) array, None}, optional
        List of indices to be used for training (if any).
    teIndices: {(numTeSamples, ) array, None}, optional
        List of indices to be used for testing (if any).
    teExpressionData: {(numTeSamples, numGenes) array, None}, optional
        Array of gene expression data for testing.
    teSampleLabels: {(numTeSamples, ) array, None}, optional
        Labels of the genes for testing.
    globalNetwork: (numGenes, numGenes) array, optional
        Upper-triangular adjacency matrix of the global network.
    numEdges: {int, None}, optional
        Number of edges of the global network.
    edges: {(numEdges, 2) array, None}, optional
        List of edges of the global network.
        Each row is an undirected edge, formatted as:
            <index of gene 1> <index of gene 2>
        By convention, the index of gene 1 is smaller than that of gene 2.
    """
    def __init__(self, expressionData, sampleLabels, trIndices=None, teIndices=None):
        """
        Parameters
        ----------
        expressionData: (self.numSamples, self.numGenes) array
            Array of gene expression data.
        sampleLabels: (self.numSamples, ) array
            1D array of sample labels.
        trIndices: {(self.numTrSamples, ) array, None}, optional
            1D array of training indices, if any.
        teIndices: {(self.numTeSamples, ) array, None}, optional
            1D array of test indices, if any.
        """
        self.expressionData = expressionData#[:, :800] # TODO Only for testing!
        self.sampleLabels = sampleLabels
        self.trIndices = trIndices
        self.teIndices = teIndices

        # It does not make sense to have test indices but not train indices
        # Only train indices can be used to work on a subset of the data.
        if not isinstance(self.trIndices, np.ndarray):
            try:
                assert not isinstance(self.teIndices, np.ndarray)
            except AssertionError:
                sys.stderr.write("You can't have train indices without test indices.\n")
                sys.stderr.write("Aborting.\n")
                sys.exit(-1)
        
        if isinstance(self.trIndices, np.ndarray):
            if isinstance(self.teIndices, np.ndarray):
                self.numTeSamples = self.teIndices.shape[0]
                self.teExpressionData = self.expressionData[self.teIndices, :]
                self.teSampleLabels = self.sampleLabels[self.teIndices]
            self.expressionData = self.expressionData[self.trIndices, :]
            self.sampleLabels = self.sampleLabels[self.trIndices]

        self.numSamples, self.numGenes = self.expressionData.shape

        self.globalNetwork = None
        self.edges = None
        self.numEdges = None
        
    #@profile
    def createGlobalNetwork(self, threshold, outPath):
        """ Create the global (population-wise) co-expression network.

        This is done by computing (on the training set if any) Pearson's correlation over:
        - the whole population
        - the positive samples
        - the negative samples
        and then thresholding.
        An edge is kept if its weight (i.e. Pearson's correlation between the expression of the 2 genes)
        is greater than the threshold in any of the three networks.

        Parameters
        ----------
        threshold: float
            Thresholding value for including an edge in the network.
        outPath: path
            Where to store the edges of the global network

        Modified attributes
        -------------------
        self.globalNetwork: (self.numGenes, self.numGenes) array
            Upper-triangular adjacency matrix of the global network.
        self.numEdges: int
            Number of edges of the global network.
        self.edges: (self.numEdges, 2) array
            List of edges of the global network.

        Created files
        -------------
        outPath/edges.gz: 
            Gzipped file containing the list of edges of the co-expression networks.
            Each line is an undirected edge, formatted as:
                <index of gene 1> <index of gene 2>
            By convention, the index of gene 1 is smaller than that of gene 2.
        """
        # Restrict the data to the positive samples
        Xpos = self.expressionData[np.where(self.sampleLabels)[0], :]
        
        # Restrict the data to the negative samples
        Xneg = self.expressionData[np.where(np.logical_not(self.sampleLabels))[0], :]
        
        # Compute Pearson's correlation, gene by gene
        self.globalNetwork = np.corrcoef(np.transpose(self.expressionData))

        # Threshold the network
        self.globalNetwork = np.where(np.logical_or(np.logical_or(self.globalNetwork > threshold,
                                                                  np.corrcoef(np.transpose(Xneg)) > threshold),
                                                    np.corrcoef(np.transpose(Xpos)) > threshold),
                                      self.globalNetwork, 0)

        # Only keep the upper triangular matrix (it's symmetric)
        self.globalNetwork[np.tril_indices(self.numGenes)] = 0
        
        sys.stdout.write("A global network of %d edges was constructed.\n" % \
                         np.count_nonzero(self.globalNetwork))

        # Save edges to file
        edgesF = '%s/edges.gz' % outPath

        # List non-zero indices (i.e edges)
        self.edges = np.nonzero(self.globalNetwork)
        self.edges = np.array([self.edges[0], self.edges[1]], dtype='int').transpose()
        self.numEdges = self.edges.shape[0]
        
        # Save edges to file
        np.savetxt(edgesF, self.edges, fmt='%d')
        sys.stdout.write("Co-expression network edges saved to %s\n" % edgesF)
        
        
    def checkScaleFree(self, plotPath=None):
        """ Compute the scale-free criteria (Zhang et al., 2005) for the global network.

        Denoting by k the connectivities of the nodes (number of neighbors),
        the authors recommend that the network be approximately scale-free, i.e.
        (1) the network should have high mean connectivity 
        (2) the slope of the regression line between log10(p(k)) and log10(k) should be close to -1
        (3) the coefficient of determination R2 between log10(p(k)) and log10(k) should be > 0.8

        Parameters
        ----------
        plotPath: {filename, None}, optional
            File where to save the regression plot of log10(freq(connectivities)) against
            log10(connectivities).

        Returns
        -------
        aveConn: float
            Average connectivity of the nodes of self.globalNetwork.
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
            assert isinstance(self.globalNetwork, np.ndarray)
        except AssertionError:
            sys.stderr.write("The global network has not been computed yet.\n")
            sys.stderr.write("Call createGlobalNetwork first.\n")
            sys.stderr.write("Aborting.\n")
            sys.exit(-1)

        connectivities = np.round(np.sum(self.globalNetwork + self.globalNetwork.transpose(), axis=1)) 

        # Compute mean network connectivity
        aveConn = np.mean(connectivities)

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
        if plotPath:
            fig.savefig(plotPath)
            sys.stdout.write("Scale-free check regression plot saved to %s\n" % plotPath)
        sys.stdout.write("Check visually that the relationship between log10(connectivities) " +\
                         " and log10(freq(connectivities)) is approximately linear\n")
        plt.show()
        
        return aveConn, slope, r2

        
    def normalizeExpressionData(self):
        """ Normalize self.expressionData so that each gene has a mean of 0
        and a standard deviation of 1.

        Modified attributes
        -------------------
        self.expressionData:
            self.expressionData is replaced with its normalized version.
        self.teExpressionData:
            If teIndices, self.teExpressionData is  replaced with its normalized version,
            using the normalization parameters computed on self.trExpressionData.       
        """

        xMean = np.mean(self.expressionData, axis=0)
        xStdv = np.std(self.expressionData, axis=0, ddof=1)
        self.expressionData -= xMean
        self.expressionData /= xStdv
        if isinstance(self.teIndices, np.ndarray):
            # also normalize self.teExpressionData
            print self.teExpressionData.shape
            print xMean.shape
            self.teExpressionData -= xMean
            self.teExpressionData /= xStdv
            

    def createSamSpecLioness(self, lionessPath):
        """ Create sample-specific co-expression networks,
        using the LIONESS approach.

        The contribution of x0, y0 to the (x, y) edge is computed as
        numSamples / (numSamples - 1) * (x0 - x_mean)/x_stdv * (y0 - y_mean)/y_stdv
        under the assumption that the number of samples is large
        and that (x0 - x_mean^0) << numSamples
        where x_mean^0 is the average expression of gene x, excluding x0.
        
        Parameters
        ----------
        lionessPath: dirname
            Name of the directory where to write the files describing the co-expression networks.
        
        Created files
        -------------
        edge_weights.gz:
            gzipped file containing the (self.numEdges, self.numSamples) array
            describing the edge weights for each sample (training samples only if self.trIndices)
        edge_weights_te.gz:
            If teIndices, gzipped file containing the (self.numEdges, self.numTeSamples) array
            describing the edge weights for each test sample.

        Reference
        ---------
        M. L. Kuijjer, M. Tung, G. Yuan, J. Quackenbush and K. Glass (2015).
        Estimating sample-specific regulatory networks.
        arXiv:1505.06440 [q-Bio].
        """
        try:
            assert isinstance(self.globalNetwork, np.ndarray)
        except AssertionError:
            sys.stderr.write("The global network has not been computed yet.\n")
            sys.stderr.write("Call createGlobalNetwork first.\n")
            sys.stderr.write("Aborting.\n")
            sys.exit(-1)

        weightsF = '%s/edge_weights.gz' % lionessPath
        if isinstance(self.teIndices, np.ndarray):
            weightsTeF = '%s/edge_weights_te.gz' % lionessPath


        # Assume expression data has been normalized: xMean = 0 and xStdv = 1.
        # Compute edge weights
        weights = np.array([np.float(self.numSamples)/np.float(self.numSamples - 1) *\
                            self.expressionData[:, e[0]] * self.expressionData[:, e[1]] \
                            for e in self.edges])
        
        # Save edge weights to file
        np.savetxt(weightsF, weights, fmt='%.5f')
        sys.stdout.write("Lioness edge weights saved to %s\n" % weightsF)

        if isinstance(self.teIndices, np.ndarray):
            # Assume expression data has been normalized: xMean = 0 and xStdv = 1.
            # Compute edge weights
            weights = np.array([np.float(self.numSamples)/np.float(self.numSamples - 1) *\
                                self.teExpressionData[:, e[0]] * self.teExpressionData[:, e[1]] \
                                for e in self.edges])
            # Save weight edges to file
            np.savetxt(weightsTeF, weights, fmt='%.5f')
            sys.stdout.write("Lioness edge weights (test data) saved to %s\n" % weightsTeF)
        
        
            
    def createSamSpecRegline(self, reglinePath):
        """ Create sample-specific co-expression networks,
        using the REGLINE approach.

        The weight of the (x0, y0) edge is computed as
        the distance from (x0, y0) to the regression line between x and y.
        
        Parameters
        ----------
        reglinePath: dirname
            Name of the directory where to write the files describing the co-expression networks.
        
        Created files
        -------------
        edge_weights.gz:
            gzipped file containing the (self.numEdges, self.numSamples) array
            describing the edge weights for each sample (training samples only if self.trIndices)
        edge_weights_te.gz:
            If teIndices, gzipped file containing the (self.numEdges, self.numTeSamples) array
            describing the edge weights for each test sample.
        """
        try:
            assert isinstance(self.globalNetwork, np.ndarray)
        except AssertionError:
            sys.stderr.write("The global network has not been computed yet.\n")
            sys.stderr.write("Call createGlobalNetwork first.\n")
            sys.stderr.write("Aborting.\n")
            sys.exit(-1)

        weightsF = '%s/edge_weights.gz' % reglinePath
        if isinstance(self.teIndices, np.ndarray):
            weightsTeF = '%s/edge_weights_te.gz' % reglinePath

        # Compute edge weights
        weights = np.ones((self.numEdges, self.numSamples))
        if isinstance(self.teIndices, np.ndarray):
            weightsTe = np.ones((self.numEdges, self.numTeSamples))

        for (edgeIdx, e) in enumerate(self.edges):
            # Fit regression line
            regW = np.linalg.lstsq(np.array([ self.expressionData[:, e[0]],
                                              np.ones(self.numSamples) ]).transpose(),
                                   self.expressionData[:, e[1]])[0]

            # Compute distance from sample to line:
            weights[edgeIdx, :] = np.abs(regW[1]*self.expressionData[:, e[0]] - \
                                          self.expressionData[:, e[1]] + regW[0]) / \
                np.sqrt(regW[1]**2+1)
            if isinstance(self.teIndices, np.ndarray):
                weightsTe[edgeIdx, :] = np.abs(regW[1]*self.teExpressionData[:, e[0]] - \
                                                self.teExpressionData[:, e[1]] + regW[0]) / \
                    np.sqrt(regW[1]**2+1)
        
        # Save edge weights to file
        np.savetxt(weightsF, weights, fmt='%.5f')
        sys.stdout.write("Regline edge weights saved to %s\n" % weightsF)

        if isinstance(self.teIndices, np.ndarray):
            # Save weight edges to file
            np.savetxt(weightsTeF, weightsTe, fmt='%.5f')
            sys.stdout.write("Regline edge weights (test data) saved to %s\n" % weightsTeF)

            
def run_whole_data(expressionData, sampleLabels, outDir, trIndices=None, teIndices=None):
    """ Build sample-specific co-expression networks.

    If trIndices is not None, use trIndices and teIndices to determine train/test samples
    for normalization and network weights parameters.
    Otherwise, build on the entire given dataset.

    Parameters
    ----------
    expressionData: (numSamples, numGenes) array
        Array of gene expression data.
        If there is training data, this is ONLY the training data.
    sampleLabels: (numSamples, ) array
        Labels for the whole data.
    outDir: path
        Path of the repository where to store the generated networks.

    Created files
    -------------
    outDir/edges.gz: 
        Gzipped file containing the list of edges of the co-expression networks.
        Each line is an undirected edge, formatted as:
            <index of gene 1> <index of gene 2>
        By convention, the index of gene 1 is smaller than that of gene 2.
    outDir/global_connectivity.png
        Regression plot of log10(p(connectivities)) against log10(connectivities)
        for the global network.
    outDir/lioness/edge_weights.gz:
        gzipped file containing the (self.numEdges, self.numSamples) array
        describing the edge weights of the LIONESS co-expression networks
        for each sample (training samples only if self.trIndices)
    outDir/regline/edge_weights.gz:
        gzipped file containing the (self.numEdges, self.numSamples) array
        describing the edge weights of the Regline co-expression networks
        for each sample (training samples only if self.trIndices)

    Created files (optional)
    -----------------------
    outDir/lioness/edge_weights_te.gz:
        gzipped file containing the (self.numEdges, self.numTeSamples) array
        describing the edge weights of the LIONESS co-expression networks
        for each test sample.
    outDir/regline/edge_weights_te.gz:
        gzipped file containing the (self.numEdges, self.numTeSamples) array
        describing the edge weights  of the Regline co-expression networks
        for each test sample.
    """
    # Create CoExpressionNetwork instance
    if not isinstance(trIndices, np.ndarray):
        sys.stdout.write("Computing networks on whole data\n")
        coExpressionNet = CoExpressionNetwork(expressionData, sampleLabels)
    else:
        sys.stdout.write("Computing networks on train / test data\n")
        coExpressionNet = CoExpressionNetwork(expressionData, sampleLabels,
                                              trIndices=trIndices, teIndices=teIndices)

    # Normalize the data
    coExpressionNet.normalizeExpressionData()

    # Create global network
    coExpressionNet.createGlobalNetwork(0.6, outDir)

    # Check whether the scale-free assumptions are verified
    scalefreePath = '%s/global_connectivity.png' % outDir
    aveConn, slope, r2 = coExpressionNet.checkScaleFree(scalefreePath)
    print "Average connectivity: ", aveConn
    print "Slope (should be close to -1): ", slope
    print "R2 (should be larger than 0.8)", r2
     
    # Create repertory in which to store co-expression networks (LIONESS)
    lionessPath = "%s/lioness" % outDir
    try: 
        os.makedirs(lionessPath)
    except OSError:
        if not os.path.isdir(lionessPath):
            raise
    # Build and store co-expression networks (LIONESS)
    coExpressionNet.createSamSpecLioness(lionessPath)
    # # (Uncomment to) Time the creation of LIONESS co-expression networks
    # execTime = timeit.timeit(utils.wrapper(coExpressionNet.createSamSpecLioness, lionessPath),
    #                          number=10)
    # sys.stdout.write("LIONESS network created in %.2f seconds (averaged over 10 repeats)\n" % \
    #                  (execTime/10))

    # Create repertory in which to store co-expression networks (REGLINE)
    reglinePath = "%s/regline" % outDir
    try: 
        os.makedirs(reglinePath)
    except OSError:
        if not os.path.isdir(reglinePath):
            raise

    # Build and store co-expression networks (REGLINE)
    coExpressionNet.createSamSpecRegline(reglinePath)
    # # (Uncomment to) Time the creation of REGLINE co-expression networks
    # execTime = timeit.timeit(utils.wrapper(coExpressionNet.createSamSpecRegline, reglinePath),
    #                          number=10)
    # sys.stdout.write("Regline network created in %.2f seconds (averaged over 10 repeats)\n" % \
    #                  (execTime/10))

    
def run_whole_data_aces(acesData, outDir, trIndices=None, teIndices=None):
    """ Build sample-specific co-expression networks, from data in ACES format.

    If trIndices is not None, use trIndices and teIndices to determine train/test samples
    for normalization and network weights parameters.
    Otherwise, build on the entire given dataset.

    Parameters
    ----------
    acesData: datatypes.ExpressionDataset.ExpressionDataset
        Data in ACES format, read using HDF5GroupToExpressionDataset.
    outDir: path
        Path of the repository where to store the generated networks.

    Created files
    -------------
    outDir/edges.gz: 
        Gzipped file containing the list of edges of the co-expression networks.
        Each line is an undirected edge, formatted as:
            <index of gene 1> <index of gene 2>
        By convention, the index of gene 1 is smaller than that of gene 2.
    outDir/global_connectivity.png
        Regression plot of log10(p(connectivities)) against log10(connectivities)
        for the global network.
    outDir/lioness/edge_weights.gz:
        gzipped file containing the (self.numEdges, self.numSamples) array
        describing the edge weights of the LIONESS co-expression networks
        for each sample (training samples only if self.trIndices)
    outDir/regline/edge_weights.gz:
        gzipped file containing the (self.numEdges, self.numSamples) array
        describing the edge weights of the Regline co-expression networks
        for each sample (training samples only if self.trIndices)

    Created files (optional)
    -----------------------
    outDir/lioness/edge_weights_te.gz:
        gzipped file containing the (self.numEdges, self.numTeSamples) array
        describing the edge weights of the LIONESS co-expression networks
        for each test sample.
    outDir/regline/edge_weights_te.gz:
        gzipped file containing the (self.numEdges, self.numTeSamples) array
        describing the edge weights  of the Regline co-expression networks
        for each test sample.   
    """
    run_whole_data(acesData.expressionData, acesData.patientClassLabels, outDir, trIndices, teIndices)


def run_crossval_data_aces(acesData, outDir, numFolds):
    """ Build sample-specific co-expression networks, in a cross-validation setting.

    Parameters
    ----------
    acesData: datatypes.ExpressionDataset.ExpressionDataset
        Data in ACES format, read using HDF5GroupToExpressionDataset.
    outDir: path
        Path of the repository where to store the generated networks.
    numFolds: int
        Number of cross-validation folds.

    Created files
    -------------
    outDir/edges.gz: 
        Gzipped file containing the list of edges of the co-expression networks.
        Each line is an undirected edge, formatted as:
            <index of gene 1> <index of gene 2>
        By convention, the index of gene 1 is smaller than that of gene 2.
    For k=1..numFolds:
        outDir/<k>/global_connectivity.png
            Regression plot of log10(p(connectivities)) against log10(connectivities)
            for the global network.
        outDir/<k>/train.indices
            List of indices of the training set (one per line).
        outDir/<k>/train.labels
            List of (0/1) labels of the training set (one per line).
        outDir/<k>/test.indices
            List of indices of the test set (one per line).
        outDir/<k>/test.labels
            List of (0/1) labels of the test set (one per line).
        outDir/<k>/lioness/edge_weights.gz:
            gzipped file containing the (self.numSamples, numEdges) array
            describing the edge weights of the LIONESS co-expression networks
            for the training samples.
        outDir/<k>/lioness/edge_weights_te.gz:
            gzipped file containing the (self.numSamples, numEdges) array
            describing the edge weights of the LIONESS co-expression networks
            for the test samples.
        outDir/<k>/regline/edge_weights.gz:
            gzipped file containing the (self.numSamples, numEdges) array
            describing the edge weights of the Regline co-expression networks
            for the training samples.
        outDir/<k>/regline/edge_weights_te.gz:
            gzipped file containing the (self.numSamples, numEdges) array
            describing the edge weights of the Regline co-expression networks
            for the test samples.
    """
    # Split the data
    foldMap = MakeRandomFoldMap(acesData, numFolds, 1)
    
    for foldNr in range(numFolds):
        # Get training and test indices
        trIndices = np.where(np.array(foldMap.foldAssignments)-1 != foldNr)[0]
        teIndices = np.where(np.array(foldMap.foldAssignments)-1 == foldNr)[0]

        # Create output directory
        foldDir = "%s/%s" % (outDir, foldNr)
        try: 
            os.makedirs(foldDir)
        except OSError:
            if not os.path.isdir(foldDir):
                raise
                
        # Save train indices to file
        trIndicesF = '%s/train.indices' % foldDir
        np.savetxt(trIndicesF, trIndices, fmt='%d')
        sys.stdout.write("Wrote training indices for fold %d to %s\n" % (foldNr, trIndicesF))

        # Save test indices to file
        teIndicesF = '%s/test.indices' % foldDir
        np.savetxt(teIndicesF, teIndices, fmt='%d')
        sys.stdout.write("Wrote test indices for fold %d to %s\n" % (foldNr, teIndicesF))

        # Save train labels to file
        trLabelsF = '%s/train.labels' % foldDir
        np.savetxt(trLabelsF, np.array(acesData.patientClassLabels[trIndices], dtype='int'),
                   fmt='%d')
        sys.stdout.write("Wrote training labels for fold %d to %s\n" % (foldNr, trLabelsF))

        # Save test labels to file
        teLabelsF = '%s/test.labels' % foldDir
        np.savetxt(teLabelsF, np.array(acesData.patientClassLabels[teIndices], dtype='int'),
                   fmt='%d')
        sys.stdout.write("Wrote test labels for fold %d to %s\n" % (foldNr, teLabelsF))

        # Create the networks
        run_whole_data(acesData.expressionData, acesData.patientClassLabels, foldDir,
                       trIndices=trIndices, teIndices=teIndices)

        
def main():
    """ Build sample-specific co-expression networks.

    Example:
        $ python CoExpressionNetwork.py DMFS outputs/U133A_combat_DMFS -k 5 
    """
    parser = argparse.ArgumentParser(description="Build sample-specific co-expression networks",
                                     add_help=True)
    parser.add_argument("dataset_name", help="Dataset name")
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

    # # Working on whole data, without using ACES:
    # expressionData = np.array(f['U133A_combat_%s' % args.dataset_name]['ExpressionData'])
    # sampleLabels = np.array(f['U133A_combat_%s' % args.dataset_name]['PatientClassLabels'])
    # run_whole_data(expressionData, sampleLabels, args.out_dir)

    # Get expression data, sample labels.
    # Do not normalize the data while loading it (so as not to use test data for normalization).
    f = h5py.File("ACES/experiments/data/U133A_combat.h5")
    acesData = HDF5GroupToExpressionDataset(f['U133A_combat_%s' % args.dataset_name],
                                            checkNormalise=False)
    f.close()

    if not args.num_folds:
        # Compute networks on whole data 
        run_whole_data_aces(acesData, args.out_dir)
    else:
        # Create cross-validation folds and compute networks on them
        run_crossval_data_aces(acesData, args.out_dir, args.num_folds)
    

if __name__ == "__main__":
    main()
