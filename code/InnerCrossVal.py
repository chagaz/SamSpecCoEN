# @Author 
# Chloe-Agathe Azencott
# chloe-agathe.azencott@mines-paristech.fr
# October 2016

import argparse
import gzip
import itertools
import numpy as np
import os
import subprocess
import sys
import tempfile

import scipy.stats as st 

from sklearn import linear_model as sklm 
from sklearn import metrics as skm
from sklearn import cross_validation as skcv 
from sklearn import preprocessing

import spams  # for elastic-net 
import glmnet # for elastic-net 

scale_data = True # whether to scale data before feeding it to l1-logreg

network_types = ['regline', 'mahalan', 'sum', 'euclide', 'euclthr'] # possible network weight types

class InnerCrossVal(object):
    """ Manage the inner cross-validation loop for learning on sample-specific co-expression networks.

    Attributes
    ----------
    self.x_tr: (num_features, num_training_samples) array
        Feature values for the samples of the training data.
    self.x_te: (num_features, num_test_samples) array
        Feature weights for the samples of the test data.
    self.y_tr: (num_training_samples, ) array
        Labels (0/1) of the training samples.
    self.y_te: (num_test_samples, ) array
        Labels (0/1) of the test samples.
    self.nr_folds: int
        Number of folds for the inner cross-validation loop.
    self.max_nr_feats: int
        Maximum number of features to return.
        Default value=400, as in [Staiger et al.]
    self.num_features: int
        Number of features.

    Optional attributes
    -------------------
    In case we want to use sfan (use_sfan=True):
    self.num_edges: int
        Number of edges.
    self.sfan_path: path
        Path to sfan code.
    self.node_weights_f: path
        File where to store node weights (computed based on correlation betwen self.x_tr and self.y_tr).
    self.ntwk_dimacs_f: path
        Dimacs version of edges.gz
    self.connected_nodes_map: dict
        Map node IDs in dimacs file to node indices in self.x_tr

    Reference
    ---------
    Staiger, C., Cadot, S., Gyoerffy, B., Wessels, L.F.A., and Klau, G.W. (2013).
    Current composite-feature classification methods do not outperform simple single-genes
    classifiers in breast cancer prognosis. Front Genet 4.  
    """
    def __init__(self, aces_data_root, trte_root, ntwk_root, network_type, nr_folds,
                 max_nr_feats=400, use_nodes=False, use_sfan=False, sfan_path=None):
        """
        Parameters
        ----------
        aces_data_root: path
            Path to folder containing ACES data
        trte_root: path
            Path to folder containing train and test indices and labels for the experiment.
        ntwk_root: path
            Path to folder containing network skeleton and edges.
        network_type: string
            Type of network to work with
            Correspond to a folder in trte_root
            Possible value are taken from network_types
        nr_folds: int
            Number of (inner) cross-validation folds.
        max_nr_feats: int
            Maximum number of features to return.
        use_nodes: bool
            Whether to use node weights rather than edge weights as features.
           (This does not make use of the network information.)
        use_sfan: bool
            Whether to use sfan on {node weights + network structure} rather than edge weights.
        sfan_path: path
            Path to sfan code.
        """
        try:
            assert network_type in network_types
        except AssertionError:
            sys.stderr.write("network_type should be one of ")
            sys.stderr.write(",".join([" '%s'" % nt for nt in network_types]))
            sys.stderr.write(".\n")
            sys.stderr.write("Aborting.\n")
            sys.exit(-1)

        # Get train/test indices for fold
        tr_indices = np.loadtxt('%s/train.indices' % trte_root, dtype='int')
        te_indices = np.loadtxt('%s/test.indices' % trte_root, dtype='int')

        # Files neede for the usage of sfan
        if use_sfan:
            print "Using sfan"
            edges_f = '%s/edges.gz' % ntwk_root
            nodes_f = '%s/genes_in_network.txt' % ntwk_root

        if use_nodes or use_sfan:
            print "Using node weights as features"
            # Read ACES data
            import h5py
            sys.path.append(aces_data_root)
            from datatypes.ExpressionDataset import HDF5GroupToExpressionDataset
            f = h5py.File("%s/experiments/data/U133A_combat.h5" % aces_data_root)
            aces_data = HDF5GroupToExpressionDataset(f['U133A_combat_RFS'], checkNormalise=False)
            f.close()

            # Get Xtr, Xte
            self.x_tr = aces_data.expressionData[tr_indices, :]
            self.x_te = aces_data.expressionData[te_indices, :]

            if use_sfan:
                # Restrict to only connected nodes
                print "Using connected nodes only"
                genes_in_network = np.loadtxt(nodes_f, dtype=int)
                self.x_tr = self.x_tr[:, genes_in_network]
                self.x_te = self.x_te[:, genes_in_network]
            
        else:
            print "Using edge weights as features"
            x_f = '%s/%s/edge_weights.gz' % (ntwk_root, network_type)
            self.x_tr = np.loadtxt(x_f).transpose()[tr_indices, :]
            self.x_te = np.loadtxt(x_f).transpose()[te_indices, :]

        # Number of features
        self.num_features = self.x_tr.shape[1]

        # Normalize data (according to training data)
        x_mean = np.mean(self.x_tr, axis=0)
        x_stdv = np.std(self.x_tr, axis=0, ddof=1)

        self.x_tr = (self.x_tr - x_mean) / x_stdv
        self.x_te = (self.x_te - x_mean) / x_stdv

        # Labels
        self.y_tr = np.loadtxt('%s/train.labels' % trte_root, dtype='int')
        self.y_te = np.loadtxt('%s/test.labels' % trte_root, dtype='int')
        self.nr_folds = nr_folds
        self.max_nr_feats = max_nr_feats

        # Compute extra files for the usage of sfan, if needed:
        if use_sfan:
            # Number of edges
            self.num_edges = np.loadtxt(edges_f).shape[0]

            # Dimacs network and connected nodes
            self.ntwk_dimacs_f = '%s/network.dimacs' % ntwk_root
            if not os.path.isfile(self.ntwk_dimacs_f):
                print "Compute dimacs file"
                self.compute_dimacs(edges_f, nodes_f)

            # Node weights
            self.node_weights_f = '%s/scores.txt' % ntwk_root
            if not os.path.isfile(self.node_weights_f):
                print "Compute scores file"
                self.compute_node_weights()

            # Path to sfan
            self.sfan_path = sfan_path
            sys.path.append(self.sfan_path)
            import evaluation_framework as ef


    def compute_dimacs_restrict_to_connected(self, edges_f):
        """ Compute dimacs version of the network file edges.gz.

        Nodes that are not connected to any other are eliminated.

        Input file
        ----------
        edges_f: gzipped file
            Gzipped file containing the list of edges of the co-expression networks.
            Each line is an undirected edge, formatted as:
                <index of gene 1> <index of gene 2>
            By convention, the index of gene 1 is smaller than that of gene 2.

        Modified file
        -------------
        self.ntwk_dimacs_f: path
            Dimacs version of edges.gz

        Modified attributes
        -------------------
        self.connected_nodes_map: dict
            Map node IDs in dimacs file to node indices in self.x_tr.
        self.x_tr, self.x_te:
            Restricted to the nodes that appear in the network.
        """
        sym_edges_dict = {} #j:[i]
        last_idx = 0

        # Temporary dimacs file (non-final node IDs)
        tmp_fname = 'tmp.dimacs'
        fd, tmp_fname = tempfile.mkstemp()

        # Keep track of nodes that have at least one neighbor
        connected_nodes = set([]) 

        with open(tmp_fname, 'w') as g:
            g.write("p max %d %d\n" % (self.num_features, self.num_edges*2))
            with gzip.open(edges_f, 'r') as f:     
                for line in f:
                    idx_1, idx_2 = [int(x) for x in line.split()]
                    # track nodes as connected
                    connected_nodes.add(idx_1)
                    connected_nodes.add(idx_2)
                    # write edges saved in sym_edges_dict:
                    for idx_3 in range(last_idx, idx_1+1):
                        if sym_edges_dict.has_key(idx_3):
                            for idx_0 in sym_edges_dict[idx_3]:
                                g.write("a %d %d 1\n" % (idx_3, idx_0))
                            # delete these entries
                            del sym_edges_dict[idx_3]                   
                    # update last_idx
                    last_idx = idx_1            
                    # write this edge
                    g.write("a %d %d 1\n" % (idx_1, idx_2))
                    # add to dictionary
                    if not sym_edges_dict.has_key(idx_2):
                        sym_edges_dict[idx_2] = []
                    sym_edges_dict[idx_2].append(idx_1)
                f.close()        
                # write the end of the dictionary
                if len(sym_edges_dict):
                    sym_edges_dict_keys = sym_edges_dict.keys()
                    sym_edges_dict_keys.sort()
                    for idx_1 in sym_edges_dict_keys:
                        for idx_0 in sym_edges_dict[idx_1]:
                            g.write("a %d %d 1\n" % (idx_1, idx_0))
            g.close()           

        # Restrict data to nodes that belong to the network:
        connected_nodes = list(connected_nodes)
        connected_nodes.sort()
        self.x_tr = self.x_tr[:, connected_nodes]
        self.x_te = self.x_te[:, connected_nodes]

        self.num_features = len(connected_nodes)
        print "%d nodes in the network." % self.num_features

        # Map node indices in temporary dimacs to node IDs in the final ones
        # and conversely
        map_idx = {}
        self.connected_nodes_map = {}
        for (old_idx, new_idx) in zip(connected_nodes, range(self.num_features)):
            map_idx[old_idx] = new_idx + 1 # indices start at 1 in dimacs file
            self.connected_nodes_map[(new_idx + 1)] = old_idx
            
        # Update node IDs in ntwk_dimacs
        with open(self.ntwk_dimacs_f, 'w') as g:
            g.write("p max %d %d\n" % (self.num_features, self.num_edges*2))
            with open(tmp_fname, 'r') as f:
                f.readline() # header
                for line in f:
                    ls = line.split()
                    g.write("a %d %d %s\n" % (map_idx[int(ls[1])], map_idx[int(ls[2])], ls[3]))
                f.close()
            g.close()

        # Delete temporary file
        os.remove(tmp_fname)        


    def compute_dimacs(self, edges_f, nodes_f):
        """ Compute dimacs version of the network file edges.gz.

        Nodes that are not connected to any other are eliminated.

        Input file
        ----------
        edges_f: gzipped file
            Gzipped file containing the list of edges of the co-expression networks.
            Each line is an undirected edge, formatted as:
                <index of gene 1> <index of gene 2>
            By convention, the index of gene 1 is smaller than that of gene 2.
        nodes_f: file
            Indices (in the gene expressiond data) of genes that are used in the network.

        Modified file
        -------------
        self.ntwk_dimacs_f: path
            Dimacs version of edges.gz

        Modified attributes
        -------------------
        # self.connected_nodes_map: dict
        #     Map node IDs in dimacs file to node indices in self.x_tr.
        self.x_tr, self.x_te:
            Restricted to the nodes that appear in the network.
        """
        # Read list of indices of genes in network
        genes_in_network = np.loadtxt(nodes_f, dtype=int)

        # Restrict data to nodes that belong to the network:
        self.x_tr = self.x_tr[:, genes_in_network]
        self.x_te = self.x_te[:, genes_in_network]

        self.num_features = len(genes_in_network)
        print "%d nodes in the network." % self.num_features

        # Read list of edges
        edges_set = set([])
        with gzip.open(edges_f, 'r') as f:     
            for line in f:
                idx_1, idx_2 = [int(x) for x in line.split()]
                edges_set.add((idx_1, idx_2))
                edges_set.add((idx_2, idx_1))
            f.close()

        edges_list = list(edges_set)
        edges_list.sort()
                
        with open(self.ntwk_dimacs_f, 'w') as g:
            g.write("p max %d %d\n" % (self.num_features, self.num_edges*2))
            for e in edges_list:
                g.write("a %d %d 1\n" % ((e[0]+1), (e[1]+1)))
            g.close()           


    # def compute_dimacs(self, edges_f):
    #     """ Compute dimacs version of the network file edges.gz.

    #     Edges are then ordered.
        
    #     Input file
    #     ----------
    #     edges_f: gzipped file
    #         Gzipped file containing the list of edges of the co-expression networks.
    #         Each line is an undirected edge, formatted as:
    #             <index of gene 1> <index of gene 2>
    #         By convention, the index of gene 1 is smaller than that of gene 2.

    #     Modified file
    #     -------------
    #     self.ntwk_dimacs_f: path
    #         Dimacs version of edges.gz

    #     """
    #     edges_set = set([])
    #     with gzip.open(edges_f, 'r') as f:     
    #         for line in f:
    #             idx_1, idx_2 = [int(x) for x in line.split()]
    #             edges_set.add((idx_1, idx_2))
    #             edges_set.add((idx_2, idx_1))
    #         f.close()

    #     edges_list = list(edges_set)
    #     edges_list.sort()
                
    #     with open(self.ntwk_dimacs_f, 'w') as g:
    #         g.write("p max %d %d\n" % (self.num_features, self.num_edges*2))
    #         for e in edges_list:
    #             g.write("a %d %d 1\n" % ((e[0]+1), (e[1]+1)))
    #         g.close()           


    def compute_node_weights(self):
        """ Compute node weights in the sfan framework.
        
        Each node is assigned the squared Pearson correlation of its corresponding feature
        with the phenotype.

        Modified file
        -------------
        self.node_weights_f: path
            File where to store node weights
            (computed based on correlation betwen self.x_tr and self.y_tr).        
        """
        scores = [st.pearsonr(self.x_tr[:, node_idx], self.y_tr)[0]**2 \
                  for node_idx in range(self.num_features)]

        np.savetxt(self.node_weights_f, scores, fmt='%.3e')
        
                

    # ================== sfan + L1-regularized logistic regression ==================
    def run_inner_sfan(self, reg_params=[itertools.product([10.**k for k in range(-3, 3)],
                                                           [2.**k for k in range(-8, -2)]),
                                         [10.**k for k in range(-3, 3)]]):
                       
        """ Run the inner loop, using sfan for feature selection,
        and a ridge-regression on the selected features for final prediction.
        
        Parameters
        ----------
        reg_params: list
            Range of lambda values and eta values to try out,
            and of lambda values for the ridge regression on the selected features.
        
            Default: [itertools.product([0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                               [0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125]),
                      [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]]
        
        Returns
        -------
        pred_values: (numTestSamples, ) array
            Probability estimates for test samples, in the same order as self.y_tr.
        features: list
            List of indices of the selected features.
        """
        # Get the optimal value of the regularization parameter by inner cross-validation
        best_reg_param = self.cv_inner_sfan(reg_params)

        # Return the predictions and selected features
        return self.train_pred_inner_sfan(best_reg_param)     

        
    def run_inner_sfan_write(self, resdir, 
                             reg_params=[itertools.product([10.**k for k in range(-3, 3)],
                                                           [2.**k for k in range(-8, -2)]),
                                         [10.**k for k in range(-3, 3)]]):
        """ Run the inner loop, using using sfan for feature selection,
        and a ridge-regression on the selected features for final prediction.
        Save outputs to files.
        
        Parameters
        ----------
        resdir: path
            Path to dir where to save outputs
        reg_params: list
            Range of lambda values and eta values to try out for sfan,
            and of lambda values for the ridge regression on the selected features.
        
            Default: [itertools.product([0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                               [0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125]),
                      [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]]
        
        Returns
        -------
        pred_values: (num_test_samples, ) array
            Probability estimates for test samples, in the same order as self.y_tr.
        features: list
            List of indices of the selected features.

        Write files
        -----------
        yte:
            Contains self.y_te
        pred_values:
            Contains predictions
        featuresList:
            Contains selected features
        """
        # Get the optimal value of the regularization parameter by inner cross-validation
        best_reg_param = self.cv_inner_sfan(reg_params)

        # Get the predictions and selected features
        [pred_values, features_list] = self.train_pred_inner_sfan(best_reg_param)

        # Save to files
        yte_fname = '%s/yte' % resdir
        np.savetxt(yte_fname, self.y_te, fmt='%d')
        
        pred_values_fname = '%s/predValues' % resdir
        np.savetxt(pred_values_fname, pred_values)
        
        features_list_fname = '%s/featuresList' % resdir
        np.savetxt(features_list_fname, features_list, fmt='%d')

        
    def cv_inner_sfan(self, reg_params=[itertools.product([10.**k for k in range(-3, 3)],
                                                          [2.**k for k in range(-8, -2)]),
                                        [10.**k for k in range(-3, 3)]]):
        """ Compute the inner cross-validation loop to determine the best regularization parameters
        to select features using sfan,
        and classify the data with an l2-regularized logistic regression.
        
        Parameters
        ----------
        reg_params: list
            Range of lambda values and eta values to try out for sfan,
            and of lambda values for the ridge regression on the selected features.

            Default: [itertools.product([0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                               [0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125]),
                      [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]]
        
        Returns
        -------
        best_reg_param: float
            Optimal values of the regularization parameters:
            lambda_sfan, eta_sfan, C_ridge.
        """
        # print "reg_params[0]"
        # for (l, e) in reg_params[0]:
        #     print l, e
        # sys.exit(0)
        msfanpy = '%s/multitask_sfan.py' % self.sfan_path
        sys.path.append(self.sfan_path)
        import evaluation_framework as ef
        
        # Create cross-validation split of the training data
        cross_validator = skcv.StratifiedKFold(self.y_tr, self.nr_folds)
                
        # Create temporary node_weights files for all splits
        tmp_node_weights_f_list = []
        num_nodes = self.x_tr.shape[1]
        for tr, te in cross_validator:
            scores = [st.pearsonr(self.x_tr[tr, node_idx], self.y_tr[tr])[0]**2 \
                      for node_idx in range(num_nodes)]

            fd, tmp_node_weights_f = tempfile.mkstemp()
            np.savetxt(tmp_node_weights_f, scores, fmt='%.3e')

            tmp_node_weights_f_list.append(tmp_node_weights_f)
        
        # Cross-validate parameters for feature selection
        best_reg_param = []
        best_ci = -10.0
        for (lbd, eta) in reg_params[0]:
            sel_list = []
            for tmp_node_weights_f in tmp_node_weights_f_list:
                argum = ['python', msfanpy, '--num_tasks', '1',
                         '--networks', self.ntwk_dimacs_f, 
                         '--node_weights', tmp_node_weights_f,
                         '-l', ('%f' % lbd), '-e', ('%f' % eta), '-m', '0']
                #print "\tRunning: ", " ".join(argum)
                p = subprocess.Popen(argum, stdout=subprocess.PIPE)
                pc = p.communicate()
                p_out = pc[0].split("\n")[2]
                sel_list.append([int(x) for x in p_out.split()])
            ci = ef.consistency_index_k(sel_list, num_nodes)
            print "\tlbd:\t%.2e\teta:\t%.2e\tcix:%.2f" % (lbd, eta, ci)
            if ci >= best_ci:
                best_reg_param = [lbd, eta]
                best_ci = ci
        print "\nbest lambda\t", best_reg_param[0], "\tbest eta\t", best_reg_param[1]
        print "\tbest cix\t", best_ci
        
        # Delete temporary node_weights files
        for tmp_node_weights_f in tmp_node_weights_f_list:
            os.remove(tmp_node_weights_f)

        # Run feature selection with sfan on whole training data, with optimal parameters
        argum = ['python', msfanpy, '--num_tasks', '1',
                 '--networks', self.ntwk_dimacs_f, 
                 '--node_weights', self.node_weights_f,
                 '-l',  ('%f' % best_reg_param[0]),
                 '-e',  ('%f' % best_reg_param[1]), '-m', '0']
        #print "Running: ", " ".join(argum)
        p = subprocess.Popen(argum, stdout=subprocess.PIPE)
        p_out = p.communicate()[0].split("\n")[2]
        # selected_features = [self.connected_nodes_map[int(x)] for x in p_out.split()]
        selected_features = [(int(x)-1) for x in p_out.split()]
        print self.num_features, "features/nodes in network"
        print len(selected_features), "features/nodes selected"
        print selected_features
        # print np.max(selected_features), ": highest selected features"

        # Initialize logistic regression cross-validation classifier
        cv_clf = sklm.LogisticRegressionCV(Cs=reg_params[1], penalty='l2', solver='liblinear',
                                           cv=self.nr_folds,
                                           class_weight='balanced', scoring='roc_auc')

        # Fit to training data, restricted to selected_features
        cv_clf.fit(self.x_tr[:, selected_features], self.y_tr)

        # Quality of fit?
        y_tr_pred = cv_clf.predict_proba(self.x_tr[:, selected_features])
        y_tr_pred = y_tr_pred[:, cv_clf.classes_.tolist().index(1)]
        print "\tTraining AUC:\t", skm.roc_auc_score(self.y_tr, y_tr_pred)

        # Get optimal value of C_ridge.
        # If there are multiple equivalent values, return the first one.
        # Note: Small C = more regularization.
        best_C_ridge = cv_clf.C_[0]
        print "\tall top C:\t", cv_clf.C_
        print "\tbest C:\t", best_C_ridge

        best_reg_param.append(best_C_ridge)
        return best_reg_param

        
    def train_pred_inner_sfan(self, best_reg_param):
        """ Run sfan (with optimal parameters) on the train set to select features,
        then train an l2-regularized logistic regression (with optimal parameter)
        on the train set, predict on the test set.
        
        Parameters
        ----------
        best_reg_param: float list
            Optimal value of the regularization parameters:
            lambda_sfan, eta_sfan, C_ridge.

        Returns
        -------
        pred_values: (num_test_samples, ) array
            Probability estimates for test samples, in the same order as trueLabels.
        features: list
            List of indices of the selected features.
        """
        msfanpy = '%s/multitask_sfan.py' % self.sfan_path

        # Run feature selection with sfan on whole training data, with optimal parameters
        argum = ['python', msfanpy, '--num_tasks', '1',
                 '--networks', self.ntwk_dimacs_f, 
                 '--node_weights', self.node_weights_f,
                 '-l',  ('%f' % best_reg_param[0]),
                 '-e',  ('%f' % best_reg_param[1]), '-m', '0']
        # print "Running: ", " ".join(argum)
        p = subprocess.Popen(argum, stdout=subprocess.PIPE)
        p_out = p.communicate()[0].split("\n")[2]
        # features = [self.connected_nodes_map[int(x)] for x in p_out.split()]
        features = [(int(x)-1) for x in p_out.split()]
        
        # Initialize l2-regularized logistic regression classifier
        classif = sklm.LogisticRegression(C=best_reg_param[2], penalty='l2',
                                          solver='liblinear',
                                          class_weight='balanced')
        
        # Train on the training set
        classif.fit(self.x_tr[:, features], self.y_tr)

        # Predict on the test set
        pred_values = classif.predict_proba(self.x_te[:, features])

        # Only get the probability estimates for the positive class
        pred_values = pred_values[:, classif.classes_.tolist().index(1)]

        # Quality of fit
        print "\tTest AUC:\t", skm.roc_auc_score(self.y_te, pred_values)

        print "\tNumber of selected features:\t", len(features)

        return pred_values, features
    # ================== End sfan + L1-regularized logistic regression ==================

        


    # ================== L1-regularized logistic regression ==================
    def run_inner_l1_logreg(self, reg_params=[10.**k for k in range(-3, 3)]):
        """ Run the inner loop, using an l1-regularized logistic regression.
        
        Parameters
        ----------
        reg_params: list
            Range of lambda values to try out.
            Default: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        
        Returns
        -------
        pred_values: (num_test_samples, ) array
            Probability estimates for test samples, in the same order as self.y_tr.
        features: list
            List of indices of the selected features.

        Write files
        -----------
        yte:
            Contains self.y_te
        pred_values:
            Contains predictions
        featuresList:
            Contains selected features
        """
        # Get the optimal value of the regularization parameter by inner cross-validation
        best_reg_param = self.cv_inner_l1_logreg(reg_params)

        # Get the predictions and selected features
        return self.train_pred_inner_l1_logreg(best_reg_param)

        
    def run_inner_l1_logreg_write(self, resdir, reg_params=[10.**k for k in range(-3, 3)]):
        """ Run the inner loop, using an l1-regularized logistic regression.
        Save outputs to files.
        
        Parameters
        ----------
        resdir: path
            Path to dir where to save outputs
        reg_params: list
            Range of lambda values to try out.
            Default: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        
        Returns
        -------
        pred_values: (num_test_samples, ) array
            Probability estimates for test samples, in the same order as self.y_tr.
        features: list
            List of indices of the selected features.

        Write files
        -----------
        yte:
            Contains self.y_te
        pred_values:
            Contains predictions
        featuresList:
            Contains selected features
        """
        # Get the optimal value of the regularization parameter by inner cross-validation
        best_reg_param = self.cv_inner_l1_logreg(reg_params)

        # Get the predictions and selected features
        [pred_values, features_list] = self.train_pred_inner_l1_logreg(best_reg_param)

        # Save to files
        yte_fname = '%s/yte' % resdir
        np.savetxt(yte_fname, self.y_te, fmt='%d')
        
        pred_values_fname = '%s/predValues' % resdir
        np.savetxt(pred_values_fname, pred_values)
        
        features_list_fname = '%s/featuresList' % resdir
        np.savetxt(features_list_fname, features_list, fmt='%d')            
        
        
    def cv_inner_l1_logreg(self, reg_params=[10.**k for k in range(-3, 3)]):
        """ Compute the inner cross-validation loop to determine the best regularization parameter
        for an l1-regularized logistic regression.
        
        Parameters
        ----------
        reg_params: list
            Range of lambda values to try out.
            Default: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        
        Returns
        -------
        best_reg_param: float
            Optimal value of the regularization parameter.
        """
        # Initialize logistic regression cross-validation classifier
        print reg_params
        cv_clf = sklm.LogisticRegressionCV(Cs=reg_params, penalty='l1', solver='liblinear',
                                           cv=self.nr_folds,
                                           refit=True,
                                           class_weight='balanced', scoring='roc_auc')

        # Fit to training data
        # print self.x_tr.shape
        # print self.y_tr.shape
        if scale_data:
            x_scaled = preprocessing.scale(self.x_tr)
            cv_clf.fit(x_scaled, self.y_tr)
        else:
            cv_clf.fit(self.x_tr, self.y_tr)

        # Quality of fit?
        if scale_data:
            y_tr_pred = cv_clf.predict_proba(x_scaled)
        else:
            y_tr_pred = cv_clf.predict_proba(self.x_tr)
        y_tr_pred = y_tr_pred[:, cv_clf.classes_.tolist().index(1)]
        print "\tTraining AUC:\t", skm.roc_auc_score(self.y_tr, y_tr_pred)

        # print cv_clf.scores_
        
        # Get optimal value of the regularization parameter.
        # If there are multiple equivalent values, return the first one.
        # Note: Small C = more regularization.
        best_reg_param = cv_clf.C_[0]
        print "\tall top C:\t", cv_clf.C_
        print "\tbest C:\t", best_reg_param
        return best_reg_param

        
    def train_pred_inner_l1_logreg(self, best_reg_param):
        """ Train an l1-regularized logistic regression (with optimal parameter)
        on the train set, predict on the test set.
        
        Parameters
        ----------
        best_reg_param: float
            Optimal value of the regularization parameter.

        Returns
        -------
        pred_values: (num_test_samples, ) array
            Probability estimates for test samples, in the same order as trueLabels.
        features: list
            List of indices of the selected features.
        """
        # Initialize logistic regression classifier
        classif = sklm.LogisticRegression(C=best_reg_param, penalty='l1', solver='liblinear',
                                          class_weight='balanced')
        
        # Train on the training set
        if scale_data:
            scaler = preprocessing.StandardScaler().fit(self.x_tr)
            x_scaled = scaler.transform(self.x_tr)
            classif.fit(x_scaled, self.y_tr)
        else:
            classif.fit(self.x_tr, self.y_tr)

        # Predict on the test set
        if scale_data:
            x_scaled = scaler.transform(self.x_te)
            pred_values = classif.predict_proba(x_scaled)
        else:
            pred_values = classif.predict_proba(self.x_te)

        # Only get the probability estimates for the positive class
        pred_values = pred_values[:, classif.classes_.tolist().index(1)]

        # Quality of fit
        print "\tTest AUC:\t", skm.roc_auc_score(self.y_te, pred_values)

        # Get selected features
        # If there are less than self.max_nr_feats, these are the non-zero coefficients
        features = np.where(classif.coef_[0])[0]
        if len(features) > self.max_nr_feats:
            # Prune the coefficients with lowest values
            features = np.argsort(classif.coef_[0])[-self.max_nr_feats:]
 
        print "\tNumber of selected features:\t", len(features)

        return pred_values, features
    # ================== End l1-regularized logistic regression ==================
        


    # ================== l2-regularized logistic regression ==================
    def run_inner_l2_logreg(self, reg_params=[10.**k for k in range(-3, 3)]):
        """ Run the inner loop, using an l2-regularized logistic regression.
        
        Parameters
        ----------
        reg_params: list
            Range of lambda values to try out.
            Default: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        
        Returns
        -------
        pred_values: (num_test_samples, ) array
            Probability estimates for test samples, in the same order as self.y_tr.
        features: list
            List of indices of the selected features.

        Write files
        -----------
        yte:
            Contains self.y_te
        pred_values:
            Contains predictions
        featuresList:
            Contains selected features
        """
        # Get the optimal value of the regularization parameter by inner cross-validation
        best_reg_param = self.cv_inner_l2_logreg(reg_params)

        # Get the predictions and selected features
        return self.train_pred_inner_l2_logreg(best_reg_param)

        
    def run_inner_l2_logreg_write(self, resdir, reg_params=[10.**k for k in range(-3, 3)]):
        """ Run the inner loop, using an l2-regularized logistic regression.
        Save outputs to files.
        
        Parameters
        ----------
        resdir: path
            Path to dir where to save outputs
        reg_params: list
            Range of lambda values to try out.
            Default: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        
        Returns
        -------
        pred_values: (num_test_samples, ) array
            Probability estimates for test samples, in the same order as self.y_tr.
        features: list
            List of indices of the selected features.

        Write files
        -----------
        yte:
            Contains self.y_te
        pred_values:
            Contains predictions
        featuresList:
            Contains selected features
        """
        # Get the optimal value of the regularization parameter by inner cross-validation
        best_reg_param = self.cv_inner_l2_logreg(reg_params)

        # Get the predictions and selected features
        [pred_values, features_list] = self.train_pred_inner_l2_logreg(best_reg_param)

        # Save to files
        yte_fname = '%s/yte' % resdir
        np.savetxt(yte_fname, self.y_te, fmt='%d')
        
        pred_values_fname = '%s/predValues' % resdir
        np.savetxt(pred_values_fname, pred_values)
        
        features_list_fname = '%s/featuresList' % resdir
        np.savetxt(features_list_fname, features_list, fmt='%d')            
        
        
    def cv_inner_l2_logreg(self, reg_params=[10.**k for k in range(-3, 3)]):
        """ Compute the inner cross-validation loop to determine the best regularization parameter
        for an l2-regularized logistic regression.
        
        Parameters
        ----------
        reg_params: list
            Range of lambda values to try out.
            Default: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        
        Returns
        -------
        best_reg_param: float
            Optimal value of the regularization parameter.
        """
        # Initialize logistic regression cross-validation classifier
        cv_clf = sklm.LogisticRegressionCV(Cs=reg_params, penalty='l2', solver='liblinear',
                                           cv=self.nr_folds,
                                           class_weight='balanced', scoring='roc_auc')

        # Fit to training data
        cv_clf.fit(self.x_tr, self.y_tr)

        # Quality of fit?
        y_tr_pred = cv_clf.predict_proba(self.x_tr)
        y_tr_pred = y_tr_pred[:, cv_clf.classes_.tolist().index(1)]
        print "\tTraining AUC:\t", skm.roc_auc_score(self.y_tr, y_tr_pred)

        # Get optimal value of the regularization parameter.
        # If there are multiple equivalent values, return the first one.
        # Note: Small C = more regularization.
        best_reg_param = cv_clf.C_[0]
        print "\tall top C:\t", cv_clf.C_
        print "\tbest C:\t", best_reg_param
        return best_reg_param

        
    def train_pred_inner_l2_logreg(self, best_reg_param):
        """ Train an l2-regularized logistic regression (with optimal parameter)
        on the train set, predict on the test set.
        
        Parameters
        ----------
        best_reg_param: float
            Optimal value of the regularization parameter.

        Returns
        -------
        pred_values: (num_test_samples, ) array
            Probability estimates for test samples, in the same order as trueLabels.
        features: list
            List of indices of the selected features.
        """
        # Initialize logistic regression classifier
        classif = sklm.LogisticRegression(C=best_reg_param, penalty='l2', solver='liblinear',
                                          class_weight='balanced')
        
        # Train on the training set
        classif.fit(self.x_tr, self.y_tr)

        # Predict on the test set
        pred_values = classif.predict_proba(self.x_te)

        # Only get the probability estimates for the positive class
        pred_values = pred_values[:, classif.classes_.tolist().index(1)]

        # Quality of fit
        print "\tTest AUC:\t", skm.roc_auc_score(self.y_te, pred_values)

        # Get selected features
        # If there are less than self.max_nr_feats, these are the non-zero coefficients
        features = np.where(classif.coef_[0])[0]
        if len(features) > self.max_nr_feats:
            # Prune the coefficients with lowest values
            features = np.argsort(classif.coef_[0])[-self.max_nr_feats:]
 
        print "\tNumber of selected features:\t", len(features)

        return pred_values, features
    # ================== End l2-regularized logistic regression ==================

        


    # ================== l1/l2-regularized logistic regression ==================
    def run_inner_enet_logreg(self, reg_params=[10, np.arange(0, 1.1, 0.1)]):
                              # reg_params=[[10.**k for k in range(-3, 3)],
                              #                   [0.25, 0.5, 0.75]]):
        """ Run the inner loop, using an l1/l2-regularized logistic regression.
        
        Parameters
        ----------
        # reg_params: list of list
        #     Range of lambda1 and lambda_ratio values to try out.
        #     Default: [[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        #               [0.25, 0.5, 0.75]]
        reg_params: list of list
            Range of regularization parameters to try out
            Default: [10, np.arange(0, 1.1, 0.1)]
                10 means 10 values of lambdas will be tested
                np.arange(0, 1.1, 0.1) is the range of alpha values (alpha l1 + (1-alpha) l2).
        
        Returns
        -------
        pred_values: (num_test_samples, ) array
            Probability estimates for test samples, in the same order as self.y_tr.
        features: list
            List of indices of the selected features.

        Write files
        -----------
        yte:
            Contains self.y_te
        pred_values:
            Contains predictions
        featuresList:
            Contains selected features
        """
        # Get the optimal value of the regularization parameter by inner cross-validation
        best_reg_param = self.cv_inner_enet_logreg(reg_params)

        # Get the predictions and selected features
        return self.train_pred_inner_enet_logreg(best_reg_param)
        

    def run_inner_enet_logreg_write(self, resdir, reg_params=[10, np.arange(0, 1.1, 0.1)]):
                                                  # reg_params=[[10.**k for k in range(-3, 3)],
                                                  #             [0.25, 0.5, 0.75]]):
        """ Run the inner loop, using an l1/l2-regularized logistic regression.
        Save outputs to files.
        
        Parameters
        ----------
        resdir: path
            Path to dir where to save outputs

        # reg_params: list of list
        #     Range of lambda1 and lambda_ratio values to try out.
        #     Default: [[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        #               [0.25, 0.5, 0.75]]
        reg_params: list of list
            Range of regularization parameters to try out
            Default: [10, np.arange(0, 1.1, 0.1)]
                10 means 10 values of lambdas will be tested;
                   if list, user-provided list of lambda values.
                np.arange(0, 1.1, 0.1) is the range of alpha values (alpha l1 + (1-alpha) l2).
        
        Returns
        -------
        pred_values: (num_test_samples, ) array
            Probability estimates for test samples, in the same order as self.y_tr.
        features: list
            List of indices of the selected features.

        Write files
        -----------
        yte:
            Contains self.y_te
        pred_values:
            Contains predictions
        featuresList:
            Contains selected features
        """
        # Get the optimal value of the regularization parameter by inner cross-validation
        best_reg_param = self.cv_inner_enet_logreg(reg_params)

        # Get the predictions and selected features
        [pred_values, features_list] = self.train_pred_inner_enet_logreg(best_reg_param)

        # Save to files
        yte_fname = '%s/yte' % resdir
        np.savetxt(yte_fname, self.y_te, fmt='%d')
        
        pred_values_fname = '%s/predValues' % resdir
        np.savetxt(pred_values_fname, pred_values)
        
        features_list_fname = '%s/featuresList' % resdir
        np.savetxt(features_list_fname, features_list, fmt='%d')            
        
        
    def cv_inner_enet_spams_logreg(self, reg_params=[[10.**k for k in range(-3, 3)],
                                               [0.25, 0.5, 0.75]]):
        """ Compute the inner cross-validation loop to determine the best regularization parameter
        for an l1/l2-regularized logistic regression.

        Use the SPAMS toolbox.
        
        Parameters
        ----------
        reg_params: list of list
            Range of lambda1 and lambda2 values to try out.
            Default: [[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                      [0.25, 0.5, 0.75]]
        
        Returns
        -------
        best_reg_param: float
            Optimal value of the regularization parameter.
        """
        # Compute cross-validation folds
        cross_validator = skcv.StratifiedKFold(self.y_tr, self.nr_folds)
        print "num samples: %d" % self.x_tr.shape[0]

        # Compute cross-validated AUCs for all parameters
        auc_dict = {}
        for lbd1 in reg_params[0]:
            for lbd_ratio in reg_params[1]:
                spams_params = {'loss': 'logistic', 
                                'regul': 'elastic-net', 
                                'lambda1': lbd1,
                                'lambda2': lbd_ratio * lbd1,
                                'max_it':200}
                y_true = []
                y_pred = []
                for tr, te in cross_validator:
                    init_weights = np.zeros((self.x_tr.shape[1], 1), dtype=np.float64, order="FORTRAN")
                    y_fortran = np.asfortranarray(np.reshape(self.y_tr[tr]*2-1,
                                                             (self.y_tr[tr].shape[0], 1)), dtype=np.float64)
                    x_fortran = np.asfortranarray(self.x_tr[tr, :], dtype=np.float64)

                    # Train
                    fit_weights = spams.fistaFlat(y_fortran, x_fortran, init_weights,
                                                   False, **spams_params)
                    # print "numfeat %.d" % len(np.nonzero(fit_weights)[0])
                    # Predict
                    ytr_te_pred = np.dot(self.x_tr[te, :], fit_weights)
                    y_pred.extend(ytr_te_pred[:, 0])
                    y_true.extend(self.y_tr[te])
                                                   

                auc = skm.roc_auc_score(y_true, y_pred)
                print "\tlambda1 %.2e" % lbd1, "\tlambda2  %.2e" % (lbd_ratio * lbd1),
                print "\tauc", auc
                if not auc_dict.has_key(auc):
                    auc_dict[auc] = []
                auc_dict[auc].append([lbd1, lbd_ratio])

        # Get best parameters
        auc_values = auc_dict.keys()
        auc_values.sort()
        best_auc = auc_values[-1]
        best_reg_param = auc_dict[best_auc][0]

        # Get optimal value of the regularization parameter.
        # If there are multiple equivalent values, return the first one.
        print "\tall top params:\t", auc_dict[best_auc]
        print "\tbest params:\t", best_reg_param
        return best_reg_param

        
    def train_pred_inner_enet_spams_logreg(self, best_reg_param):
        """ Train an l1-regularized logistic regression (with optimal parameter)
        on the train set, predict on the test set.
        
        Use the SPAMS toolbox.
        
        Parameters
        ----------
        best_reg_param: float
            Optimal value of the regularization parameters.
            [best_lambda1, best_lambda2]

        Returns
        -------
        pred_values: (num_test_samples, ) array
            Probability estimates for test samples, in the same order as trueLabels.
        features: list
            List of indices of the selected features.
        """
        # Initialize logistic regression classifier
        spams_params = {'loss': 'logistic', 
                        'regul': 'elastic-net', 
                        'lambda1': best_reg_param[0],
                        'lambda2': best_reg_param[1]*best_reg_param[0],
                        'max_it':200}       
        
        # Train on the training set
        init_weights = np.zeros((self.x_tr.shape[1], 1), dtype=np.float64, order="FORTRAN")
        y_fortran = np.asfortranarray(np.reshape(self.y_tr*2-1,
                                                 (self.y_tr.shape[0], 1)), dtype=np.float64)
        x_fortran = np.asfortranarray(self.x_tr, dtype=np.float64)
        fit_weights = spams.fistaFlat(y_fortran, x_fortran, init_weights,
                                      False, **spams_params)

        # Quality of fit
        y_tr_pred = np.dot(self.x_tr, fit_weights)
        print "\tTraining AUC:\t", skm.roc_auc_score(self.y_tr, y_tr_pred)

        # Predict on the test set
        pred_values = np.dot(self.x_te, fit_weights)
        print "\tTest AUC:\t", skm.roc_auc_score(self.y_te, pred_values)

        # Get selected features
        # If there are less than self.max_nr_feats, these are the non-zero coefficients
        features = np.nonzero(fit_weights)[0]
        if len(features) > self.max_nr_feats:
            # Prune the coefficients with lowest values
            features = np.argsort(fit_weights)[-self.max_nr_feats:]

        print "\tNumber of selected features:\t", len(features)

        return pred_values, features



        
    def cv_inner_enet_logreg(self, reg_params=[10, np.arange(0, 1.1, 0.1)]):
        """ Compute the inner cross-validation loop to determine the best regularization parameter
        for an l1/l2-regularized logistic regression.

        Use python-glmnet
        
        Parameters
        ----------
        reg_params: list of list
            Range of regularization parameters to try out
            Default: [10, np.arange(0, 1.1, 0.1)]
                10 means 10 values of lambdas will be tested;
                   if list, user-provided list of lambda values.
                np.arange(0, 1.1, 0.1) is the range of alpha values (alpha l1 + (1-alpha) l2).
        
        Returns
        -------
        best_reg_param: float
            Optimal values of the regularization parameters.
        """
        cv_score_dict = {} # cv_score:[alpha, lambda]
        for alpha in reg_params[1]:
            # Initialize logistic regression cross-validation classifier
            cv_clf = glmnet.LogitNet(alpha=alpha,
                                     n_lambda=reg_params[0],
                                     scoring='roc_auc', n_splits=self.nr_folds)

            # Fit to training data
            cv_clf.fit(self.x_tr, self.y_tr)

            # Get optimal lambda value
            best_lambda = cv_clf.lambda_max_

            # Save cv_score            
            cv_score = cv_clf.cv_mean_score_[cv_clf.lambda_max_inx_]
            
            print "\talpha %.2e" % alpha, "\tlambda opt  %.2e" % best_lambda,
            print "\tcv_score", cv_score
            if not cv_score_dict.has_key(cv_score):
                cv_score_dict[cv_score] = []
            cv_score_dict[cv_score].append([alpha, best_lambda])

        # Get best parameters
        cv_score_values = cv_score_dict.keys()
        cv_score_values.sort()
        best_cv_score = cv_score_values[-1]
        best_reg_param = cv_score_dict[best_cv_score][0]
            
        print "\tbest alpha:\t", best_reg_param[0]
        print "\tbest lambda:\t", best_reg_param[1]
        return best_reg_param

        
    def train_pred_inner_enet_logreg(self, best_reg_param):
        """ Train an l1-regularized logistic regression (with optimal parameter)
        on the train set, predict on the test set.
        
        Use python-glmnet
        
        Parameters
        ----------
        best_reg_param: [best_alpha, best_lambda]
            best_alpha: float
            Optimal value of the alpha ratio.
        
            best_lamb: float
            Optimal value of the lambda regularization parameter (total amount of regularization).

        Returns
        -------
        pred_values: (num_test_samples, ) array
            Probability estimates for test samples, in the same order as trueLabels.
        features: list
            List of indices of the selected features.
        """
        [best_alpha, best_lamb] = best_reg_param
        # Initialize ElasticNet classifier
        classif = glmnet.LogitNet(alpha=best_alpha, lambda_path=[best_lamb], n_splits=0)
        
        # Train on the training set
        classif.fit(self.x_tr, self.y_tr)

        # Predict on the train set
        y_tr_pred = classif.predict_proba(self.x_tr, lamb=best_lamb)

        # Only get the probability estimates for the positive class
        y_tr_pred = y_tr_pred[:, classif.classes_.tolist().index(1)]

        # Quality of fit (train set)
        print "\tTraining AUC:\t", skm.roc_auc_score(self.y_tr, y_tr_pred)


        # Quality of fit (test set)
        # Predict on the test set
        y_te_pred = classif.predict_proba(self.x_te, lamb=best_lamb)

        # Only get the probability estimates for the positive class
        y_te_pred = y_te_pred[:, classif.classes_.tolist().index(1)]
        print "\tTest AUC:\t", skm.roc_auc_score(self.y_te, y_te_pred)

        # Get selected features
        # If there are less than self.max_nr_feats, these are the non-zero coefficients
        features = np.where(classif.coef_path_[0, :, 0])[0]
        if len(features) > self.max_nr_feats:
            # Prune the coefficients with lowest values
            features = np.argsort(classif.coef_path_[0, :, 0])[-self.max_nr_feats:]
 
        print "\tNumber of selected features:\t", len(features)

        return y_te_pred, features
    # ================== End l1/l2-regularized logistic regression ==================
        

def main():
    """ Run an inner cross-validation on sample-specific co-expression networks.
    Save results to file.

    Example
    -------
        $ python InnerCrossVal.py ../ACES ../outputs/U133A_combat_RFS/ \
    ../outputs/U133A_combat_RFS/subtype_stratified/repeat0/fold0 \
    regline -k 5 -m 1000 

    Files created
    -------------
    <results_dir>/yte
        True test labels.

    <results_dir>/predValues
        Predictions for test samples.

    <results_dir>/featuresList
        Selected features.
    """
    parser = argparse.ArgumentParser(description="Inner CV of sample-specific co-expression networks",
                                     add_help=True)
    parser.add_argument("aces_data_path", help="Path to the folder containing the ACES data")
    parser.add_argument("network_path", help="Path to the folder containing network skeleton and weights")
    parser.add_argument("trte_path", help="Path to the folder containing train/test indices")
    parser.add_argument("network_type", help="Type of co-expression networks")
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
        assert args.network_type in network_types
    except AssertionError:
        sys.stderr.write("network_type should be one of ")
        sys.stderr.write(",".join([" '%s'" % nt for nt in network_types]))
        sys.stderr.write(".\n")
        sys.stderr.write("Aborting.\n")
        sys.exit(-1)

    # Whether or not to use sfan
    use_sfan = False
    if args.sfan:
        use_sfan = True

    # Initialize InnerCrossVal
    icv = InnerCrossVal(args.aces_data_path, args.trte_path, args.network_path,
                        args.network_type, args.num_inner_folds, 
                        max_nr_feats=args.max_nr_feats, 
                        use_nodes=args.nodes, use_sfan=use_sfan, sfan_path=args.sfan)

    # Results directory
    if args.nodes:
        if args.sfan:
            args.results_dir = "%s/results/nodes/sfan" % args.trte_path
        elif args.enet:
            args.results_dir = "%s/results/nodes/enet" % args.trte_path
        else:
            args.results_dir = "%s/results/nodes" % args.trte_path
    elif args.enet:
        args.results_dir = "%s/results/%s/enet" % (args.trte_path, args.network_type)
    else:
        args.results_dir = "%s/results/%s" % (args.trte_path, args.network_type)
        
    # Create results dir if it does not exist
    if not os.path.isdir(args.results_dir):
        sys.stdout.write("Creating %s\n" % args.results_dir)
        try: 
            os.makedirs(args.results_dir)
        except OSError:
            if not os.path.isdir(args.results_dir):
                raise

    # ========= sfan =========
    if use_sfan:
        ridge_C = [10.**k for k in range(-4, 1)]
        # To have l2 log reg on selected features:
        # icv.run_inner_l2_logreg_write(args.results_dir, reg_params=ridge_C)
        
        # Use sfan to select features
        sfan_eta_values = [0.008, 0.01, 0.02, 0.05, 0.1]#[5**(k) for k in range(-5, -2)]
        sfan_lbd_values = [0.008, 0.01, 0.02, 0.05, 0.1] #[5**(k) for k in range(-5, -2)]
        sfan_reg_params = [itertools.product(sfan_lbd_values,
                                             sfan_eta_values), ridge_C]
        # sfan_reg_params = [[[l, e] for l, e in itertools.product(sfan_lbd_values,
        #                                                         sfan_eta_values)]]
        sfan_reg_params.append(ridge_C)
        # print "sfan_reg_params", sfan_reg_params
        # for (l, e) in sfan_reg_params[0]:
        #     print l, e
        icv.run_inner_sfan_write(args.results_dir,
                                 reg_params=sfan_reg_params)
    # ========= End sfan =========
        
    else:
        # ========= l1 regularization =========
        if not args.enet:
            # Run the inner cross-validation for the l1 regularization
            # print "l1 regularization (sklearn)"
            # icv.run_inner_l1_logreg_write(args.results_dir,
            #                               reg_params=[2.**k for k in range(-7, 0)])

            print "l1 regularization (python-glmnet)"
            number_of_lambda_values = 50
            alphas = [1.0]
            icv.run_inner_enet_logreg_write(args.results_dir,
                                            reg_params=[number_of_lambda_values,
                                                        alphas])
        # ========= End l1 regularization =========


        # ========= l1/l2 regularization =========
        else:
            print "l1/l2 regularization"
            
            # Run the inner cross-validation for the l1/l2 regularization
            # l1_values = [2.**k for k in range(-8, -2)]
            # lbd_ratio_values = [0.5, 1.0, 1.5]
            # icv.run_inner_enet_logreg_write(args.results_dir,
            #                                 reg_params=[l1_values, lbd_ratio_values])
            number_of_lambda_values = 50
            alphas = np.arange(0.1, 1, 0.1)
            icv.run_inner_enet_logreg_write(args.results_dir,
                                            reg_params=[number_of_lambda_values,
                                                        alphas])
        # ========= End l1/l2 regularization =========


if __name__ == "__main__":
    main()
                
