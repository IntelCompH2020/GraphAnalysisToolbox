#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Python libraries
import logging
import numpy as np
import pandas as pd
import scipy.sparse as scsp
from scipy.sparse import save_npz
import os
from time import time

# Local imports
from rdigraphs.supergraph.snode import DataGraph
from rdigraphs.sim_graph.sim_bigraph import SimBiGraph

EPS = np.finfo(float).tiny


class SEdge(DataGraph):
    """
    Generic class defining a super-edge: a bipartite datagraph.

    It is inherited from DataGraph. This is because a bipartite graph is
    nothing but a particular type of graph.

    The SEdge class distinguishes between source nodes and target nodes. Thus,
    the DataGraph class is extended with some attributes to label the type of
    each node.

    However, the graph may be undirected (links from target nodes to source
    nodes are allowed)

    """

    def __init__(self, label="dg", path=None, label_source=None,
                 label_target=None, load_data=True, edge_class='directed'):
        """
        Defines the superedge structure

        If a superedge exists in the given path, it is loaded. Otherwise,
        a new empty superegde is created.

        Parameters
        ----------
        label : str or None, optional (default='dg')
            Name os the superedge
        path : str or None, optional (default=None)
            Path to the folder that contains, or will contain, the graph data
        label_source : str or None, optional (default=None)
            Generic name of the source nodes
        label_target : str or None, optional (default=None)
            Generic name of the target nodes
        load_data : bool, optional (default=True)
            If True (default) the graph data are loaded.
            If False, only the graph metadata are loaded

        Attributes
        ----------
        n_source : int
            Number of source nodes
        n_target : int
            Number of target nodes
        label_source : str
            Generic label for the source nodes
        label_target : str
            Generic label for the target nodes

        Notes
        -----
        These are the specific attributes of the SEdge class. See the parent
        class documentation to see more attributes.

        The source and target nodes in the parent class attribute self.df_edges
        are located in columns 'Source' and 'Target', because these are the
        standard names for Gephi graphs. Thus, the names in label_source and
        label_target are not used in self.df_edges.
        """

        # Class attributes (besides those of the parent class)
        # These attributes are redundant, because they can be found in
        # self.metadata. I can consider removing them in the future.
        self.n_source = None
        self.n_target = None
        self.label_source = label_source
        self.label_target = label_target

        self.Xs = None         # Source feature matrix, one row per node
        self.Xt = None         # Target feature matrix, one row per node
        self.save_X = False   # By default, feature matrices will not be saved

        # Call the initialization method in the parent class
        # This will update self.n_source and self.n_target
        super().__init__(label, path, load_data)

        if path:
            # paths to nodes, edges, metadata and feature matrix
            self.path2Xs = os.path.join(path, 'source_model_sparse.npz')
            self.path2Xt = os.path.join(path, 'target_model_sparse.npz')

        if (self.n_nodes > 0
                and self.metadata['graph']['category'] != 'bigraph'):
            logging.error(f"---- {label} is not a bipartite datagraph")

        if self.n_nodes == 0:
            logging.info("---- Empty bigraph created")
            self.n_source = 0
            self.n_target = 0

            # Default labels
            if label_source is None:
                self.label_source = "Source"
            if label_target is None:
                self.label_target = "Target"

            # Update metadata dictionary
            self.metadata['graph'].update({'category': 'bigraph',
                                           'subcategory': None,
                                           'source': self.label_source,
                                           'target': self.label_target})
            self.update_metadata()

        else:
            # Check consistency of label names
            ls = self.metadata['graph']['source']    # Abbreviation
            lt = self.metadata['graph']['target']    # Abbreviation
            if ((label_source is not None and ls != label_source)
                    or (label_target is not None and lt != label_target)):
                logging.warning(
                    "---- You have entered a new label for source or target ",
                    "nodes.\n",
                    "Labels cannot be changed, so new labels will be ignored.")

            self.label_source = self.metadata['graph']['source']
            self.label_target = self.metadata['graph']['target']

            self.n_source = self.metadata['nodes']['n_source']
            self.n_target = self.metadata['nodes']['n_target']

        return

    def update_metadata(self):
        """
        Updates metadata dictionary with the self variables directly computed
        from df_nodes and df_edges
        """

        # Update metadata entries of a generic graph
        super().update_metadata()

        # Update bigraph specific entries
        # I need to check if self.n_source or self.n_target are None because
        # this method is called by __init__(), before the actual number of
        # source and target nodes are known.
        if self.n_source is not None:
            self.metadata['nodes']['n_source'] = self.n_source
        if self.n_target is not None:
            self.metadata['nodes']['n_target'] = self.n_target

        return

    def _computeK(self):
        """
        Given the list of affinity values in self.affinity, computes a sparse
        symmetric affinity matrix.

        This method modifies computeK() from the parent class.

        The parent class could compute a square affinity matrix including
        affinities between all nodes in the bipartite graph. On the contrary,
        this method computes a non-square matrix including affinities between
        source and target nodes only.

        Returns
        -------
        K : sparse csr matrix
            Affinity matrix
        """

        # Recompute indices from self.edge_ids
        k_ids = [(s, t - self.n_source) for s, t in self.edge_ids]
        K = scsp.csr_matrix((self.weights, zip(*k_ids)),
                            shape=(self.n_source, self.n_target))

        return K

    def _computeD(self):
        """
        Given the list of affinity values in self.affinity, computes a sparse
        symmetric "distance" matrix, were distances are computed as one minus
        affinities.

        This method modifies computeK() from the parent class.

        The parent class could compute a square affinity matrix including
        affinities between all nodes in the bipartite graph. On the contrary,
        this method computes a non-square matrix including affinities between
        source and target nodes only.

        Returns
        -------
        D : sparse csr matrix
            Distance matrix
        """

        # EPS is used here to avoid zero distances.
        # "For weighted graphs the edge weights must be greater than zero."
        # (NetworkX documentation, betweenness centrality)
        # Maybe I should use even higher values
        dists = [1 - x + EPS for x in self.weights]

        # Recompute indices from self.edge_ids
        k_ids = [(s, t - self.n_source) for s, t in self.edge_ids]
        D = scsp.csr_matrix((dists, zip(*k_ids)),
                            shape=(self.n_source, self.n_target))

        return D

    def get_terminals(self):
        """
        Returns the name of the source and target snodes of a given sedge

        Returns
        -------
        s_label : str
            Name of the source snode
        t_label : str
            Name of the target snode
        """

        # Get names of the terminal snodes
        s_label = self.metadata['graph']['source']
        t_label = self.metadata['graph']['target']

        return s_label, t_label

    def get_source_nodes(self):
        """
        Get list of source nodes

        Returns
        -------
        list of source nodes
        """

        return [x for x in self.df_nodes[self.REF] if x[0] == 's']

    def get_target_nodes(self):
        """
        Get list of target nodes

        Returns
        -------
        list of target nodes
        """

        return [x for x in self.df_nodes[self.REF] if x[0] == 't']

    def set_nodes(self, nodes_orig=[], nodes_dest=[], Xs=None, Xt=None,
                  save_T=False):
        """
        Loads a superedge with a given set of source and target nodes.

        The new sets of nodes overwrite any existing ones.

        Parameters
        ----------
        nodes_orig : list, optional (default=[])
            Source nodes
        nodes_dest : list, optional (default=[])
            Target nodes
        Xs : array or None, optional (default=None)
            Source feature matrix: one row per source node, one column per
            feature
        Xt : array or None, optional (default=None)
            Target feature matrix: one row per source node, one column per
            feature
        save_T : bool, optional (default=False)
            If True, features matrices are saver into npz files.
        """

        # Remove duplicate names between source and target nodes:
        # if len(set(nodes_orig).intersection(set(nodes_dest))) > 0:
        #     logging.warning(
        #         '---- Source and target nodes have common names.\n' +
        #         '     Prefixes s_ and t_ will be used to discriminate ' +
        #         'types of nodes')

        #     nodes_orig = ['s_' + str(x) for x in nodes_orig]
        #     nodes_dest = ['t_' + str(x) for x in nodes_dest]
        #     prefix_names = True
        # else:
        #     prefix_names = False
        nodes_orig = ['s_' + str(x) for x in nodes_orig]
        nodes_dest = ['t_' + str(x) for x in nodes_dest]
        prefix_names = True

        # Default lists of nodes
        if len(nodes_orig) == 0:
            if Xs is not None:
                nodes_orig = list(range(Xs.shape[0]))
        if len(nodes_dest) == 0:
            if Xt is not None:
                nodes_dest = list(range(Xt.shape[0]))

        # Add nodes
        super().set_nodes(nodes_orig + nodes_dest, save_T=save_T)
        self.n_source = len(nodes_orig)
        self.n_target = len(nodes_dest)

        # Add atribute to nodes specifying the category of each node (node
        # from the source snode or node from the destination snode)
        Cat = ([self.label_source] * self.n_source
               + [self.label_target] * self.n_target)
        self.add_attributes('Cat', Cat)

        if Xs is not None:
            # Check consistency between nodes and features
            if Xs.shape[0] == self.n_source:
                self.Xs = Xs
            else:
                logging.error("-- -- The number of source nodes must be "
                              + "equal to the number of rows in Xs")
        if Xt is not None:
            # Check consistency between nodes and features
            if Xt.shape[0] == self.n_target:
                self.Xt = Xt
            else:
                logging.error("-- -- The number of target nodes must be "
                              + "equal to the number of rows in Xt")

        # Update metadata dictionary
        self.metadata['graph'].update({'prefix_names': prefix_names})
        self.update_metadata()

        return

    def set_edges(self, source_nodes, target_nodes, weights=None):
        """
        This method modifies set_edges from the parent class to test name
        collisions in source and target nodes.

        Parameters
        ----------
        source_nodes : list
            Source nodes
        target_nodes : list
            Target nodes
        weights : list or None, optional (default=None)
            Edge weights. If None, unit weights are assumed
        """

        # Testing name collision
        if self.metadata['graph']['prefix_names']:
            source_nodes = ['s_' + str(x) for x in source_nodes]
            target_nodes = ['t_' + str(x) for x in target_nodes]

        super().set_edges(source_nodes, target_nodes, weights)

        return

    def computeSimBiGraph(self, s_min=None, n_gnodesS=None, n_gnodesT=None,
                          n_edges=None, similarity='He2', g=1,
                          blocksize=25_000, useGPU=False, verbose=True):
        """
        Computes a sparse similarity bipartite graph for the self graph
        structure. The self graph must contain a T-matrix, self.T

        Parameters
        ----------
        s_min : float or None, optional (default=None)
            Similarity threshold. Edges link all data pairs with similarity
            higher than R. This forzes a sparse graph.
        n_gnodesS : int or None, optional (default=None)
            Number of nodes in the source subgraph.
            If None, all nodes are used
            If n_gnodesS < no. of rows in self.Xs, a random subsample is taken.
        n_gnodesT : int or None, optional (default=None)
            Number of nodes in the target subgraph.
            If None, all nodes are used.
            If n_gnodesT < no. of rows in self.Xt, a random subsample is taken.
        n_edges : int or None, optional (default=None)
            Target number of edges. n_edges is an alternative to radius.
            Only one of both must be specified (i.e., not None)
        similarity : str {'He2', 'He2->JS'}, optional (default='He2')
            Similarity measure used to compute affinity matrix
            Available options are:
            'He2' (1 minus squared Hellinger distance (self implementation));
            'He2->JS' (1 minus Jensen-Shannon (JS) divergence)
        g : float, optional (default=1)
            Exponent for the affinity mapping
        blocksize : int, optional (default=25_000)
            Size of each block for the computation of affinity values. Large
            sizes might imply a large memory consumption.
        useGPU : bool, optional (default=False)
            If True, matrix operations are accelerated using GPU
        verbose : bool, optional (default=True)
            (Only for he_neighbors_graph()). If False, block-by-block
            messaging is omitted
        """

        # Time control
        t0 = time()

        # Total number of items in the graph. Note that this can be higher
        # than n_nodes, which is the number of items in the subgraph
        if n_gnodesS is None:
            n_gnodesS = self.n_source
        if n_gnodesT is None:
            n_gnodesT = self.n_target

        # Select random samples from the whole graph
        # s_i2n and t_i2n are the lists of indices of the selected nodes
        # The name is a nemononic n = i2n[i] converts an index of the
        # list of selected nodes to an index of the complete list of nodes
        np.random.seed(3)    # For comparative purposes (provisional)
        if self.n_source <= n_gnodesS:
            s_i2n = range(self.n_source)
            n_gnodesS = self.n_source
        else:
            s_i2n = np.random.choice(range(self.n_source), n_gnodesS,
                                     replace=False)
        if self.n_target <= n_gnodesT:
            t_i2n = range(self.n_target)
            n_gnodesT = self.n_target
        else:
            t_i2n = np.random.choice(range(self.n_target), n_gnodesT,
                                     replace=False)

        # Take the topic matrices
        Xs = self.Xs[s_i2n]
        Xt = self.Xt[t_i2n]

        # ##########################
        # Computing Similarity graph
        sbg = SimBiGraph(Xs, Xt, blocksize, useGPU=useGPU)
        if not verbose:
            # Disable log messages
            root_logger = logging.getLogger()
            root_logger.disabled = True
        sbg.sim_graph(s_min=s_min, n_edges=n_edges, sim=similarity, g=g,
                      verbose=verbose)
        if not verbose:
            # Re-enable log messages
            root_logger = logging.getLogger()
            root_logger.disabled = False
        # NEW: indices in self.edge_ids refer to locations in self.nodes
        # OLD VERSION: self.edge_ids = sg.edge_ids
        self.edge_ids = [(s_i2n[i], t_i2n[j]) for i, j in sbg.edge_ids]
        self.weights = sbg.weights
        self.n_edges = len(self.edge_ids)

        # Create pandas structures for edges.
        # Each node is a project, indexed by field REFERENCIA.
        orig_REF = [self.nodes[m] for m, n in self.edge_ids]
        dest_REF = [self.nodes[n] for m, n in self.edge_ids]
        self.df_edges = pd.DataFrame(
            list(zip(orig_REF, dest_REF, self.weights)),
            columns=['Source', 'Target', 'Weight'])
        self.df_edges['Type'] = 'Undirected'

        # Update metadata. Note that there may be some keys in
        # self.metadata['graph'] that are not modified (like 'nodes')
        tm = time() - t0
        self.metadata['graph'].update({'subcategory': 'similarity'})
        # Update generig metadata entries
        self.update_metadata()
        # Update metadate entries that are specific of the similarity graph
        self.metadata['edges'].update({
            'metric': similarity,
            'R': float(sbg.R),    # float to avoid yaml errors with np.float
            'g': g,
            'time': tm,
            'n_sampled_nodes': n_gnodesS + n_gnodesT,
            'sampling_factor': (n_gnodesS + n_gnodesT) / self.n_nodes,
            'neighbors_per_sampled_node': (
                2 * self.n_edges / (n_gnodesS + n_gnodesT)),
            'density': 2 * self.n_edges / (self.n_source * self.n_target)})

        logging.info(f"-- -- Bipartite graph generated with {self.n_edges} "
                     f"edges in {tm} seconds")

        return

    def save_feature_matrix(self):
        """
        Save feature matrices in self.Xs and self.Xt, if they exist.
        """

        # Save equivalent feature matrix
        if self.Xs is not None:
            save_npz(self.path2Xs, self.Xs)
        if self.Xt is not None:
            save_npz(self.path2Xt, self.Xt)

        return

    # WARNING: The following methods have not been implemented for sedges yet.
    # The versions in the parent class are incomplete for sedges, because they
    # do not update the attributes that are specific of sedges (self.n_source
    # and self.n_target). For this reason, an error is raised.
    # The implementation of this methods is not difficult, but I left it for
    # the future...
    def add_single_node(self, node, attributes={}):
        """
        Add single node

        Parameters
        ----------
        node : str
            Node name
        attributes : dict, optional (default={})
            Dictionary of attributes
        """

        logging.error("---- Method add_single_node has not been implemented "
                      "for sedges. Since the method from the parent class "
                      " is not appropriate, no action is taken.")
        return

    def add_single_edge(self, source, target, weight=1, attributes={}):
        """
        Add single edge

        Parameters
        ----------
        source : str
            Source node name
        target : str
            Target node name
        weight : float, optional (default=1)
            Edge weight
        attributes : dict, optional (default={})
            Dictionary of attributes
        """

        logging.error("---- Method add_single_edge has not been implemented "
                      "for sedges. Since the method from the parent class "
                      " is not appropriate, no action is taken.")
        return

    def drop_single_node(self, node):
        """
        Add single node

        Parameters
        ----------
        node : str
            Node name
        """

        logging.error("---- Method drop_single_edge has not been implemented "
                      "for sedges. Since the method from the parent class "
                      " is not appropriate, no action is taken.")
        return

    def disconnect_nodes(self, source, target, directed=False):
        """
        Disconnect nodes by removing edges

        Parameters
        ----------
        source : str
            Source node name
        target : str
            Target node name
        directed : bool, optional (default=True)
            True if only edge source->target should be removed
        """

        logging.error("---- Method add_single_edge has not been implemented "
                      "for sedges. Since the method from the parent class is "
                      "not appropriate, no action is taken.")
        return

    # def computeBinFeatures(self, Ygraph, XYgraph, field, fitXY=False,
    #                        reverse=False, nPCA=None, normalize=True):
    #     """
    #     Given a Ygraph with nodes and an XYgraph connecting nodes in the self
    #     graph with nodes in the Ygraph, compute a feature matrix for the self
    #     graph.

    #     The feature vector for some node x in the self graph is computed as
    #     the binary link vector. The i-th component is nonzero if node x is
    #     connected to node i in the Ygraph.

    #     Args:
    #         :Ygraph:  A raw graph with a topic matrix in Ygraph.T.
    #         :XYgraph: A graph relating the X and y graphs. It should contain
    #                     XYgraph.nodes: A list of x-nodes.
    #                     XYgraph.df_nodes: a pandas structure with at list two
    #                         fields: 'id' (with the x-nodes) and field (with
    #                         the y-nodes)
    #         :field:   The field in the XYgraph that contains the y-nodes.
    #         :fitXY:   If False, all x-nodes are assumed to be in the XY-graph
    #                   are nodes of the self graph.
    #                   If True, the XY-graph is filtered to make sure that all
    #                   nodes in the XY-graph are nodes of the self graph.

    #                   If the constructor of the XY-graph has removed strange
    #                   x-nodes, you can take fitXY=False (which is default)
    #         :reverse: If true, input XYgraph is assumed to containg a Y-X
    #                   graph instead of a X-Y graph.
    #         :nPCA:    If None, no PCA is applied to the feature matrix.
    #                   If an integer, the feature vector are proyected to
    #                   the subspace of the nPCA principal components.
    #         :normalize: If True the nonzero value of feature vectors is not
    #                   unity: feature vectors are normalized to sum up to one.
    #                   If false no normalization is applied
    #                   If normalize = 2, normalization is done to sum up to
    #                   the square root of the original sum.
    #     """

    #     if not reverse:

    #         # Ynodes = list(set(XYgraph.df_nodes[field]))
    #         Ynodes = Ygraph.nodes

    #         # Compute inverse indices for the X and Y nodes
    #         X_to_idX = dict(zip(self.nodes, range(self.n_nodes)))
    #         Y_to_idY = dict(zip(Ynodes, range(len(Ynodes))))

    #         # Set of links from nodes existing in the self graph.
    #         if fitXY:
    #             # Remove x-nodes in XYgraph that are not self nodes
    #             # At the time or writing this, this step is not required
    #             # if the XYgraph is constructed by the SuperGraph class,
    #             # because the graph constructor (addSuperEdge) removes them.
    #             XY_links_all = zip(XYgraph.nodes, XYgraph.df_nodes[field])
    #             XY_links = filter(lambda x: x[0] in self.nodes, XY_links_all)
    #         else:
    #             XY_links = zip(XYgraph.nodes, XYgraph.df_nodes[field])

    #         # Take graph elements
    #         row = [X_to_idX[x[0]] for x in XY_links]
    #         col = [Y_to_idY[x[1]] for x in XY_links]
    #         data = [1.0] * len(XY_links)

    #     else:

    #         YXgraph = XYgraph     # Just to make things clear.

    #         # Ynodes = list(set(YXgraph.nodes))
    #         Ynodes = Ygraph.nodes

    #         # Compute inverse indices for the X and Y nodes
    #         X_to_idX = dict(zip(self.nodes, range(self.n_nodes)))
    #         Y_to_idY = dict(zip(Ynodes, range(len(Ynodes))))

    #         # Set of links from nodes existing in the self graph.
    #         if fitXY:
    #             # Remove x-nodes in XYgraph that are not self nodes
    #             # At the time or writing this, this step is not required
    #             # if the XYgraph is constructed by the SuperGraph class,
    #             # because the graph constructor (addSuperEdge) removes them.
    #             YX_links_all = zip(YXgraph.nodes, YXgraph.df_nodes[field])
    #             YX_links = filter(lambda x: x[1] in self.nodes, YX_links_all)
    #         else:
    #             YX_links = zip(YXgraph.nodes, YXgraph.df_nodes[field])

    #         # Take graph elements
    #         row = [X_to_idX[x[1]] for x in YX_links]
    #         col = [Y_to_idY[x[0]] for x in YX_links]
    #         data = [1.0] * len(YX_links)

    #     nx = self.n_nodes
    #     ny = len(Ygraph.nodes)

    #     # Construct the sparse feature matrix
    #     # self.B = csr_matrix((data, (row, col)), shape=(nx, ny)).toarray()
    #     self.B = csr_matrix((data, (row, col)), shape=(nx, ny))

    #     if normalize:
    #         # Normalize feature vectors to sum up to 1.
    #         d = 1.0 / np.asarray(np.sum(self.B, axis=1)).flatten()
    #         self.B = diags(d).dot(self.B)
    #     if normalize == 2:
    #         # Renormalize feature vectors. After multiplying by d, we devide
    #         # by sqrt(d), so as the overall normalization is by sqrt(d).
    #         # This is used to compute similarity functions that can take into
    #         # account the number of active Features in each node
    #         d2 = 1.0 / np.sqrt(d)
    #         self.B = diags(d2).dot(self.B)

    #     if nPCA is not None:
    #         pca = PCA(n_components=nPCA)
    #         logging.info("-- -- -- Reducing dimension with PCA ...")
    #         self.B = pca.fit_transform(self.B.toarray())

    #     logging.info(
    #         "-- -- -- Feature vectors have dimension {0}".format(ny))

    # def computeFeatures2(self, Ygraph, XYgraph, field, l_field, nPCA=None):
    #     """
    #     Given a Ygraph with nodes and an XYgraph connecting nodes in the self
    #     graph with nodes in the Ygraph, compute a feature matrix for the self
    #     graph.

    #     The feature vector for some node x in the self graph is computed as
    #     the  binary link vector. The i-th component is nonzero if node x is
    #     connected to node i in the Ygraph.

    #     The nonzero value of feature vectors are not unity: feature vectors
    #     are normalized to sum up to one.

    #     Args:
    #         :Ygraph:  A raw graph with a topic matrix in Ygraph.T.
    #         :XYgraph: A graph relating the X and y graphs. It should contain
    #                     XYgraph.nodes: A list of x-nodes.
    #                     XYgraph.df_nodes: pandas structure with at least two
    #                         fields: 'id' (with the x-nodes) & field (with the
    #                         y-nodes)
    #         :field:   The field in the XYgraph that contains the y-nodes.
    #     """

    #     # Ynodes = list(set(XYgraph.df_nodes[field]))

    #     # Compute inverse indices for the X and Y nodes
    #     X_to_idX = dict(zip(self.nodes, range(self.n_nodes)))
    #     # Y_to_idY = dict(zip(Ynodes, range(len(Ynodes))))

    #     # Set of links from nodes existing in the self graph.
    #     XY_links = zip(XYgraph.nodes, XYgraph.df_nodes[field])
    #     XY_links_red0 = filter(lambda x: x[0] in self.nodes, XY_links)

    #     # Replace every node y in the xy link by its representative
    #     # according to the l_field
    #     XY_links_red = []
    #     logging.info(
    #         "Entering loop with size {0}".format(len(XY_links_red0)))
    #     logging.info("Ygraph with size {0}".format(len(Ygraph.nodes)))

    #     for xy in XY_links_red0:

    #         # Get nodes linked to the destination node in the Ygraph
    #         if xy[1] in Ygraph.nodes:
    #             y_linked0 = Ygraph.df_nodes[
    #                 Ygraph.df_nodes[self.REF] == xy[1]][l_field].tolist()
    #             if y_linked0[0] is None:
    #                 y_linked = [xy[1]]
    #             else:
    #                 y_linked = y_linked0 + [xy[1]]

    #             # Take the first node in alphabetical order
    #             xy_new = (xy[0], min(y_linked))
    #             XY_links_red.append(xy_new)

    #     # Redefine the node list with the
    #     Ynodes = list(set([x[1] for x in XY_links_red]))
    #     Y_to_idY = dict(zip(Ynodes, range(len(Ynodes))))

    #     # Take graph elements
    #     row = [X_to_idX[x[0]] for x in XY_links_red]
    #     col = [Y_to_idY[x[1]] for x in XY_links_red]
    #     data = [1.0] * len(XY_links_red)

    #     nx = self.n_nodes
    #     ny = len(Ynodes)

    #     # Construct the sparse feature matrix
    #     # self.B = csr_matrix((data, (row, col)), shape=(nx, ny)).toarray()
    #     self.B = csr_matrix((data, (row, col)), shape=(nx, ny))

    #     if nPCA is not None:
    #         pca = PCA(n_components=20)
    #         logging.info("-- -- -- Reducing dimension with PCA ...")
    #         self.B = pca.fit_transform(self.B.toarray())

    #     logging.info(
    #         "-- -- -- Feature vectors have dimension {0}".format(ny))
