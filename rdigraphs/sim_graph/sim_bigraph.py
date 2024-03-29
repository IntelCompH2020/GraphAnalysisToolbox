#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # Python libraries
import logging
from time import time

import numpy as np
from scipy.sparse import issparse
# # import cupy.sparse
# # import GPUtil

# # Local imports
from rdigraphs.sim_graph.sim_graph import SimGraph

# memory_pool = cupy.cuda.MemoryPool()
# cupy.cuda.set_allocator(memory_pool.malloc)
# pinned_memory_pool = cupy.cuda.PinnedMemoryPool()
# cupy.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)

EPS = np.finfo(float).tiny

# Upper bound of several distance measures between probability
# distributions
R2_MAX_HE = 2   # Maximum squared Hellinger distance
R2_MAX_JS = 1   # Maximum squared JS divergence
R2_MAX_L2 = 2   # Maximum squared L2 distance
R_MAX_L1 = 2    # Maximum L1 distance


class SimBiGraph(SimGraph):

    """
    Generic class to generate similarity biparite graphs from data
    """

    def __init__(self, X, Y, blocksize=25_000, useGPU=False):
        """
        Stores the main attributes of a datagraph object and loads the graph
        data as a list of node attributes from a database

        Parameters
        ----------
        X : scipy.sparse.csr or numpy.array
            Matrix of node attribute vectors
        Y : scipy.sparse.csr or numpy.array
            Matrix of node attribute vectors
        blocksize : int, optional (default=25_000)
            Size (number of rows) of each block in blocwise processing.
        useGPU : bool, optional (default=False)
            If True, matrix operations are accelerated using GPU
        """

        # Call the initialization method in the parent class
        # This will update self.n_source and self.n_target
        super().__init__(X, blocksize, useGPU)

        # Feature vectors of the target nodes.
        self.Y = Y

        # ###############
        # Graph variables
        self.n_source = X.shape[0]
        self.n_target = Y.shape[0]
        self.s_min = None       # Similarity threshold       

        # ###############
        # Other variables

        # Variables for equivalence classes
        self.Yeq = None     # Reduced feature matrix (one row per equiv. class)

        # Number of equivalence classes at source and target nodes
        self.n_clusters_source = None
        self.n_clusters_target = None
        # Number of distinct nonzero feature patterns at source & target nodes
        self.n_preclusters_source = None
        self.n_preclusters_target = None

        # Equivalent classes of each of the source and target nodes
        self.cluster_ids_source = None
        self.cluster_ids_target = None

        return

    def sim_graph(self, s_min=None, n_edges=None, **kwargs):
        """
        Computes a sparse graph for a given radius or for a given number of
        edges

        Parameters
        ----------
        s_min : float or None, optional (default=None)
            Similarity threshold. Edges link all data pairs with similarity
            higher than R. This forzes a sparse graph.
        n_edges : int
            Number of edges
        """

        # Check if the selected similarity measure is available
        if kwargs['sim'] in ['He2', 'He2->JS']:
            super().sim_graph(s_min=s_min, n_edges=n_edges, **kwargs)
        else:
            logging.error(f"-- -- Similarity {kwargs['sim']} is not "
                          "available for bipartite graphs")

        return

    def _compute_sim_graph_from_threshold(
            self, s_min=None, sim='He2', mapping='linear', g=1, verbose=True):
        """
        Computes a sparse graph for the self graph structure.
        The self graph must contain a feature matrix, self.X

        Parameters
        ----------
        s_min : float or None, optional (default=None)
            Similarity threshold. Edges link all data pairs with similarity
            higher than R. This forzes a sparse graph.
        sim : string
            Similarity measure used to compute affinity matrix
            Available options are:

            'He2', 1 minus squared Hellinger distance (self implementation)

            'He2->JS', same as He-Js, but using the self implementation of He

        g : float
            Exponent for the affinity mapping
        verbose : boolean, optional (default=True)
            (Only for he_neighbors_graph()). If False, block-by-block
            messaging is omitted

        Returns
        -------
        self : object
            Changes in attributes self.edge_ids (List of edges, as pairs (i, j)
            of indices) and self.weights (list of affinity values for each pair
            in edge_ids)
        """

        logging.info(f"-- Computing {sim} graph with {self.n_nodes} nodes")
        t0 = time()

        # This is just to abbreviate
        X = self.X
        Y = self.Y

        # Compute edges and weights of the similarity graph
        if sim == 'He2':
            R2 = self.sim2div(s_min, g=g, mapping=mapping, B=R2_MAX_HE)
            self.edge_ids, d2 = self.he_neighbors_bigraph(
                X, Y, R2=R2, mode='distance', verbose=verbose)
            # Transform list of distances into similarities
            self.weights = self.div2sim(
                d2, mapping=mapping, g=g, B=R2_MAX_HE)

        elif sim == 'He2->JS':
            R2 = self.sim2div(s_min, g=g, mapping=mapping, B=R2_MAX_JS)
            R_he = np.sqrt(2 * R2)
            logging.info(f'-- -- Hellinger-radius bound for JS: {R_he}')
            self.connectivity_graph(R=R_he, metric='He2', verbose=verbose)
            n_edges = len(self.edge_ids)
            logging.info(f"-- -- Computing affinities for {n_edges} edges...")
            self.edge_ids, self.weights = self.JS2_affinity(X, Y, R2=R2, g=g)
            n_edges = len(self.edge_ids)
            logging.info(f"      reduced to {n_edges} edges")

        else:
            raise ValueError('Unknown similarity measure')

        logging.info(f'      Computed in {time()-t0:.4f} seconds')

        return

    def connectivity_graph(self, R=None, metric='JS', verbose=True):
        """
        Computes a sparse connectivity graph for the self graph structure.
        The self graph must contain matrix self.X

        Parameters
        ----------
        R : float
            Radius. Edges link all data pairs at distance lower than R
            This is to forze a sparse graph.
        metric : string
            Similarity measure used to compute affinity matrix
            Available options are:

            'He2', 1 minus squared Hellinger distance (self implementation)

        verbose : boolean, optional (default=True)
            (Only for he_neighbors_graph()). If False, block-by-block
            messaging is omitted

        Returns
        -------
        self : object
            Changes in attributes self.edge_ids (List of edges, as pairs (i, j)
            of indices) and self.weights (list of affinity values for each pair
            in edge_ids)
        """

        logging.info(f"-- -- Computing {metric} connectivity graph with "
                     f"{self.n_nodes} nodes")

        # #############################
        # Computing Connectivity Matrix

        # Compute the connectivity graph of all pair of nodes at distance
        # below R
        t0 = time()
        if metric == 'He2':
            self.edge_ids = self.he_neighbors_bigraph(
                self.X, self.Y, R2=R**2, mode='connectivity', verbose=verbose)
        else:
            logging.error("connectivity_graph: Unknown similarity measure")
            exit()

        n_edges = len(self.edge_ids)
        logging.info(f'      Computed in {time()-t0} seconds')
        logging.info(f"-- -- Connectivity graph generated with {self.n_nodes} "
                     f"nodes and {n_edges} edges")

        return

    def _compute_sim_graph_from_nedges(
            self, n_edges, sim='He2', mapping='linear', g=1, verbose=True):
        """
        Computes a sparse graph for a fixed number of edges.

        It computes the sparse graph from matrix self.X. The distance threshold
        R to sparsify the graph is chosend in such a way that the resultin
        graph has n_edges edges.

        Parameters
        ----------
        n_edges:    int
            Target number of edges
        sim : string
            Similarity measure used to compute affinity matrix
            Available options are:

            'He2', 1 minus squared Hellinger distance (self implementation)

            'He2->JS', same as He-Js, but using the self implementation of He

        g : float
            Exponent for the affinity mapping (not used for 'Gauss')
        verbose : boolean
            (Only for he_neighbors_graph()). If False, block-by-block
            messaging is omitted

        Returns
        -------
        self : object
            Changes in attributes self.edge_ids (List of edges, as pairs (i, j)
            of indices) and self.weights (list of affinity values for each pair
            in edge_ids)
        """

        # Compute sub graph
        size_ok = False

        # Since there is not a direct and exact method to compute a graph with
        # n_edges, we will try to find a graph with approximately n_edges_top,
        # where n_edges_top > n_edges
        n_edges_top = n_edges     # Initial equality, but revised below

        while not size_ok:
            # Excess number of edges.
            n_edges_top = int(1.2 * n_edges_top) + 1

            # ##############################################################
            # First goal: find a dense graph, with less nodes but n_edges...

            # Initial number of source and target nodes to get a dense graph
            # with n_edges_top
            n_s = min(
                int(np.sqrt(n_edges_top * self.n_source / self.n_target)) + 1,
                self.n_source)
            n_t = min(
                int(np.sqrt(n_edges_top * self.n_target / self.n_source)) + 1,
                self.n_target)

            # Initial similarity threshold to guarantee a dense graph
            s_min = -0.01    # Any number smaller than any lower bound ...

            # Take n_n nodes selected at random
            np.random.seed(3)
            idx = sorted(np.random.choice(range(self.n_source), n_s,
                                          replace=False))
            X_sg = self.X[idx]

            idy = sorted(np.random.choice(range(self.n_target), n_t,
                                          replace=False))
            Y_sg = self.Y[idy]

            # Compute dense graph
            subg = SimBiGraph(X_sg, Y_sg, blocksize=self.blocksize)
            subg.sim_graph(
                s_min=s_min, sim=sim, mapping=mapping, g=g, verbose=verbose)

            # Check if the number of edges is highet than the target. This
            # should not happen. Maybe only for X_sg with repeated rows
            n_e = len(subg.weights)
            size_ok = ((n_e >= n_edges)
                       | (n_s == self.n_source) & (n_t == self.n_target))
            if not size_ok:
                logging.info(f'-- -- Insufficient graph with {n_e} < '
                             f'{n_edges} edges. Trying with more nodes')

        # Scale factor for the expected number of edges in the second trial.
        # The main idea is the following (assume, for simplicity, a square
        # graph: if, for a fixed theshold R, we get two graphs, one with n
        # nodes and e edges, and other with n' nodes and e' edges, we can
        # expect
        #   n'**2 / n**2 = e' / e
        # (with approximate equality). We have n and n' (i.e. self.n_nodes and
        # n_n) and e (i.e. the target n_edges_top). Thuse, we can compute e'
        # (i.e., n_edges_subg below)
        alpha = self.n_source * self.n_target / (n_s * n_t)
        n_edges_subg = int(n_edges_top / alpha)
        # Since n_e > n_edges_sub, we can compute the threshold value providing
        # n_edges_subg, which should be approximately equal to the one
        # providing n_edges_top

        if n_s == self.n_source and n_t == self.n_target:
            size_ok = True
            # The final graph has been computed. Just read it from
            self.edge_ids = subg.edge_ids
            self.weights = subg.weights
        else:
            size_ok = False
            # Compute the similarity value to get n_edges_subg
            s = sorted(list(zip(subg.weights, range(n_e))), reverse=True)
            s_min = s[n_edges_subg - 1][0]

        while not size_ok:

            # Compute graph with the target number of links
            logging.info(f'-- -- Trying threshold s_min = {s_min:.4f}...')
            self.sim_graph(
                s_min=s_min, sim=sim, mapping=mapping, g=g, verbose=verbose)

            size_ok = (len(self.weights) >= n_edges)

            if not size_ok:
                # It the method failed, reduce the similarity threshold
                s_min = 0.8 * s_min
                # This is to deal with the case R = 0
                if s_min == 1:
                    # Take the value of R corresponding to the highest w less
                    # than 1
                    s_min = np.max([x if x < 1 else 0 for x in subg.weights])
                    if s_min > 0:
                        s_min = s_min
                    else:
                        # If R is still zero, take a fixed value.
                        s_min = 0.99
                logging.warning(
                    f'-- -- Too sparse graph. Trying R = {s_min}...')

        # If we are here, we have got a graph with more than n_edges edges and
        # all nodes. We just need to fit the threshold to get exactpli n_edges
        n_e = len(self.weights)
        w = sorted(list(zip(self.weights, range(n_e))), reverse=True)
        w = w[:n_edges]

        if len(w) > 0:
            w_min = w[-1][0]
            ew = [x for x in zip(self.edge_ids, self.weights) if x[1] >= w_min]
        else:
            ew = []

        if len(ew) > 0:
            self.edge_ids, self.weights = zip(*ew)
        else:
            self.edge_ids, self.weights = [], []

        # Maybe not used inside the class, but useful outside
        self.s_min = s_min

        return

    def cluster_equivalent_nodes(self, reduceX=False):
        """
        Computes two graphs where each node is formed by all nodes at zero
        distance. One graph uses the source nodes, the other the target nodes

        Parameters
        ----------
        reduceX : boolean
            If True, it computes self.Xeq, a data matrix without rows at zero
            distance
        """

        # Compute equivalent classes from the source nodes
        sg = SimGraph(self.X, blocksize=self.blocksize)
        sg.cluster_equivalent_nodes(reduceX=reduceX)
        self.cluster_ids_source = sg.cluster_ids
        self.n_clusters_source = sg.n_clusters
        self.n_preclusters_source = sg.n_preclusters
        if reduceX:
            self.Xeq = sg.Xeq

        # Compute equivalent classes from the target nodes
        sg = SimGraph(self.Y, blocksize=self.blocksize)
        sg.cluster_equivalent_nodes(reduceX=reduceX)
        self.cluster_ids_source = sg.cluster_ids
        self.n_clusters_source = sg.n_clusters
        self.n_preclusters_source = sg.n_preclusters
        if reduceX:
            self.Yeq = sg.Xeq

        return

    def he_affinity(self, X, Y=None, R2=10, mapping='linear', g=1):
        """
        Compute all Hellinger's affinities between all feature vectors in X
        and all feature vectors in self.Y

        It assumes that all attribute vectors are normalized to sum up to 1
        Attribute matrix X can be sparse

        Note that self.Y is not passed as argument. This is to facilitate the
        use of methods from the parent class.

        Parameters
        ----------
        X : numpy array
            Input matrix of probabilistic attribute vectors
        Y : numpy array or None, optional (default=None)
            Input matrix of probabilistic attribute vectors. If None, it is
            assumed Y=X
        R : float, optional (default=1)
            Radius (maximum L1 distance. Edges with higher distance are
            removed)
        g : float, optional (default=1)
            Exponent for the final affinity mapping
        rescale : boolean, optional (deafault=True)
            If True, affinity values are rescaled so that the minimum value is
            zero and the maximum values is one.

        Returns
        -------
        edge_id : list of tuples
            List of edges
        weights : list
            List of edge weights

        Notes
        -----
        If X is sparse, self.Y must be sparse too.
        If X is dense, self.Y must be dense too.
        """

        # ################################
        # Compute affinities for all edges

        # I take the square root here. This is inefficient if X has many
        # rows and just af few edges will be computed. However, we can
        # expect the opposite (the list of edges involves the most of the
        #  nodes).
        Zx = np.sqrt(X)
        Zy = np.sqrt(Y)

        # Divergences are compute by blocks. This is much faster than a
        # row-by-row computation, specially when X is sparse.
        d2_he = []
        for i in range(0, len(self.edge_ids), self.blocksize):
            edge_ids = self.edge_ids[i: i + self.blocksize]

            # Take the (matrix) of origin and destination attribute vectors
            i0, i1 = zip(*edge_ids)

            if issparse(X):
                P = Zx[list(i0)].toarray()
                Q = Zy[list(i1)].toarray()
            else:
                P = Zx[list(i0)]
                Q = Zy[list(i1)]

            # Squared Hellinger's distance
            # The maximum is used here just to avoid 2-2s<0 due to
            # precision errors
            s = np.sum(P * Q, axis=1)
            d2_he += list(np.maximum(2 - 2 * s, 0))

        # #########
        # Filtering

        # Filter out edges with He distance above R.
        ed = [z for z in zip(self.edge_ids, d2_he) if z[1] < R2]
        if len(ed) > 0:
            edge_id, d2 = zip(*ed)
        else:
            edge_id, d2 = [], []

        # ####################
        # Computing affinities

        # Transform squared distances into affinity values.
        # Note that we set B equal to the tightest bound on the
        # squared He distance
        weights = self.div2sim(d2, mapping=mapping, g=g, B=R2_MAX_HE)

        return edge_id, weights
