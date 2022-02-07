#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Python libraries
import numpy as np

import os
import logging
from time import time

from scipy.sparse import issparse
from sklearn.neighbors import radius_neighbors_graph
from collections import Counter, defaultdict
import igraph   # "install -c conda-forge python-igraph"

import matplotlib.pyplot as plt

# import cupy.sparse
# import GPUtil

# memory_pool = cupy.cuda.MemoryPool()
# cupy.cuda.set_allocator(memory_pool.malloc)
# pinned_memory_pool = cupy.cuda.PinnedMemoryPool()
# cupy.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)

# Local imports
from rdigraphs.sim_graph.th_ops import ThOps  # base_ThOps as ThOps

EPS = np.finfo(float).tiny


def JSdist(p, q):
    """
    Compute the Jensen-Shannon distance between probability distributions p
    and q.
    It assumes that both p and q are normalized and sum up to 1

    Parameters
    ----------
    p : numpy array
        Probability distribution
    q : numpy array
        Probability distribution (with the same size as p)

    Returns
    -------
    d : float
        JS distance
    """

    pe = p + EPS
    qe = q + EPS
    m = 0.5 * (pe + qe)
    # I used entropy method in older versions, but it is much slower.
    # D = 0.5 * (entropy(p, m, base=2) + entropy(q, m, base=2))
    D = 0.5 * (np.sum(pe * np.log2(pe / m)) + np.sum(qe * np.log2(qe / m)))

    return np.sqrt(D)


def JSdist_sp(p, q):
    """
    Compute the Jensen-Shannon distance between probability distributions p
    and q.
    It assumes that both p and q are normalized and sum up to 1 p and q can
    be sparse vectors (this is the main difference wrt JSdist())

    Parameters
    ----------
    p : numpy array or sparse vector
        Probability distribution
    q : numpy array or sparse vector
        Probability distribution (with the same size as p)

    Returns
    -------
    d : float
        JS distance
    """

    pi = p.toarray().flatten() + EPS
    qi = q.toarray().flatten() + EPS
    m = 0.5 * (pi + qi)
    D = 0.5 * (np.sum(pi * np.log2(pi / m)) + np.sum(qi * np.log2(qi / m)))

    return np.sqrt(D)


class SimGraph(ThOps):

    """
    Generic class to generate similarity graphs from data
    """

    def __init__(self, X, blocksize=25_000, useGPU=False, tmp_folder=None,
                 save_every=1e300):
        """
        Stores the main attributes of a class instance

        Parameters
        ----------
        X : scipy.sparse.csr or numpy.array
            Matrix of node attribute vectors
        blocksize : int, optional (default=25_000)
            Size (number of rows) of each block in blocwise processing.
        useGPU : bool, optional (default=False)
            If True, matrix operations are accelerated using GPU
        tmp_folder : str or None, optional (defautl = None)
            Name of the folder to save temporary files
        save_every : int, optional (default=0)
            Maximum size of the growing lists. The output lists are constructed
            incrementally. To avooy memory overload, growing lists are saved
            every time they reach this size limit. The full liests are thus
            incrementally saved in files.
            The default value is extremely large, which de facto implies no
            temporary saving.
        """

        # Call the initialization method from the parent class to set the
        # self.blocksize attribute
        super().__init__(blocksize, useGPU, tmp_folder, save_every)

        # ###############
        # Graph variables
        self.X = X                        # Data matrix
        self.n_nodes, self.dim = X.shape  # Data dimensions
        self.R = None                     # Radius (distance threshold)

        # ###############
        # Other variables

        # Edges and weights
        self.edge_ids = None   # List of edges, as pairs (i, j) of indices.
        self.weights = None

        # Variables for equivalence classes
        self.Xeq = None     # Reduced feature matrix (one row per equiv. class)
        self.n_clusters = None     # Number of equivalence classes
        self.n_preclusters = None  # No. of distinct nonzero feature patterns
        self.cluster_ids = None    # Equivalence class of each node
        # WARNING: the following variable does not represent the number of
        # edges in the complete graph, but possibly of the equivalence graph.
        # (TBD: rename this variable)
        self.n_edges = None       # Number of edges of the equivalence graph

        return

    def _d2_to_sim(self, d2, sim, g=1, rescale=False, R=None):
        """
        Transforms a list of squared distance values into a list of similarity
        values

        Parameters
        ----------
        d2 : list
            A list of squared distance values
        sim : string
            Similarity measure. It is used to compute the radius bound
        g : int or float, optional (default=1)
            Power factor. Use g != 1 to apply nonlinear mapping
        rescale : boolean, optional (default=False)
            If True, similarities are rescaled such that distance R is mapped
            to zero similarity
        R : float or None, optional (default=None)
            (only for rescale=True). Radius bound

        Returns
        -------
        s :  list
            A list of similarity values with the same size than d
        """

        # Here we need to reassign R to avoid R >> 2 (which is the max
        # He value). Otherwise, normalization z/R**2 could be harmful.
        if sim in ['JS', 'He->JS', 'He2->JS', 'l1->JS']:
            Rmax = 1
            R2max = 1
        elif sim in ['l1']:
            Rmax = 2
            R2max = 4
        elif sim in ['He', 'He2', 'l2']:
            Rmax = np.sqrt(2)
            R2max = 2
        else:
            logging.error("-- -- Error in _d2_to_sim: unknown similarity")

        if rescale:
            Rmax = min(R, Rmax)
            R2max = min(R**2, R2max)

        # Compute similarities from distances
        if g != 1:
            # s = [(1 - x**g / Rmax**(2 * g)) for x in d2]
            s = [(1 - x**g / R2max**g) for x in d2]
        else:
            # This is a particular case, but it is the default case, and can
            # be computed avoiding x**g
            # s = [(1 - x / Rmax**2) for x in d2]
            s = [(1 - x / R2max) for x in d2]

        return s

    def computeGraph(self, R=None, n_edges=None, **kwargs):
        """
        Computes a sparse graph for a given radius or for a given number of
        edges

        Parameters
        ----------
        R : float
            (only for rescale=True). Radius bound
        n_edges : int
            Number of edges
        """

        if R is not None:
            # Call the standard graph computation method
            self.R = R
            self._computeGraph(R, **kwargs)
        elif n_edges is not None:
            self._compute_graph_from_nedges(n_edges, **kwargs)
        else:
            logging.error("-- -- At least R or n_edges must be specified")

        return

    def _computeGraph(self, R=None, sim='JS', g=1, th_gauss=0.1, rescale=False,
                      verbose=True):
        """
        Computes a sparse graph for the self graph structure.
        The self graph must containg a T-matrix, self.T

        Parameters
        ----------
        R : float
            Radius. Edges link all data pairs at distance lower than R
            This is to forze a sparse graph.
        sim : string
            Similarity measure used to compute affinity matrix
            Available options are:

            'JS', 1 minus Jensen-Shannon (JS) divergence (too slow);

            'l1', 1 minus l1 distance

            'He', 1-squared Hellinger's distance (sklearn-based implementation)

            'He2', 1 minus squared Hellinger distance (self implementation)

            'Gauss', an exponential function of the squared l2 distance

            'l1->JS', same as JS, but the graph is computed after pre-selecting
            edges using l1 distances and a theoretical bound

            'He->JS', same as JS, but the graph is computed after preselecting
            edges using Hellinger's distances and a theoretical bound

            'He2->JS', same as He-Js, but using the self implementation of He

        g : float
            Exponent for the affinity mapping (not used for 'Gauss')
        th_gauss : float
            Similarity threshold All similarity values below this threshold are
            set to zero. This is only for the gauss method, the rest of them
            compute the threshold automatically from R).
        rescale : boolean, optional (default=False)
            If True, affinities are computed from distances by rescaling
            values so that the minimum is zero and maximum is 1.
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

        logging.info(f"-- Computing graph with {self.n_nodes} nodes")
        logging.info(f"-- Similarity measure: {sim}")

        # #########################
        # Computing Distance Matrix

        # This is just to abbreviate
        X = self.X

        # Select Distance measure for radius_neighbor_graph
        if sim in ['Gauss', 'He', 'He->JS']:
            d = 'l2'     # Note: l2 seems equivalent to minkowski (p=2)
        elif sim in ['l1', 'l1->JS']:
            d = 'l1'     # Note: l1 seems equivalent to manhattan
        elif sim == 'JS':
            if issparse(X):
                logging.warning(
                    'At the time of writing this code, SKLEARN does not ' +
                    'admit callable metrics for sparse inputs. Thus, an ' +
                    'error will raise very likely right now ...')
                d = JSdist_sp
            else:
                d = JSdist
        elif sim in ['He2', 'He2->JS']:
            # No distance metric is selected, because a proper implementation
            # is used instead sklearn
            pass
        else:
            logging.error("computeTsubGraph ERROR: Unknown similarity measure")
            exit()

        # Select secondary radius
        if sim == 'l1->JS':
            R0 = np.sqrt(8 * np.log(2)) * R
            # Refined R0. Not relevant effect for small R0
            R0 = (12 / np.sqrt(2) * np.sqrt(np.sqrt(1 + R0**2 / 36) - 1))
            logging.info(f'-- -- L1-radius bound for JS: {R0}')
        elif sim in ['He->JS', 'He2->JS']:
            R0 = np.sqrt(2) * R
            logging.info(f'-- -- Hellinger-radius bound for JS: {R0}')
        else:
            # The target radius
            R0 = R

        # Compute the connectivity graph of all pair of nodes at distance
        # below R0
        # IMPORTANT: Note that, despite radius_neighbors_graph has an option
        # 'distance' that returns the distance values, it cannot be used in
        # any case because the distance matrix does not distinghish between
        # nodes at distance > R0 and nodes at distance = 0
        t0 = time()
        logging.info(f'-- -- Computing neighbors_graph ...')
        if sim == 'He2->JS':
            self.edge_ids = self.he_neighbors_graph(
                X, R0, mode='connectivity', verbose=verbose)
        elif sim == 'He2':
            self.edge_ids, he_dist2 = self.he_neighbors_graph(
                X, R0, mode='distance', verbose=verbose)
        elif sim in ['He', 'He->JS']:
            # We must compute the connectivity graph because module
            # radius_neighbors_graph looses edges between nodes at zero
            # distance
            D = radius_neighbors_graph(np.sqrt(X), radius=R0,
                                       mode='connectivity', metric=d)
        elif sim in ['l1', 'l1->JS', 'Gauss', 'JS']:
            D = radius_neighbors_graph(X, radius=R0, mode='connectivity',
                                       metric=d)

        logging.info(f'       in {time()-t0} seconds')

        # ##############################################
        # From distance matrix to list of weighted edges

        if sim not in ['He2', 'He2->JS']:
            # Compute lists with origin, destination and value for all edges in
            # the graph affinity matrix.
            orig_id, dest_id = D.nonzero()

            # Since the graph is undirected, we select ordered pairs orig_id,
            # dest_id only
            self.edge_ids = list(filter(lambda i: i[0] < i[1],
                                        zip(orig_id, dest_id)))

        # ####################
        # Computing Affinities

        n_edges = len(self.edge_ids)
        logging.info(f"-- -- Computing affinities for {n_edges} edges...")
        t0 = time()

        if sim in ['JS', 'l1->JS', 'He->JS', 'He2->JS']:
            # For methods ->JS, the distance computed by the neighors_graph
            # method is not the target distance JS.
            # A new self.edge_ids is returned because the function filters out
            # affinity values below th.
            self.edge_ids, self.weights = self.JS_affinity(
                X, R=R, g=g, rescale=rescale)

        elif sim == 'He':
            self.edge_ids, self.weights = self.he_affinity(
                X, R=R, g=g, rescale=rescale)

        elif sim == 'He2':
            # Transform list of distances into similarities
            self.weights = self._d2_to_sim(he_dist2, sim, g, rescale, R)
        elif sim == 'l1':
            self.edge_ids, self.weights = self.l1_affinity(
                X, R=R, g=g, rescale=rescale)

        elif sim == 'Gauss':
            self.edge_ids, self.weights = self.l2_affinity(X, R, th_gauss)

        n_edges = len(self.edge_ids)
        logging.info(f"      reduced to {n_edges} edges")
        logging.info(f'      Computed in {time()-t0} seconds')
        logging.info(f"-- -- Graph generated with {self.n_nodes} nodes and " +
                     f" {n_edges} edges")

        return

    def _compute_graph_from_nedges(self, n_edges, sim='JS', g=1, th_gauss=0.1,
                                   rescale=False, verbose=True):
        """
        Computes a sparse graph for a fixed number of edges.

        It computes the sparse graph form matrig X. The distance threshold R
        to sparsify the graph is chosend in such a way that the resulting graph
        has n_edges edges.

        Parameters
        ----------
        n_edges:    int
            Target number of edges
        sim : string
            Similarity measure used to compute affinity matrix
            Available options are:

            'JS', 1 minus Jensen-Shannon (JS) divergence (too slow);

            'l1', 1 minus l1 distance

            'He', 1-squared Hellinger's distance (sklearn-based implementation)

            'He2', 1 minus squared Hellinger distance (self implementation)

            'Gauss', an exponential function of the squared l2 distance

            'l1->JS', same as JS, but the graph is computed after pre-selecting
            edges using l1 distances and a theoretical bound

            'He->JS', same as JS, but the graph is computed after preselecting
            edges using Hellinger's distances and a theoretical bound

            'He2->JS', same as He-Js, but using the self implementation of He

        g : float
            Exponent for the affinity mapping (not used for 'Gauss')
        th_gauss : float
            Similarity threshold All similarity values below this threshold are
            set to zero. This is only for the gauss method, the rest of them
            compute the threshold automatically from R).
        rescale : boolean, optional (default=False)
            If True, affinities are computed from distances by rescaling
            values so that the minimum is zero and maximum is 1.
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
        # n_edges, we will try to find a graph with aproximately n_edges_top,
        # where n_edges_top > n_edges
        n_edges_top = n_edges     # Initial equality, but revised below
        while not size_ok:
            # Excess number of edges.
            n_edges_top = int(1.2 * n_edges_top)

            # ##############################################################
            # First goal: find a dense graph, with less nodes but n_edges...

            # Initial number of nodes to get a dense graph with n_edges_top
            n_n = min(int(np.sqrt(2 * n_edges_top)), self.n_nodes)
            # Initial radius to guarantee a dense graph
            R = 1e100    # A very large number higher than any upper bound ...

            # Take n_n nodes selected at random
            np.random.seed(3)
            idx = sorted(np.random.choice(range(self.n_nodes), n_n,
                                          replace=False))
            X_sg = self.X[idx]

            # Compute dense graph
            subg = SimGraph(X_sg, blocksize=self.blocksize)
            subg.computeGraph(R=R, sim=sim, g=g, rescale=False,
                              verbose=verbose)

            # Check if the number of edges is higher than the target. This
            # should not happen. Maybe only for X_sg with repeated rows
            n_e = len(subg.weights)
            size_ok = (n_e >= n_edges) | (n_n == self.n_nodes)
            if not size_ok:
                logging.info(f'-- -- Insufficient graph with {n_e} < ' +
                             f'{n_edges} edges. Trying with more nodes')

        # Scale factor for the expected number of edges in the second trial.
        # The main idea is the following: if, for a fixed theshold R, we get
        # two graphs, one with n nodes and e edges, and other with n' nodes and
        # e' edges, we can expect
        #   n'**2 / n**2 = e' / e
        # (with approximate equality).
        # Our graphs: the target graph, with n'=self.n_nodes and  e'n_edges
        #             the subgraph, with     n=n_n nodes and n_e edges
        # In ofder to satisfy the above relation, the subgraph should have:
        # n_edges_subg = e' n**2 / n'**2 = ...
        alpha = (n_n / self.n_nodes) ** 2
        # n_edges_subg = int(n_edges_top / alpha)
        n_edges_subg = int(n_e * alpha)
        # Since n_e > n_edges_sub, we can compute the threshold value providing
        # n_edges_subg, which should be approximately equal to the one
        # providing n_edges_top

        # Compute the threshold required to get the target links
        # This is the radius that should provide n_links edges.
        if sim in ['JS', 'He->JS', 'He2->JS', 'l1->JS']:
            Rmax = 1
        elif sim in ['l1']:
            Rmax = 2
        elif sim in ['He', 'He2']:
            Rmax = np.sqrt(2)

        if n_n == self.n_nodes:
            size_ok = True
            # The final graph has been computed. Just read it from
            self.edge_ids = subg.edge_ids
            self.weights = subg.weights
        else:
            size_ok = False
            # Compute the similarity value to get n_edges_subg
            w = sorted(list(zip(subg.weights, range(n_e))), reverse=True)
            w_min = w[n_edges_subg - 1][0]
            R = Rmax * (1 - w_min) ** (1 / (2 * g))

        while not size_ok:

            # Compute graph with the target number of links
            self.computeGraph(R=R, sim=sim, g=g, rescale=False,
                              verbose=verbose)

            size_ok = (len(self.weights) >= n_edges)

            if not size_ok:
                # It the method failed, increase the radius threshold
                R = 1.2 * R
                # This is to deal with the case R = 0
                if R == 0:
                    # Take the value of R corresponding to the highes w less
                    # than 1
                    w_min = np.max([x if x < 1 else 0 for x in subg.weights])
                    if w_min > 0:
                        R = Rmax * (1 - w_min) ** (1 / (2 * g))
                    else:
                        # If R is still zero, take a fixed value.
                        R = 0.01
                logging.warning(f'-- -- Too sparse graph. Trying R = {R}...')

        # If we are here, we have got a graph with more than n_edges edges and
        # all nodes. We just need to fit the threshold to get exactpli n_edges
        n_e = len(self.weights)
        w = sorted(list(zip(self.weights, range(n_e))), reverse=True)
        w = w[:n_edges]
        w_min = w[-1][0]

        ew = [x for x in zip(self.edge_ids, self.weights) if x[1] >= w_min]
        if len(ew) > 0:
            self.edge_ids, self.weights = zip(*ew)
        else:
            self.edge_ids, self.weights = [], []
        self.R = R

        # Rescale weights if necessary
        if rescale and R < Rmax:
            # The following equation results from computing square distances
            # from the weights, and recomputing the weight with rescaling:
            # d2**g = (1 - w) * Rmax**(2*g)
            # w = 1 - d2**g / R**(2*g)
            # which is equivalent to...
            if R > 0:
                self.weights = [max(1 - (1 - w) * (Rmax / R)**(2 * g), 0)
                                for w in self.weights]
            # (The max operator is just to ensure no negative weighs caused
            # by finite precision errors)
            # IF R = 0 then w_min = 1 and, thus, all weights should be 1,
            # which implies that no normalization is possible

        return

    def compute_id_graph(self, R=1e-100, verbose=True):
        """
        Computes identity graph.

        The identity graph connects nodes a and b with weight 1 iff a and b
        have the same feature vector in self.T. Otherwise, a and be are
        disconnected

        Parameters
        ----------
        R : float
            Radius. Edges link all data pairs at distance lower than R.
            It should be a very small value in order to link nodes with almost
            equal attribute. Nonzero values may be used to allow slight
            deviations from equality.
        verbose : boolean
            If False, block-by-block messaging is omitted in the call to
            he_neighbors graph.

        Returns
        -------
        self : object
            Updated attribute self.edge_ids (list of edges, as pairs (i, j) of
            indices)
        """

        logging.info(f"-- Computing graph with {self.n_nodes} nodes")

        # Compute the connectivity graph of all pair of nodes at distance
        # below R0
        t0 = time()
        self.edge_ids = self.he_neighbors_graph(
            self.X, R, mode='connectivity', verbose=verbose)
        logging.info(f'       in {time()-t0} seconds')
        logging.info(f"-- -- Graph generated with {self.n_nodes} nodes and " +
                     f"{len(self.edge_ids)} edges")

        return

    def cluster_equivalent_nodes(self, reduceX=False):
        """
        Computes a graph where each node is formed by all nodes at zero
        distance

        Parameters
        ----------
        reduceX : boolean
            If True, it computes self.Xeq, a data matrix without rows at zero
            distance
        """

        logging.info(f'-- -- Computing equivalence classes')

        # ###################################
        # Particion matrix by non-zero topics

        # This is just to abbreviate
        X = self.X
        n_nodes, n_topics = X.shape

        # We assign an integer to each set of nonzero topics, based on the
        # binary rows. To do so, we need an array with the powers of 2
        id_nodes, id_topics = (X > 0).nonzero()
        powers2 = [2**(n_topics - 1 - n) for n in range(n_topics)]
        # I wanted to do this, but is not allowed for very large ints:
        #     tclass = ((X > 0) @ powers2).T.tolist()[0]
        # So I do the following
        # Replace topic ids by their powers of two:
        pw_ids = [powers2[n] for n in id_topics]
        partition0 = [0] * n_nodes
        for i, node in enumerate(id_nodes):
            partition0[node] += pw_ids[i]

        # The elements of the partition are named preclusters to avoid
        # confusion with the final clusters.
        # - Preclusters group all nodes with the same nonzero topics
        #   (i.e., with the same integer value in partition0)
        # - Clusters group all nodes at zero distance
        # Each precluster will be particioned into clusters.
        precluster_sizes = Counter(partition0)

        # Compute inverse dictionary
        pc2nodes = defaultdict(list)
        for i, s in enumerate(partition0):
            pc2nodes[s].append(i)

        # edge_ids = []     # List of edges
        # Cluster of each node
        cluster_ids = np.zeros((n_nodes, 1)).astype(int)

        n_clusters = 0    # Counter of clusters (equivalence classes)
        n_edges = 0       # Counter of edges
        n = 0             # Node counter (only for status printing)

        for pc, pc_size in precluster_sizes.items():

            # Get data submatrix of particion pc
            ind = pc2nodes[pc]
            n += len(ind)

            if pc_size > 1:

                print(f"-- -- Processing node {n} out of {n_nodes} " +
                      f"in precluster with {pc_size} nodes      \r", end="")
                Xc = X[ind]

                if Xc[0].count_nonzero() > 1:

                    # Compute zero-radius similarity graph.
                    sg = SimGraph(Xc)

                    if len(ind) < 5000:
                        # This is to disable large logging messaging
                        logger = logging.getLogger()
                        logger.setLevel(logging.ERROR)
                        for handler in logger.handlers:
                            handler.setLevel(logging.ERROR)
                    else:
                        print(f"-- -- Processing node {n} out of {n_nodes} " +
                              f"in precluster with {pc_size} nodes      ")
                    sg.compute_id_graph(R=1e-100, verbose=False)
                    if len(ind) < 5000:
                        # Restore logging messages
                        logger.setLevel(logging.INFO)
                        for handler in logger.handlers:
                            handler.setLevel(logging.INFO)

                    # Get connected components
                    G = igraph.Graph(n=pc_size, edges=sg.edge_ids,
                                     directed=False)
                    cc = G.clusters()

                    # Get the membership vector
                    cluster_labels = cc.membership

                    # Assign labels to new clusters:
                    cluster_ids[ind] = (n_clusters +
                                        np.array([cluster_labels]).T)
                    n_clusters += len(set(cluster_labels))
                    n_edges += len(sg.edge_ids)

                else:
                    # If topics have only one nonzero element, all of them must
                    # be equal. Thus, there is no need to call simgraph
                    cluster_ids[ind] = n_clusters
                    n_clusters += 1
                    n_edges += pc_size * (pc_size - 1) // 2

            else:
                cluster_ids[ind] = n_clusters
                n_clusters += 1

        # Convert np array of cluster ids into list
        self.cluster_ids = cluster_ids.T.tolist()[0]
        self.n_edges = n_edges
        self.n_clusters = n_clusters
        self.n_preclusters = len(precluster_sizes)

        if reduceX:
            self.computeXeq()

        return

    def computeXeq(self):
        """
        Computes the reduced feature matrix X, with a single row per each
        equivalent class.

        self.X[i] contains the feature vector of all nodes from equivalence
        class i.
        """

        ind = [0] * self.n_clusters
        for n, c in enumerate(self.cluster_ids):
            ind[c] = n

        self.Xeq = self.X[ind]

        return

    def he_affinity(self, X, R=2, g=1, rescale=True):
        """
        Compute all Hellinger's affinities between all nodes in the graph based
        on the node attribute vectors in matrix

        It assumes that all attribute vectors are normalized to sum up to 1
        Attribute matrix X can be sparse

        Parameters
        ----------
        X : numpy array
            Input matrix of probabilistic attribute vectors
        R0 : float
            Radius (maximum He distance. Edges at higher distance are removed)
        g : float
            Exponent for the final affinity mapping
        rescale : boolean
            If True, affinity values are rescaled so that the minimum value is
            zero and the maximum values is one.

        Returns
        -------
        edge_id : list of tuples
            List of edges
        weights : list
            List of edge weights
        """

        # ################################
        # Compute affinities for all edges

        # I take the square root here. This is inefficient if X has many
        # rows and just af few edges will be computed. However, we can
        # expect the opposite (the list of edges involves the most of the
        #  nodes).
        Z = np.sqrt(X)

        # Divergences are compute by blocks. This is much faster than a
        # row-by-row computation, specially when X is sparse.
        d2_he = []
        for i in range(0, len(self.edge_ids), self.blocksize):
            edge_ids = self.edge_ids[i: i + self.blocksize]

            # Take the (matrix) of origin and destination attribute vectors
            i0, i1 = zip(*edge_ids)

            if issparse(X):
                P = Z[list(i0)].toarray()
                Q = Z[list(i1)].toarray()
            else:
                P = Z[list(i0)]
                Q = Z[list(i1)]

            # Squared Hellinger's distance
            # The maximum is used here just to avoid 2-2s<0 due to
            # precision errors
            s = np.sum(P * Q, axis=1)
            d2_he += list(np.maximum(2 - 2 * s, 0))

        # #########
        # Filtering

        # Filter out edges with He distance above R.
        ed = [z for z in zip(self.edge_ids, d2_he) if z[1] < R**2]
        if len(ed) > 0:
            edge_id, d2 = zip(*ed)
        else:
            edge_id, d2 = [], []

        # ####################
        # Computing affinities

        # Transform distances into affinity values.
        weights = self._d2_to_sim(d2, 'He', g, rescale, R)

        return edge_id, weights

    def l1_affinity(self, X, R=2, g=1, rescale=True):
        """
        Compute all l1's affinities between all nodes in the graph based on the
        node attribute vectors
        It assumes that all attribute vectors are normalized to sum up to 1
        Attribute matrix X can be sparse

        Parameters
        ----------
        X : numpy array
            Input matrix of probabilistic attribute vectors
        R0 : float
            Radius (maximum L1 distance. Edges at higher distance are removed)
        g : float
            Exponent for the final affinity mapping
        rescale : boolean
            If True, affinity values are rescaled so that the minimum value is
            zero and the maximum values is one.

        Returns
        -------
        edge_id : list of tuples
            List of edges
        weights : list
            List of edge weights
        """

        # ################################
        # Compute affinities for all edges

        # I take the square root here. This is inefficient if X has many
        # rows and just af few edges will be computed. However, we can
        # expect the opposite (the list of edges involves the most of the
        #  nodes).

        # Divergences are compute by blocks. This is much faster than a
        # row-by-row computation, specially when X is sparse.
        d_l1 = []
        for i in range(0, len(self.edge_ids), self.blocksize):
            edge_ids = self.edge_ids[i: i + self.blocksize]

            # Take the (matrix) of origin and destination attribute vectors
            i0, i1 = zip(*edge_ids)
            if issparse(X):
                P = X[list(i0)].toarray()
                Q = X[list(i1)].toarray()
            else:
                P = X[list(i0)]
                Q = X[list(i1)]

            # l1 distance
            d_l1 += list(np.sum(np.abs(P - Q), axis=1))

        # #########
        # Filtering

        # Filter out edges with L1 distance above R.
        ed = [z for z in zip(self.edge_ids, d_l1) if z[1] < R]
        if len(ed) > 0:
            edge_id, d_l1 = zip(*ed)
        else:
            edge_id, d_l1 = [], []

        # ####################
        # Computing affinities

        # Transform distances into affinity values. Note that we do not use
        # self._d2_to_sim method becasue distances in d are not squared.
        Rmax = 2
        if rescale:
            # Here we need to reassign R to avoid R >> 2 (which is the max
            # l1 value). Otherwise, normalization z/R**2 could be harmful.
            Rmax = min(R, Rmax)
        # Rescale affinities so that the minimum value is zero.
        weights = [(1 - z**g / (Rmax**g + EPS)) for z in d_l1 if z < Rmax]

        return edge_id, weights

    def l2_affinity(self, X, R=2, th_gauss=0.1):
        """
        Compute all l2's affinities between all nodes in the graph based on the
        node attribute vectors
        It assumes that all attribute vectors are normalized to sum up to 1
        Attribute matrix X can be sparse

        Parameters
        ----------
        X : numpy array
            Input matrix of probabilistic attribute vectors
        R0 : float
            Radius (maximum L2 distance. Edges at higher distance are removed)
        th_gauss : float
            Similarity threshold All similarity values below this threshold are
            set to zero.

        Returns
        -------
        edge_id : list of tuples
            List of edges
        weights : list
            List of edge weights
        """

        # ################################
        # Compute affinities for all edges

        # I take the square root here. This is inefficient if X has many
        # rows and just af few edges will be computed. However, we can
        # expect the opposite (the list of edges involves the most of the
        #  nodes).

        # Divergences are compute by blocks. This is much faster than a
        # row-by-row computation, specially when X is sparse.
        d_l2 = []
        for i in range(0, len(self.edge_ids), self.blocksize):
            edge_ids = self.edge_ids[i: i + self.blocksize]

            # Take the (matrix) of origin and destination attribute vectors
            i0, i1 = zip(*edge_ids)
            if issparse(X):
                P = X[list(i0)].toarray()
                Q = X[list(i1)].toarray()
            else:
                P = X[list(i0)]
                Q = X[list(i1)]

            # l1 distance
            d_l2 += list(np.sum((P - Q)**2, axis=1))

        # #########
        # Filtering

        # Filter out edges with JS distance above R (divergence above R**2).
        edge_id = [z[0] for z in zip(self.edge_ids, d_l2) if z[1] < R**2]

        # ####################
        # Computing affinities

        # The value of gamma to get min edge weight th_gauss at distance R
        gamma = - np.log(th_gauss) / (R**2 + EPS)
        # Nonzero affinity values
        weights = [np.exp(-gamma * z) for z in d_l2 if z < R**2]

        return edge_id, weights

    def JS_affinity(self, X, Y=None, R=1, g=1, rescale=True):
        """
        Compute all truncated Jensen-Shannon affinities between all pairs of
        connected nodes in the graph based on the node attribute vectors

        It assumes that all attribute vectors are normalized to sum up to 1
        Attribute matrices in X and Y can be sparse

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
        """

        # ################
        # Set right matrix
        if Y is None:
            Y = X

        # ################################
        # Compute affinities for all edges

        # Divergences are compute by blocks. This is much faster than a
        # row-by-row computation, specially when X is sparse.
        divJS = []
        for i in range(0, len(self.edge_ids), self.blocksize):
            edge_ids = self.edge_ids[i: i + self.blocksize]
            i0, i1 = zip(*edge_ids)

            if issparse(X):
                P = X[list(i0)].toarray() + EPS
                Q = Y[list(i1)].toarray() + EPS
            else:
                P = X[list(i0)] + EPS
                Q = Y[list(i1)] + EPS

            M = 0.5 * (P + Q)
            divJS += list(0.5 * (np.sum(P * np.log2(P / M), axis=1) +
                                 np.sum(Q * np.log2(Q / M), axis=1)))

        # #########
        # Filtering

        # Filter out edges with JS distance not below R
        # (divergence above R**2).
        ed = [z for z in zip(self.edge_ids, divJS) if z[1] < R**2]
        if len(ed) > 0:
            edge_id, d2 = zip(*ed)
        else:
            edge_id, d2 = [], []

        # ####################
        # Computing affinities

        # Transform distances into affinity values.
        weights = self._d2_to_sim(d2, 'JS', g, rescale, R)

        return edge_id, weights

    def show_JS_bounds(self, R=None, sim=None, g=1, out_path=None,
                       verbose=True):
        """
        Computes JS bounds for a given similarity measure.

        Parameters
        ----------
        R : float
            Radius. Edges link all data pairs at distance lower than R
            This is to forze a sparse graph.
        sim : string
            Similarity measure used to compute affinity matrix
            Available options are:

            'l1->JS', same as JS, but the graph is computed after preselecting
            edges using l1 distances and a theoretical bound

            'He->JS', same as JS, but the graph is computed after preselecting
            edges using Hellinger's distances and a theoretical bound

            'He2->JS', same as He-Js, but using the self implementation of He.

        g : float
            Exponent for the affinity mapping
        verbose : boolean
            (Only for he_neighbors_graph()). If False, block-by-block
            messaging is omitted

        Returns
        -------
        self : object
            Modifies self.edge_ids (ist of edges, as pairs (i, j) of indices)
            and self.weights (list of affinity values for each pair in
            edge_ids)
        """

        # #########################
        # Computing Distance Matrix

        # This is just to abbreviate
        X = self.X

        # Select Distance measure for radius_neighbor_graph
        if sim == 'l1->JS':
            d = 'l1'     # Note: l1 seems equivalent to manhattan
            label = 'l1'
        elif sim == 'He->JS':
            d = 'l2'     # (He is an L2 distance over the square root of X)
            label = 'Hellinger'
        elif sim == 'He2->JS':
            # No distance metric is selected, because a proper implementation
            # is used instead sklearn
            label = 'Hellinger'
        else:
            logging.error(f"-- JS bounds not available for similarity = {sim}")
            exit()

        # Select secondary radius
        if sim == 'l1->JS':
            R0 = np.sqrt(8 * np.log(2)) * R
            # Refined R0. Not relevant effect for small R0
            R0 = (12 / np.sqrt(2) * np.sqrt(np.sqrt(1 + R0**2 / 36) - 1))
        elif sim in ['He->JS', 'He2->JS']:
            R0 = np.sqrt(2) * R
        logging.info(f'-- -- {label}-radius bound for JS: {R0}')

        # Compute the connectivity graph of all pair of nodes at distance
        # below R0
        # IMPORTANT: Note that, despite radius_neighbors_graph has an option
        # 'distance' that returns the distance values, it cannot be used in
        # any case because the distance matrix does not distinghish between
        # nodes at distance > R0 and nodes at distance = 0
        t0 = time()
        logging.info(f'-- -- Computing neighbors_graph ...')
        if sim == 'He2->JS':
            self.edge_ids = self.he_neighbors_graph(
                X, R0, mode='connectivity', verbose=verbose)
        elif sim == 'He->JS':
            # We must compute the connectivity graph because module
            # radius_neighbors_graph looses edges between nodes at zero
            # distance
            D = radius_neighbors_graph(np.sqrt(X), radius=R0,
                                       mode='connectivity', metric=d)
        elif sim == 'l1->JS':
            D = radius_neighbors_graph(X, radius=R0, mode='connectivity',
                                       metric=d)
        logging.info(f'       in {time()-t0} seconds')

        # ##############################################
        # From distance matrix to list of weighted edges
        if sim != 'He2->JS':
            # Compute lists with origin, destination and value for all edges in
            # the graph affinity matrix.
            orig_id, dest_id = D.nonzero()

            # Since the graph is undirected, we select ordered pairs orig_id,
            # dest_id only
            self.edge_ids = list(filter(lambda i: i[0] < i[1],
                                        zip(orig_id, dest_id)))

        # ####################
        # Computing Affinities

        logging.info(f"-- -- Computing affinities for {len(self.edge_ids)}" +
                     " edges ...",)
        t0 = time()

        # For methods ->JS, the distance computed by the neighors_graph
        # method is not the target distance JS.
        # A new self.edge_ids is returned because the function filters out
        # affinity values below th.
        self.edge_ids, self.weights = self.JS_affinity(X, R, g)
        n_edges = len(self.edge_ids)

        # A temporary plot to visualize the differences between l1 or He and JS
        if sim in ['l1->JS', 'He->JS']:
            if sim == 'l1->JS':
                # Compute the distance graph
                D = radius_neighbors_graph(X, radius=R0, mode='distance',
                                           metric=d)
            elif sim == 'He->JS':
                # Compute the distance graph
                D = radius_neighbors_graph(np.sqrt(X), radius=R0,
                                           mode='distance', metric=d)
            fpath = os.path.join(out_path, f'simtest_{sim}.png')
            self.simTest(D, R, sim, g=g, fpath=fpath, label=label)

        logging.info(f"      reduced to {n_edges} edges")
        logging.info(f'      Computed in {time()-t0} seconds')

        logging.info(f"-- -- Graph generated with {self.n_nodes} nodes and " +
                     f"{n_edges} edges")

        return

    def simTest(self, D, R, sim=None, g=1, fpath=None, label='l1'):
        """
        Plot the values in weights vs the values in D selected by the indices
        in edge_id.

        This is used to visualize the efect of selecting samples using one
        measure (l1, l2) as a first step to reduce the sample set used to
        elect edges based on the Jensen-Shannon divergence.

        Parameters
        ----------
        D : 2D array
            Data matrix
        R : float
            Radius bound
        sim : string
            Name of the similarity function
        g : float
            Exponent value
        fpath : string
            Output path
        label : string
            Label for the figure plot
        """

        # Get values of the distance measure used as a reference
        div = [D[x] for x in self.edge_ids]

        plt.figure()
        # Plot the sampled points
        plt.plot(div, self.weights, '.', label='Data samples')

        if sim in ['He->JS', 'He2->JS']:
            bound = R * np.sqrt(2)
        elif sim == 'l1->JS':
            r = R * np.sqrt((8 * np.log(2)))
            bound = (12 / np.sqrt(2) * np.sqrt(np.sqrt(1 + r**2 / 36) - 1))

        if bound is not None:
            # Plot the line stating the distance theshold applied to select the
            # sampled points
            plt.plot([bound, bound], [0, 1], ':', label='Bound')

            # Plot the line stating the distance theshold applied to select the
            # sampled points
            if sim is not None:
                aff = np.linspace(0, 1, 100)
                if sim in ['He->JS', 'He2->JS']:
                    r = R * np.sqrt(2) * np.sqrt(1 - aff**(1 / g))
                elif sim == 'l1->JS':
                    r = R * np.sqrt((8 * np.log(2)) * (1 - aff**(1 / g)))
                    r = (12 / np.sqrt(2) * np.sqrt(np.sqrt(1 + r**2 / 36) - 1))
                plt.plot(r, aff, label='Bound function')
        plt.xlabel(label + ' distance')
        plt.ylabel('JS-based similarity')
        plt.legend()
        plt.show(block=False)

        if fpath is not None:
            plt.savefig(fpath)

        return

