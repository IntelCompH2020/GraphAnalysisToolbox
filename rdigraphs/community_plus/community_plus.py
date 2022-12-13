# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# # Python libraries
import numpy as np
import collections
import logging

from scipy.sparse import csr_matrix, identity
from sklearn.cluster import spectral_clustering

# # For community detection algorithms
import networkx as nx
import community      # "install python-louvain"

try:
    import leidenalg      # "install -c conda-forge leidenalg"
except:
    print("WARNING: leidenalg import error. Some methods will fail")

try:
    import igraph         # "install -c conda-forge python-igraph"
except:
    print("WARNING: igraph import error. Some methods will fail")


class CommunityPlus(object):

    """
    Generic class for the detection and analysis of communities from graphs
    """

    def __init__(self):
        """
        Defines variables of the community plus object
        """

        # #################################
        # Complete list of class attributes
        self.n_nodes = None      # Number of nodes
        self.resolution = None   # Resolution parameter of Louvain algorithm
        self.dendrogram = None   # A useful subproduct of some CD algorithms

        return

    def _computeK(self, edges, weights, diag=True):
        """
        Computes a sparse symmetric affinity matrix from edges and weights

        The size of the matrix will be taken from self.n_nodes.

        Parameters
        ----------
        edges : list of tuples of int
            List of edges, as pairs of indices (i, j)
        weights : list of float
            Weight of each edge
        diag : boolean
            If True, unit diagonal values are added to the matrix

        Returns
        -------
            K : sparse csr matrix
                Affinity matrix
        """

        if len(edges) > 0:
            K0 = csr_matrix((weights, zip(*edges)),
                            shape=(self.n_nodes, self.n_nodes))
        else:
            # If there are no edges, create an empty csr matrix
            K0 = csr_matrix(([], ([], [])), shape=(self.n_nodes, self.n_nodes))

        # Make the affinity matrix symmetric and with a unit diagonal.
        if diag:
            K = K0 + K0.T + identity(self.n_nodes)
        else:
            K = K0 + K0.T

        return K

    def _sortClusterIds(self, cluster_labels):
        """
        Given a np array of cluster labels, reassign cluster indices in such a
        way the clusters get sorted by decreasing cluster size

        If cluster_labels is not a numpy array, it is converted to.

        Parameters
        ----------
        cluster_labels : list or numpy array
            Array of cluster labels

        Returns
        -------
        new_cluster_labels : numpy array
            Array of cluster labels with modified indices
        cluster_sizes : numpy array
            Size of each cluster
        """

        l_min = min(cluster_labels)
        if l_min != 0:
            logging.warning(
                f'-- -- The minimum label value is {l_min}. Label values ' +
                f'will be shifted to get minimum label value equal to 0')
        cluster_labels = np.array(cluster_labels) - l_min

        # Sort cluster indices by size.
        nc = int(np.max(cluster_labels)) + 1
        cluster_sizes = np.zeros(nc)
        cluster_sizes_dict = collections.Counter(cluster_labels)
        for c, v in cluster_sizes_dict.items():
            cluster_sizes[c] = v

        new_indices = np.argsort(- cluster_sizes)

        # Compute inverse index
        N_to_posN = dict(zip(new_indices, range(nc)))

        # Reassign labels
        new_cluster_labels = np.array([N_to_posN[l] for l in cluster_labels])

        # Order clusters sizes
        cluster_sizes = cluster_sizes[new_indices]

        return new_cluster_labels, cluster_sizes

    def detect_communities(self, edges, weights, n_nodes=None, alg='louvain',
                           ncmax=None, resolution=1e100):
        """
        Applies a Community Detection algorithm to the graph given by a list
        of edges and a list of weights.

        Parameters
        ----------
        edges : list of tuples of int
            List of edges, as pairs of indices (i, j)
        weights : list of float
            Weight of each edge
        n_nodes : int
            Total number of nodes. In general, it must be specified because
            isolated nodes do not appear in the list of edges.
            If none, it is taken as the largest index in list or edges
        ncmax : int
            Number of clusters.
        alg : string, default='louvain
            Community detection algorithm.
            It must be one of 'louvain', 'spectral', 'fastgreedy', 'walktrap',
            'infomap' or 'labelprop'
        resolution: float
            Resolution parameter of the Louvain algorithm

        Returns
        -------
        cluster_labels : numpy array
            Index of the cluster assigned to each node
        cluster_sizes : numpy array
            Size of each cluster
        """

        # Start clock
        logging.info(f'-- -- Computing communities with the {alg} algorithm')

        if n_nodes is None:
            self.n_nodes = max([max(e) for e in edges]) + 1
        else:
            self.n_nodes = n_nodes

        # ############################################
        # Community detection or clustering algorithms

        # Reset dendrogram. This is not required, but it is used to reset
        # dendrograms from previous calls to this method.
        self.dendrogram = None

        # Apply the clustering algorithm
        if alg == 'spectral':
            # Get affinity matrix
            # Compute a sparse affinity matrix from a list of affinity values
            K = self._computeK(edges, weights)

            # Spectral clustering
            cluster_labels = spectral_clustering(
                K, n_clusters=ncmax, n_components=None, eigen_solver=None,
                n_init=10, eigen_tol=0.0, assign_labels='kmeans')

        elif alg == 'louvain':
            # Get affinity matrix
            # Compute a sparse affinity matrix from a list of affinity values
            K = self._computeK(edges, weights)

            G = nx.from_scipy_sparse_matrix(K)
            labels = community.best_partition(
                G, partition=None, weight='weight', resolution=resolution,
                randomize=None)
            cluster_labels = [labels[n] for n in range(K.shape[0])]
            # graphplot = nx.draw(G, K, node_size=40, width=0.5,)

        elif alg in ['cc', 'fastgreedy', 'walktrap', 'infomap', 'labelprop',
                     'leiden']:

            # These are the community detection algorithms based on igraph
            # library
            G = igraph.Graph(n=self.n_nodes, edges=edges, directed=False,
                             edge_attrs={'weight': weights})

            if alg == 'cc':
                # Get the connected components
                clusters = G.clusters()

            elif alg == 'fastgreedy':
                # Calculate dendrogram
                self.dendrogram = G.community_fastgreedy(weights='weight')
                # Convert it into a flat clustering
                clusters = self.dendrogram.as_clustering()

            elif alg == "walktrap":
                # Calculate dendrogram
                self.dendrogram = G.community_walktrap(
                    weights='weight', steps=4)
                # Convert it into a flat clustering
                clusters = self.dendrogram.as_clustering()

            elif alg == "infomap":
                # Calculate communities
                clusters = G.community_infomap(
                    edge_weights='weight', vertex_weights=None, trials=10)

            elif alg == "labelprop":
                # Calculate communities
                clusters = G.community_label_propagation(
                    weights='weight', initial=None, fixed=None)

            elif alg == "leiden":
                # This is the basic usage of leidenalg. For further options see
                # https://leidenalg.readthedocs.io/en/latest/reference.html#module-leidenalg
                clusters = leidenalg.find_partition(
                    G, leidenalg.ModularityVertexPartition, n_iterations=-1)

            # get the membership vector
            cluster_labels = clusters.membership

        else:
            exit("---- computeClusters: Unknown clustering algorithm")

        # ######################
        # Order clusters by size

        # To facilitate the visualization of the main clusters in gephi,
        # smallest indices are assigned to clusters with highest size

        # Order clusters by size
        cluster_labels, cluster_sizes = self._sortClusterIds(cluster_labels)

        # Update number of clusters
        n_com = len(cluster_sizes)
        logging.info(f"------ {alg} has found {n_com} communities")

        return cluster_labels, cluster_sizes

    def community_metric(self, edges, weights, clabels, parameter):
        """
        Computes a community metric with respect to a given graph

        Parameters
        ----------
        edges : list of tuples of int
            List of edges, as pairs of indices (i, j)
        weights : list of float
            Weight of each edge
        clabels : list
            List of community labels
        cd_alg : str
            Name of the community detection algorithm
        parameter : str
            Name of the global parameter to compute.
            Available options are: 'coverage', 'performance' and 'modularity'

        Returns
        -------
        q : float
            Value of the selecte metric
        """

        # ############################################
        # Local graph analysis algorithms

        if parameter == 'coverage':
            # Networkx implementation: too slow
            # partition = list(self.df_nodes[[cd_alg]].groupby(
            #     cd_alg).groups.values())
            # q = nxalg.community.quality.coverage(G, partition)
            sources, targets = zip(*edges)
            clabels = np.array(clabels)
            is_intra = (clabels[np.array(sources)] ==
                        clabels[np.array(targets)])

            # Compute coverage.
            sum_intra = np.sum(is_intra * np.array(weights))
            sum_extra = np.sum((1 - is_intra) * np.array(weights))
            q = sum_intra / (sum_intra + sum_extra)

        elif parameter == 'performance':
            # Networkx implementation: too slow
            # partition = list(self.df_nodes[[cd_alg]].groupby(
            #     cd_alg).groups.values())
            # q = nxalg.community.quality.coverage(G, partition)

            n_nodes = len(clabels)
            sources, targets = zip(*edges)
            clabels = np.array(clabels)
            is_intra = (clabels[np.array(sources)] ==
                        clabels[np.array(targets)])

            # Compute coverage.
            sum_intra = np.sum(is_intra)
            sum_extra = len(edges) - np.sum(is_intra)

            com_sizes = collections.Counter(clabels)

            # Total possible number of intra edges
            n_tot_intra = sum([x * (x - 1) / 2 for x in com_sizes.values()])
            # Total possible number of edges
            n_tot = n_nodes * (n_nodes - 1) / 2
            # Total possible number of extra edges
            n_tot_extra = n_tot - n_tot_intra
            # Number of possible extra edges that does not exist in graph
            n_non_edges_extra = n_tot_extra - sum_extra

            # Performance
            q = (sum_intra + n_non_edges_extra) / n_tot

        elif parameter == 'modularity':
            # see https://perso.crans.org/~aynaud/communities/api.html
            # This computes the modularity measure used by the lovain
            # algorithm

            # Build dictionary of pairs node: community
            # Note that we do not use the node identifiers in self.REF column,
            # but their indices in self.df_nodes.
            self.n_nodes = len(clabels)
            com = dict(zip(range(self.n_nodes), clabels))

            # Get graph
            K = self._computeK(edges, weights)
            G = nx.from_scipy_sparse_matrix(K)

            # Compute modularity
            q = community.modularity(com, G, weight='weight')

            # We can check also modularity from igraph:
            # https://igraph.org/python/doc/
            #     igraph-pysrc.html#Graph.modularity

        else:
            logging.error(f"---- Unknown global parameter: {parameter}.")

        return q

    def compare_communities(self, comm1, comm2, method='vi',
                            remove_none=False):
        """
        This is a wrapper to methods compare_communities and
        split_joing_distance from igraph.

        Compares two community structures using various distance measures.

        Parameters
        ----------
        comm1:  1st community structure as a membership list
        comm2:  2nd community structure as a membership list
        method: str
            Measure to use. Options are:

            'vi' | 'meila' :  variation of information metric [Meila]

            'nmi' | 'danon' : normalized mutual information [Danon]

            'rand' : Rand index [Rand],

            'adjusted_rand' : means the adjusted Rand index [Hubert]

            'split-join' : split-join distance [van Dongen]

            'split-join-proj' : assymetric split-join distance [van Dongen]

        remove_none : boolean
            Whether to remove None entries from the membership lists.

        Returns
        -------
            The calculated measure.

        References
        ----------
        Meila M: Comparing clusterings by the variation of information. In:
            Scholkopf B, Warmuth MK (eds). Learning Theory and Kernel Machines:
            16th Annual Conference on Computational Learning Theory and 7th
            Kernel Workship, COLT/Kernel 2003, Washington, DC, USA. LCNS, vol.
            2777, Springer, 2003. ISBN: 978-3-540-40720-1.
        Danon L, Diaz-Guilera A, Duch J, Arenas A: Comparing community
            structure identification. J Stat Mech P09008, 2005.
        van Dongen D: Performance criteria for graph clustering and Markov
            cluster experiments. Technical Report INS-R0012, National Research
            Institute for Mathematics and Computer Science in the Netherlands,
            Amsterdam, May 2000.
        Rand WM: Objective criteria for the evaluation of clustering methods.
            J Am Stat Assoc 66(336):846-850, 1971.
        Hubert L and Arabie P: Comparing partitions. Journal of Classification
            2:193-218, 1985.

        Notes
        -----
        'vi' and 'split-join' are divergence measures (minimum at 0).
        'nmi', 'rand' and 'adjusted_rand' are similarity measures (max at 1).

        """

        if method in ['vi', 'meila', 'nmi', 'danon', 'rand', 'adjusted_rand',
                      'split-join']:
            # Call the igraph method.
            d = igraph.compare_communities(comm1, comm2, method, remove_none)

        elif method == 'split-join-proj':
            # Compute the assymmetric split-join.
            # Note that the projection distance is asymmetric (it is not
            # actually a distance). This function returns the projection
            # distance of comm1 from comm2 and the projection distance of comm2
            # from comm1.
            # User option 'split-join' for a symmetric measure, which sums both
            # projections.
            d, d2 = igraph.split_join_distance(comm1, comm2, remove_none)

        return d

