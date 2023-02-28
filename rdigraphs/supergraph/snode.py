#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Python libraries
import numpy as np
import pandas as pd
import logging
import pprint
import yaml
import pathlib
import copy

from time import time

from scipy.sparse import csr_matrix, identity, save_npz, load_npz, issparse
# This is not being used, because it is quite slow
# from scipy.stats import entropy

# For community detection algorithms
import networkx as nx
import networkx.algorithms as nxalg

import matplotlib
matplotlib.use("Agg")
import seaborn as sns

# Graph layout.
# fa2 is not available for python>3.8 as of Dec. 2022.
# To allow using the rest of the code for python>3.8, we catch the error
# to keep running.
try:
    from fa2 import ForceAtlas2   # "pip install fa2"
except Exception:
    print("WARNING: fa2 could not be imported."
          "Force-atlas layout not available")

# Local imports
from rdigraphs.sim_graph.sim_graph import SimGraph
from rdigraphs.community_plus.community_plus import CommunityPlus

EPS = np.finfo(float).tiny


class DataGraph(object):

    """
    Generic class defining a graph of data
    """

    def __init__(self, label="dg", path=None, load_data=True, edge_class=None):
        """
        Defines the datagraph structure

        If a datagraph exists in the given path, it is loaded. Otherwise,
        a new empty graph is created

        Parameters
        ----------
        label : str
            Name os the supernode
        path : str
            Path to the folder that contains, or will contain, the graph data
        load_data : bool
            If True (default) the graph data are loaded.
            If False, only the graph metadata are loaded
        edge_class : str or None, optional (default=None)
            Type of edges: 'directed' or 'undirected'. If None, 'undirected'
            is assumed
        """

        # #################################
        # Complete list of class attributes
        self.label = label     # Name of the datagraph

        # Paths...
        if path is not None:
            self.path2graph = pathlib.Path(path)   # to the graph folder
        else:
            self.path2graph = path
        self.path2nodes = None   # to nodes
        self.path2edges = None   # to edges
        self.path2mdata = None   # to metadata
        self.path2T = None       # to feature matrix

        # Nodes:
        self.df_nodes = pd.DataFrame()  # Dataframe of node attributes
        self.nodes = []                 # List of nodes
        self.n_nodes = 0                # Number of nodes
        # List of indices of a subset of selected nodes. This is useful for
        # processing very large graphs with a subsample of nodes
        self.T = None         # Feature matrix, one row per node
        self.Teq = None       # Equivalent T (will merge equal rows in T)
        self.save_T = False   # By default, feature matrix will not be saved
        self.B = None      # Binary sparse feature matrix, one row per node
        # The following is the name of the column in self.df_nodes that will
        # contain the node identifiers. It is actually a constant, because it
        # is not modified inside the class (and there is no clear reason to
        # modify it outside).
        # It is taken as 'Id' because it is the corresponding column name in
        # csv files of nodes for visualization in Gephi.
        self.REF = 'Id'

        # Edges:
        self.df_edges = pd.DataFrame(
            columns=['Source', 'Target', 'Weight'])   # Dataframe of edges
        # List of edges, as pairs (i, j) of indices.
        self.edge_ids = []
        self.weights = []    # List of affinity values per pair in edge_ids
        self.n_edges = 0
        self.edge_class = edge_class

        # Communities:
        self.nc = {}                 # Number of clusters
        self.cluster_labels = None   #
        self.cluster_sizes = None    # Size of each community
        self.cluster_centers = None  # Centroids for each label value
        self.CD = CommunityPlus()
        self.CD_report = None        # Metadata of the last call to a Community
        #                              Detection algorithm

        # Simgraph
        # WARNING: This is actually a fake SimGraph object. SimGraph objects
        # are used by some methods in this class, but not as class attributed.
        # A SimGraph attribute is declared here just to forze a link in the UML
        # diagram generated with pyreverse.
        self.sg = SimGraph(np.array([[]]))
        self.sg = None   # <-- Just to make clear that this object is not used

        # ########################
        # Read data, if available

        if path:
            # paths to nodes, edges, metadata and feature matrix
            path = pathlib.Path(path)
            self.path2nodes = path / (label + '_nodes.csv')
            self.path2edges = path / (label + '_edges.csv')
            self.path2mdata = path / (label + '_mdata.yml')
            self.path2T = path / 'feature_matrix.npz'
            # This is for backward compatibility. The name has been changed
            # because the feature matrix might be non-sparse.
            self.path2T_old = path / 'modelo_sparse.npz'

        # Read metadata
        self.load_metadata()
        if self.edge_class is None:
            try:
                # Read edge_class from metadata, if available
                self.edge_class = self.metadata['graph']['edge_class']
            except Exception:
                # Use default
                self.edge_class = 'undirected'

        # Read graph data
        if load_data:
            # Load the graph data (nodes and edges)
            self.load_graph()

        # The following is to deal with issues raising in different
        # non-standard situations: graph data are not loaded, no metadata file
        # exists or inconsistencies between graph data and metadata.
        if load_data:
            if self.metadata == {}:
                # Default graph metadata
                if edge_class is None:
                    edge_class = 'undirected'
                self.metadata = {'graph': {'category': 'graph',
                                           'subcategory': None,
                                           'edge_class': edge_class}}
            else:
                # Report data inconsistencies, because they are clues to bugs
                logging.info(
                    f'-- -- Graph {label} loaded with {self.n_nodes} nodes '
                    f'and {self.n_edges} edges')
                if self.n_nodes != self.metadata['nodes']['n_nodes']:
                    logging.warning("-- -- Inconsistent number of nodes in "
                                    "data and metadata")
                if self.n_edges != self.metadata['edges']['n_edges']:
                    logging.warning("-- -- Inconsistent number of edges in "
                                    "data and metadata")
            # Update metadata, fixing inconsistencies, if any
            self.update_metadata()

        else:
            if self.metadata == {}:
                logging.info(
                    '-- -- Neither metadata nor graph data have been loaded')
            else:
                self.n_nodes = self.metadata['nodes']['n_nodes']
                self.n_edges = self.metadata['edges']['n_edges']
                logging.info(f'-- -- Metadata from graph {self.label} loaded.')

        return

    def load_metadata(self):
        """
        Loads metadata file
        """

        # Load metadata
        if self.path2mdata and self.path2mdata.is_file():
            with open(self.path2mdata, 'r') as f:
                self.metadata = yaml.safe_load(f)
        else:
            self.metadata = {}

        return

    def has_saved_features(self):

        if (self.path2T is not None) and self.path2T.is_file():
            return True
        # This elif is for backward compatibility with old names of the
        # feature matrix
        elif (self.path2T is not None) and self.path2T_old.is_file():
            return True
        else:
            return False

    def update_metadata(self):
        """
        Updates metadata dictionary with the self variables directly computed
        from df_nodes and df_edges
        """

        if 'nodes' not in self.metadata:
            self.metadata['nodes'] = {}
        if 'edges' not in self.metadata:
            self.metadata['edges'] = {}

        self.metadata['nodes'].update({
            'attributes': list(set(self.df_nodes.columns) - {self.REF}),
            'n_nodes': self.n_nodes})

        self.metadata['edges'].update({
            'attributes': list(set(self.df_edges.columns)
                               - {'Source', 'Target', 'Weight'}),
            'n_edges': self.n_edges})

        return

    def load_graph(self):
        '''
        Load a base graph from files.

        This version assumes csv files with the format used by Gephi software.

        Parameters
        ----------
        path2nodes : str
            Path to a csv file with the nodes
        path2edges : str
            Path to a csv file with the edges.
        '''

        if self.path2nodes.is_file():

            logging.info(f'-- -- Loading GRAPH from {self.path2graph}')

            # ##########
            # Load nodes
            self.df_nodes = pd.read_csv(self.path2nodes)
            # The following is required because if the node ids are integers
            # they are loaded by pandas as int, even though they were saved
            # as str.
            if any(not isinstance(x, str) for x in self.df_nodes[self.REF]):
                self.df_nodes[self.REF] = self.df_nodes[self.REF].astype(str)

            self._df_nodes_2_atts()

            if self.n_nodes != len(set(self.nodes)):
                logging.warn(
                    "-- -- There exist multiple nodes with the same name.")

            # ##########
            # Load edges
            if self.path2edges.is_file():
                self.df_edges = pd.read_csv(self.path2edges)

                # The following is required because if the source node ids are
                # integers, they are loaded by pandas as int, no matter if
                # they are saved as str
                if any(not isinstance(x, str)
                       for x in self.df_edges['Source']):
                    self.df_edges['Source'] = self.df_edges['Source'].astype(
                        str)
                # The same for target node ids
                if any(not isinstance(x, str)
                       for x in self.df_edges['Target']):
                    self.df_edges['Target'] = self.df_edges['Target'].astype(
                        str)

                # Copy edge info in other
                self._df_edges_2_atts()

            if self.path2T.is_file():
                # Since we do not know if the feature matrix is sparse or not,
                # we try first with the sparse case.
                # Note that we could use np.load for both the sparse and non
                # sparse cases, but I prefer to used load_npz because I am
                # not sure if some parameters would be needed for the np.load()
                # in the sparse case.
                try:
                    self.T = load_npz(self.path2T)
                except ValueError:
                    self.T = np.load(self.path2T)['T']

                # This is just a flag indicating that this graph has a saved
                # feature file
                self.save_T = True

            # This is for backward compatibility with old name of feature file
            elif self.path2T_old.is_file():
                try:
                    self.T = load_npz(self.path2T_old)
                except ValueError:
                    self.T = np.load(self.path2T_old)['T']

                # This is just a flag indicating that this graph has a saved
                # feature file
                self.save_T = True

        return

    def _df_edges_2_atts(self):
        """
        Copies edge information from self.df_edges into several attributes:
        self.edge_ids, self.weights and self.n_edges.
        These variables are all redundant with self.df_edges, but they are
        created for computational and coding convenience.
        """

        # Replace node names by their respective indices
        # This is too slow for many nodes and edges
        # sour_ids = [self.nodes.index(x)
        #             for x in self.df_edges['Source']]
        # targ_ids = [self.nodes.index(x)
        #             for x in self.df_edges['Target']]
        # This is much better
        if len(self.df_edges) > 0:
            z = dict(zip(self.nodes, range(self.n_nodes)))
            source_ids = [z[x] for x in self.df_edges['Source']]
            target_ids = [z[x] for x in self.df_edges['Target']]

            # Compute edge attributes
            self.edge_ids = list(zip(source_ids, target_ids))
            self.weights = self.df_edges['Weight'].tolist()
            self.n_edges = len(self.edge_ids)

        else:

            # Compute edge attributes
            self.edge_ids = []
            self.weights = []
            self.n_edges = 0

        return

    def _df_nodes_2_atts(self):
        """
        Copies node information from self.df_nodes into several object
        attributes: self.nodes and self.n_nodes.
        These variables are redundant with self.df_nodes, but they are created
        for computational and coding convenience.
        """
        self.nodes = self.df_nodes[self.REF].tolist()
        self.n_nodes = len(self.nodes)

        return

    def _computeK(self, diag=True):
        """
        Given the list of affinity values in self.weights, computes a sparse
        symmetric affinity matrix.

        This method has bee translated to CommunityPlus class. It is not being
        used by snode at the time of writing this comment, but we keep it here
        just in case...

        Parameters
        ----------
        diag : bool
            If True, a unit diagonal component is added

        Returns
        -------
        K : sparse csr matrix (n_nodes, n_nodes)
            Affinity matrix
        """

        if len(self.edge_ids) > 0:
            K = csr_matrix((self.weights, zip(*self.edge_ids)),
                           shape=(self.n_nodes, self.n_nodes))
        else:
            # If there are no edges, create an empty csr matrix
            K = csr_matrix(([], ([], [])), shape=(self.n_nodes, self.n_nodes))

        # hasattr is used here for backward compatibility.
        if not hasattr(self, 'edge_class') or self.edge_class == 'undirected':
            # Make the affinity matrix symmetric.
            if diag:
                # Make the affinity matrix symmetric and with a unit diagonal.
                K = K + K.T + identity(self.n_nodes)
            else:
                # Make it symmetric
                K = K + K.T
        elif self.edge_class == 'directed':
            # Make the affinity matrix symmetric.
            if diag:
                # Make the affinity matrix symmetric and with a unit diagonal.
                K = K + identity(self.n_nodes)
            else:
                pass
        else:
            logging.error("-- -- Unknown edge type")

        return K

    def _computeD(self):
        """
        Given the list of affinity values in self.weights, computes a sparse
        symmetric "distance" matrix, were distances are computed as one minus
        affinities.

        Returns
        -------
        D : sparse csr matrix (n_nodes, n_nodes)
            Affinity matrix
        """

        # EPS is used here to avoid zero distances.
        # "For weighted graphs the edge weights must be greater than zero."
        # (NetworkX documentation, betweenness centrality)
        # Maybe I should use even higher values
        if len(self.edge_ids) > 0:
            dists = [1 - x + EPS for x in self.weights]
            D = csr_matrix((dists, zip(*self.edge_ids)),
                           shape=(self.n_nodes, self.n_nodes))
        else:
            # If there are no edges, create an empty csr matrix
            D = csr_matrix(([], ([], [])), shape=(self.n_nodes, self.n_nodes))

        # hastattr is used here for backward compatibility.
        if not hasattr(self, 'edge_class') or self.edge_class == 'undirected':
            # Make the affinity matrix symmetric.
            D = D + D.T
        elif self.edge_class == 'directed':
            pass
        else:
            logging.error("-- -- Unknown edge type")

        return D

    def pprint(self, n=10):
        """
        Pretty prints basic information about the current graph

        Parameters
        ----------
        n : Maximum number of nodes and edges to visualize
        """

        print("\n-- Graph attributes:")
        atts = self.get_attributes()
        print(f"-- -- {self.label}: {', '.join(atts)}")

        # Show graph (snode) dimensions
        md = self.metadata   # Just to abbreviate
        print("\n-- Graph dimensions:")
        print(f"-- -- Number of nodes: {md['nodes']['n_nodes']}")
        print(f"-- -- Number of edges: {md['edges']['n_edges']}")

        # Preview nodes and edges
        print("\n-- Preview of nodes:")
        print(self.df_nodes.head(n))
        print("\n-- Preview of edges:")
        print(self.df_edges.head(n))

        # Zero version. This is expected to be improved...
        print("\n-- Graph metadata:")
        pprint.pprint(self.metadata)

        return

    # #####################
    # Graph edition methods
    # #####################
    def add_single_node(self, node, attributes={}):
        '''
        Adds a single node to datagraph.

        Making a difference with self.set_nodes, add_single_nodes assumes that
        the current datagraph may be not empty.

        Parameters
        ----------
        node : str
            Name of the node to add.
        attributes dict, optional (default = {}):
            Dictionary of node attributes
        '''

        # Check if node already exists
        if node in self.nodes:
            logging.warning(
                f'-- -- DUPLICATION: Node {node} already exists in graph '
                f'{self.label}. Existing node and their edges will be '
                f'removed')
            self.drop_single_node(node)

        # Create graph with the input node
        row = {self.REF: node, **attributes}
        if self.n_nodes == 0:
            # The input node is the first node in the graph
            self.df_nodes = pd.DataFrame([row])
        else:
            self.df_nodes2 = self.df_nodes.append(row, ignore_index=True)
            # Using pandas.concat() to add a row
            new_row = pd.DataFrame(row, index=[0])
            self.df_nodes = pd.concat([self.df_nodes, new_row]).reset_index(
                drop=True)

        self.n_nodes += 1
        self.nodes.append(node)
        self.update_metadata()

        return

    def add_single_edge(self, source, target, weight=1, attributes={}):
        '''
        Adds a single edge to the graph.

        Parameters
        ----------
        source : str
            Name of the source node
        target : str
            Name of the target node
        weight : float, optional (default=1)
            Weight of the edge
        attributes: dict, optional (default={})
            Dictionary of node attributes
        '''

        # Check if node already exist
        if source not in self.nodes:
            logging.warning(
                f'-- -- NODE UNKNOWN: Source node {source} does not exist '
                f'in graph {self.label}. No action taken')
        elif target not in self.nodes:
            logging.warning(
                f'-- -- NODE UNKNOWN: Target node {target} does not exist '
                f'in graph {self.label}. No action taken')
        else:
            # Create graph with the input node
            row = {'Source': source, 'Target': target, 'Weight': weight,
                   **attributes}
            if self.n_edges == 0:
                # The input node is the first node in the graph
                self.df_edges = pd.DataFrame([row])
            else:
                self.df_edges = self.df_edges.append(row, ignore_index=True)

            self._df_edges_2_atts()

            self.metadata['graph'].update({'category': 'graph',
                                           'subcategory': 'imported_edges'})
            self.update_metadata()

        return

    def drop_single_node(self, node):
        '''
        Resets a single node from graph.

        Parameters
        ----------
        node : str
            Name of the node to remove.
        '''

        # Check if node already exists
        if node not in self.nodes:
            logging.warning(f'-- -- Node {node} does not exist in graph'
                            f'{self.label}. No action taken')

        else:
            # Create graph with the input node
            self.df_nodes = self.df_nodes[self.df_nodes[self.REF] != node]
            self.n_nodes -= 1
            self.nodes.remove(node)

            # Remove edges with the given node.
            if self.n_edges > 0:
                self.df_edges = self.df_edges[self.df_edges.Source != node]
                self.df_edges = self.df_edges[self.df_edges.Target != node]
                # Recompute self attributes about edges.
                self._df_edges_2_atts()

            # Update new graph size in metadata
            self.update_metadata()

        return

    def disconnect_nodes(self, source, target, directed=False):
        '''
        Drops all edges between nodes source and target.

        Parameters
        ----------
        source : str
            Id of the source node
        target : str
            Id of the target node
        directed : bool, optional (default=False)
            If True, only edges from source to target are removed.
        '''

        # Check if node already exist
        if source not in self.nodes:
            logging.warning(
                f'-- -- NODE UNKNOWN: Source node {source} does not exist '
                f'in graph {self.label}. No action taken')
        elif target not in self.nodes:
            logging.warning(
                f'-- -- NODE UNKNOWN: Target node {target} does not exist '
                f'in graph {self.label}. No action taken')
        else:
            # Remove source->target edges
            self.df_edges = self.df_edges[(self.df_edges.Source != source)
                                          | (self.df_edges.Target != target)]

            if not directed:
                # Remove target->source edges
                self.df_edges = self.df_edges[
                    (self.df_edges.Source != target)
                    | (self.df_edges.Target != source)]

            # Recompute self attributes about edges.
            self._df_edges_2_atts()

            # Update no. of edges in metadata
            self.update_metadata()

        return

    def set_nodes(self, nodes=None, T=None, save_T=False):
        '''
        Add nodes to datagraph.

        This version assumes that the current graph is empty.

        Parameters
        ----------
        nodes : list or None, optional (default=None)
            A list of node identifiers
            If None, integer identifiers will be assigned
        T : array (n_nodes, n_features) or None, optional (default=None)
            A matrix of feature vectors: one row per node, one column per
            feature.
        save_T : bool, optional (default=False)
            If True, the feature matrix T is saved into an npz file.
        '''

        # This is just a marker to indicate the graph saver to save the
        # feature matrix too
        self.save_T = save_T

        # Graph is loaded with the input nodes without edges
        if self.n_nodes > 0:
            logging.warning(f'-- -- Graph {self.label} is not empty. '
                            'The current graph content will be removed.')

        # Default list of nodes
        if nodes is None:
            if T is None:
                # Default is the empty list
                nodes = []
            else:
                # Consecutive integers will be used as node identifiers.
                nodes = list(range(T.shape[0]))

        # Save nodes in the object variables
        self.n_nodes = len(nodes)
        logging.info(
            f'-- -- Uploading {self.n_nodes} nodes to graph {self.label}.')
        self.nodes = nodes
        self.df_nodes = pd.DataFrame(nodes, columns=[self.REF])

        if T is not None:
            # Check consistency between nodes and features
            if T.shape[0] == self.n_nodes:
                self.T = T
            else:
                logging.error(
                    "-- -- ERROR: The number of nodes in the list of nodes "
                    f"({self.n_nodes}) must be equal to the number of rows "
                    f"in T {T.shape[0]}.")
                exit()

        # Reset edges:
        self.df_edges = pd.DataFrame()   # Dataframe of edges
        self.edge_ids = []
        self.weights = []    # List of affinity values per pair in edge_ids
        self.n_edges = 0

        # Re-build metadata dictionary
        self.update_metadata()

        return

    def add_new_nodes(self, nodes):
        '''
        Add new nodes to datagraph.

        This version assumes that the self graph has no feature matrix in
        self.T.

        If a feature matrix exist, an error is returned to avoid inconsistency
        between the number of nodes in the new graph and the number of rows
        in self.T.

        Nodes in the input list that already exist in the self graph are
        ignored

        Parameters
        ----------
        nodes : list
            A list of node identifiers
        '''

        # Check if feature matrix exist
        if self.T is not None:
            logging.error(
                "-- -- No more nodes can be added if a feature matrix exist. "
                "You should remove self.T beforehand to add new nodes")
            exit()

        # Remove from variable nodes the nodes that already exist in the graph
        new_nodes = list(set(nodes) - set(self.nodes))
        if len(new_nodes) < len(nodes):
            logging.warning("-- -- Some nodes in the input list already "
                            "exist in the graph.")

        # Update dataframe of nodes
        df_new_nodes = pd.DataFrame({self.REF: new_nodes})
        self.df_nodes = pd.concat([self.df_nodes, df_new_nodes], sort=False,
                                  ignore_index=True)
        # Update redundant object variables
        self._df_nodes_2_atts()

        # Update metadata
        self.update_metadata()

        return

    def add_feature_matrix(self, T, save_T=None):
        '''
        Add feature matrix to datagraph.

        Parameters
        ----------
        T : array (n_nodes, n_features) or None, optional (default=None)
            A matrix of feature vectors: one row per node, one column per
            feature.
        save_T : bool, optional (default=False)
            If True, the feature matrix T is saved into an  npz file.
        '''

        # This is just a marker to indicate the graph saver to save the
        # feature matrix too
        if save_T is not None:
            self.save_T = save_T

        # Check consistency between nodes and features
        if T.shape[0] == self.n_nodes:
            self.T = T
        else:
            logging.error("-- -- The number of nodes must be equal to the "
                          "number of rows in the feature matrix")

        return

    def set_edges(self, source_nodes, target_nodes, weights=None):
        '''
        Add edges to datagraph.

        This version assumes that the current graph has no edges.

        Parameters
        ----------
        source_nodes : list
            A list of the source nodes. All of them must exist in self.nodes
        target_nodes : list
            A list of the respective target nodes
        weights : list or None, optional (default=None)
            The list of weights. If None, unit weights will be assumed
        '''

        n_edges = len(source_nodes)

        # Default weights are all unity.
        if weights is None:
            weights = [1] * n_edges

        # Edge datagrames
        self.df_edges = pd.DataFrame(data={'Source': source_nodes,
                                           'Target': target_nodes,
                                           'Weight': weights})
        # Redundant attributes
        self._df_edges_2_atts()

        # Update metadata dictionary
        self.update_metadata()

        return

    def add_attributes(self, names, values, fill_value=None):
        """
        Add attributes to the snode

        This is a preliminary version.
        Future versions should evolve in different ways: Check if attribute
        exists and act accordingly, or allow partial value assignments

        Parameters
        ----------
        names : str or list
            Attribute name
            If values is a pandas dataframe, names contains the name of the
            column in values that will be used as key for merging
        values : list or pandas dataframe or dict
            If list: it contains the attribute values. If names is list, it
            is a list of lists. The order of the values must correspond with
            the order of nodes in the self node.
            If pandas dataframe, dataframe containing the new attribute values.
            This dataframe will be merged with the dataframe of nodes. In
            such case, name
            If dict, the keys must refer to values in the reference column of
            the snode.
        fill_value : None, scalar or dict, optional (default=None)
            Specifies what to do with NaN values in the ouput dataframe
            If None, there are not changed
            If dict, fillna containg the value used to replace each column.
            If scalar, all NaN's are replaced by the given scalar value
        """

        if isinstance(values, pd.DataFrame) or isinstance(values, dict):

            if isinstance(values, pd.DataFrame):
                df_atts = values

                # Use the same name for the reference column to avoid column
                # duplication during merge
                df_atts.rename(columns={names: self.REF}, inplace=True)
            else:
                if isinstance(names, str):
                    names = [names]
                df_atts = pd.DataFrame(
                    values.items(), columns=[self.REF] + names)

            # Merge values into the self nodes dataframe.
            # Note that, if the right_on column (i.e. names) is not equal to
            # self.REF, the output dataframe will contain columns self.REF and
            # names that could be redundant. We preserve both of them just in
            # case a duplicate column with the original column names is
            # convenient
            try:
                self.df_nodes = self.df_nodes.merge(
                    df_atts, how='left', on=self.REF)
            except Exception:
                self.df_nodes = self.df_nodes.merge(
                    df_atts.astype({self.REF: 'str'}), how='left', on=self.REF)

        else:
            if isinstance(names, str):
                names = [names]
                values = [values]

            for i, name in enumerate(names):
                self.df_nodes[name] = values[i]

        if fill_value is not None:
            self.df_nodes.fillna(fill_value)

        # Update list of current attributes in the metadata dictionary
        self.update_metadata()

        return

    def remove_attributes(self, names):
        """
        Add attributes to the snode

        This is a preliminary version.
        Future versions should evolve in different ways: Check if attribute
        exists and act accordingly, or allow partial value assignments

        Parameters
        ----------
        names : str or list
            Attribute name or names
        """

        if isinstance(names, str):
            names = [names]

        for name in names:
            if name in self.df_nodes.columns:
                self.df_nodes.drop([name], axis=1, inplace=True)

        # Update list of current attributes in the metadata dictionary
        self.update_metadata()

        return

    # ############################
    # Graph information extraction
    # ############################
    def get_attributes(self):
        """
        Returns the list of node attributes of the self snode.
        """

        # return list(set(self.df_nodes.columns) - set([self.REF]))
        return self.metadata['nodes']['attributes']

    def get_nodes_by_value(self, att, value):
        """
        Return a list of all nodes from the self graph taking a given value
        on a given attribute

        Parameters
        ----------
        att : str
            Name of the attribute
        value : str or int or list or None
            Allowed value of the attribute.
            If None, select columns with null values (pandas nan's).
            If str or int, select collumns taking the given value
            If list, select nodes taking any value in the list
        """

        if value is None:
            df = self.df_nodes[self.df_nodes[att].isnull()]
        elif isinstance(value, list):
            df = self.df_nodes[self.df_nodes[att].isin(value)]
        else:
            df = self.df_nodes[self.df_nodes[att] == value]

        if len(df) > 0:
            nodes = df[self.REF].tolist()
        else:
            nodes = []

        return nodes

    def get_nodes_by_novalue(self, att, value):
        """
        Return a list of all nodes from the self graph NOT taking a given value
        on a given attribute

        Parameters
        ----------
        att : str
            Name of the attribute
        value : str or int or None
            Value of the attribute. If None, select columns with no null values
            of the attribute (i.e. no pandas nan's)
        """

        if value is None:
            df = self.df_nodes[~self.df_nodes[att].isnull()]
        else:
            df = self.df_nodes[self.df_nodes[att] != value]

        if len(df) > 0:
            nodes = df[self.REF].tolist()
        else:
            nodes = []

        return nodes

    def get_nodes_by_threshold(self, att, th, bound='lower'):
        """
        Return a list of all nodes from the self graph whose value of some
        attribute is above or below a given threshold.

        Parameters
        ----------
        att : str
            Name of the attribute
        th : int or float
            Threshold value.
        bound : str {'lower', 'upper'}, optional (default='lower')
            States if the threshold is a lower (default) or an upper bound.
        """

        if bound == 'lower':
            df = self.df_nodes[self.df_nodes[att] >= th]
        elif bound == 'upper':
            df = self.df_nodes[self.df_nodes[att] <= th]
        else:
            logging.error("-- -- Unknown value of input argument bound")

        if len(df) > 0:
            nodes = df[self.REF].tolist()
        else:
            nodes = []

        return nodes

    def get_matrix(self):
        """
        Returns the sparse_csr weight matrix for the self graph
        """

        data = self.weights
        row, col = zip(*self.edge_ids)

        W = csr_matrix((data, (row, col)),
                       shape=(self.n_nodes, self.n_nodes))

        return W

    # ################
    # Graph processing
    # ################
    def filter_edges(self, th):
        """
        Removes all edges with a similarity value below th

        Parameters
        ----------
        th : float
            Threshold
        """

        # Remove edges
        self.df_edges = self.df_edges[self.df_edges['Weight'] >= th]

        # Update redundant snode attributes
        self._df_edges_2_atts()

        return

    def filter_nodes_by_threshold(self, att, th, bound='lower'):
        """
        Removes all nodes whose value of a given attribute is below or above a
        given threshold.

        Parameters
        ----------
        att : str
            Name of the attribute
        th : int or float
            Threshold value.
        bound : str {'lower', 'upper'}, optional (default='lower')
            States if the threshold is a lower (default) or an upper bound.
            If "lower", all nodes with attribute less than the bound are
            removed
        """

        # Select edges
        nodes = self.get_nodes_by_threshold(att, th, bound=bound)
        subgraph = self.sub_graph(nodes, sampleT=True)

        self.df_nodes = subgraph['nodes']
        self.df_edges = subgraph['edges']
        self.T = subgraph['T']
        # The equivalent feature matrix, if it exist, is no longer valid.
        self.Teq = None

        # Update redundant snode and sedge attributes
        self._df_nodes_2_atts()
        self._df_edges_2_atts()
        self.update_metadata()

        return

    def filter_nodes_by_value(self, att, value):
        """
        Removes all nodes whose value of a given attribute is not in a given
        list of allowed values.

        Parameters
        ----------
        att : str
            Name of the attribute
        value : str or int or list or None
            Allowed value of the attribute.
            If None, select columns with null values (pandas nan's).
            If str or int, select collumns taking the given value
            If list, select nodes taking any value in the list
        """

        # Select edges
        nodes = self.get_nodes_by_value(att, value)
        subgraph = self.sub_graph(nodes, sampleT=True)

        self.df_nodes = subgraph['nodes']
        self.df_edges = subgraph['edges']
        self.T = subgraph['T']
        # The equivalent feature matrix, if it exist, is no longer valid.
        self.Teq = None

        # Update redundant snode and sedge attributes
        self._df_nodes_2_atts()
        self._df_edges_2_atts()
        self.update_metadata()

        return

    def computeSimGraph(self, s_min=None, n_gnodes=None, n_edges=None,
                        similarity='JS', g=1, blocksize=25_000, useGPU=False,
                        tmp_folder=None, save_every=1e300, verbose=True):
        """
        Computes a sparse similarity graph for the self graph structure.
        The self graph must contain a T-matrix, self.T

        Parameters
        ----------
        s_min : float or None, optional (default=None)
            Similarity threshold. Edges link all data pairs with similarity
            higher than R. This forzes a sparse graph.
        n_gnodes : int or None, optional (default=None)
            Number of nodes in the subgraph.
            If None, all nodes are used
            If n_gnodes is less than the no. of rows in self.T, a random
            subsample is taken.
        n_edges : int or None, optional (default=None)
            Target number of edges. n_edges is an alternative to radius.
            Only one of both must be specified (i.e., not None)
        similarity : str {'JS', 'l1', 'He', 'He2', 'Gauss', 'l1->JS', \
                     'He->JS', 'He2->JS'}, optional (default='JS')
            Similarity measure used to compute affinity matrix
            Available options are:
            'JS' (1 minus Jensen-Shannon (JS) divergence (too slow);
            'l1' (1 minus l1 distance);
            'He' (1 minus squared Hellinger's distance (sklearn-based
            implementation));
            'He2' (1 minus squared Hellinger distance (self implementation));
            'Gauss' (an exponential function of the squared l2 distance);
            'l1->JS' (same as JS, but the graph is computed after
            preselecting edges using l1 distances and a theoretical bound;
            'He->JS' (same as JS, but the graph is computed after
            preselecting edges using He distances and a theoretical bound
            'He2->JS' (same as He-Js, but using the self implementation of He)
        g : float, optional (default=1)
            Exponent for the affinity mapping
        blocksize : int, optional (default=25_000)
            Size of each block for the computation of affinity values. Large
            sizes might imply a large memory consumption.
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
        verbose : bool, optional (default=True)
            (Only for he_neighbors_graph()). If False, block-by-block
            messaging is omitted

        Returns
        -------
        self : object
            The following attributes are created or modified:
            self.edge_ids (list of edges, as pairs (i, j) of indices);
            self.weights (list of affinity values for each pair in edge_ids);
            self.df_edges (Pandas dataframe with one row per edge and columns
            'Source', 'Target' and 'Weihgt'. The weight is equal to the
            (mapped) affinity value)
        """

        # Time control
        t0 = time()

        # Total number of items in the graph. Note that this can be higher
        # than n_nodes, which is the number of items in the subgraph
        if n_gnodes is None:
            n_gnodes = self.n_nodes

        # Select random sample from the whole graph
        # i2n is the list of indices of the selected nodes
        # The name is a nemononic n = i2n[i] converts an index of the
        # list of selected nodes to an index of the complete list of nodes
        if self.n_nodes <= n_gnodes:
            i2n = range(self.n_nodes)
            n_gnodes = self.n_nodes
        else:
            np.random.seed(3)    # For comparative purposes (provisional)
            i2n = np.random.choice(range(self.n_nodes), n_gnodes,
                                   replace=False)

        # Take the topic matrix
        Tg = self.T[i2n]

        # ##########################
        # Computing Similarity graph
        sg = SimGraph(Tg, blocksize=blocksize, useGPU=useGPU,
                      tmp_folder=tmp_folder, save_every=save_every)
        if not verbose:
            # Disable log messages
            root_logger = logging.getLogger()
            root_logger.disabled = True

        sg.sim_graph(s_min=s_min, n_edges=n_edges, sim=similarity, g=g,
                     verbose=verbose)

        if not verbose:
            # Re-enable log messages
            root_logger = logging.getLogger()
            root_logger.disabled = False
        # NEW: indices in self.edge_ids refer to locations in self.nodes
        # OLD VERSION: self.edge_ids = sg.edge_ids
        self.edge_ids = [(i2n[i], i2n[j]) for i, j in sg.edge_ids]
        self.weights = sg.weights
        self.n_edges = len(self.edge_ids)

        # Create pandas structures for edges.
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
            # float to avoid yaml errors with np.float
            's_min': float(sg.s_min),
            'g': g,
            'time': tm,
            'n_sampled_nodes': n_gnodes,
            'sampling_factor': n_gnodes / self.n_nodes,
            'neighbors_per_sampled_node': self.n_edges / n_gnodes,
            'density': 2 * self.n_edges / (self.n_nodes * (self.n_nodes - 1))})

        logging.info(f"-- -- Graph generated with {n_gnodes} nodes and "
                     f"{self.n_edges} edges in {tm} seconds")

        return

    def sub_graph(self, ynodes, sampleT=True):
        """
        Returns a subgraph from the given graph by selecting a subset of nodes
        and all the edges between them.

        The graph may be specified by a target number of nodes or by a list of
        nodes. If a list, nodes in the list that are not nodes in the self
        graph will be ignored

        Parameters
        ----------
        ynodes : int or list
            If list: List of nodes of the output subgraph.
            If int: Number of nodes to sample. The list of nodes is taken
            at random without replacement from the graph nodes
        sampleT : bool, optional (default=True)
            If True, a sampled T is computed and returned

        Returns
        -------
        subgraph : dict
            A dictionary with up to 3 entries:
            'edges' (dataframe of edges);
            'nodes' (dataframe of node attributes);
            'idx' (list of indices of selected nodes (only for int ynodes))
            'T' (feature submatrix of the selected nodes (only for int ynodes
                 and sampleT=True))
        """

        subgraph = {}

        if isinstance(ynodes, int):
            n = ynodes
            if n > self.n_nodes:
                # Take all nodes from the self graph, in the original order
                idx = list(range(self.n_nodes))
            else:
                # Take n nodes selected at random
                np.random.seed(3)
                idx = sorted(np.random.choice(range(self.n_nodes), n,
                                              replace=False))
            # Select nodes
            subgraph['nodes'] = self.df_nodes.iloc[idx]
            ynodes = [self.nodes[i] for i in idx]

        else:
            subgraph['nodes'] = self.df_nodes.loc[
                self.df_nodes[self.REF].isin(ynodes)]
            idx = subgraph['nodes'].index.tolist()

        # Keep indices of the selected nodes (they will be used to sample T)
        subgraph['idx'] = idx
        # Reset integer indices.
        subgraph['nodes'].reset_index(drop=True, inplace=True)
        # Not that this list may differ from the one in ynodes in size and
        # order

        # Subsample feature matrix if it exists
        if sampleT and self.T is not None:
            subgraph['T'] = self.T[idx]
        else:
            subgraph['T'] = None

        # Discard edges with a source node out from the input list
        if len(self.df_edges) > 0:
            edges = self.df_edges[self.df_edges['Source'].isin(ynodes)]
            # Discard edges with a target node out from the input list
            subgraph['edges'] = edges[edges['Target'].isin(ynodes)]
        else:
            subgraph['edges'] = self.df_edges

        return subgraph

    def remove_isolated(self):
        """
        Removes isolated nodes from the self graph.
        """

        # ############
        # Source snode
        n0 = self.n_nodes

        # ###############
        # Filtering nodes

        sources = [e[0] for e in self.edge_ids]
        targets = [e[1] for e in self.edge_ids]
        connected_nodes = sorted(list(set(sources + targets)))

        # Select only those source nodes that exist in s0.
        self.nodes = [self.nodes[n] for n in connected_nodes]

        # Update node and edge parameters
        self.n_nodes = len(self.nodes)
        self.df_nodes = self.df_nodes.iloc[connected_nodes]
        # The positions of the nodes in the node list have changed,
        # so self.edge_ides must be updated.
        self._df_edges_2_atts()
        self.update_metadata()

        logging.info(f'-- -- {n0-self.n_nodes} isolated nodes removed from '
                     f'graph {self.label}')

        return

    def sort_nodes(self):
        """
        Sorts nodes in the nodes dataframe.

        This method is useful to align different graphs that may have the same
        nodes but in different orders. Using the same order is necessary to
        make consistent operations between graph matrices.
        """

        # Sort nodes in dataframe
        self.df_nodes.sort_values(self.REF, axis=0, inplace=True)
        # Copy order to the list of nodes
        self.nodes = self.df_nodes[self.REF].tolist()
        # Update list of edge indices
        self._df_edges_2_atts()

        return

    # ##############
    # Graph analysis
    # ##############
    def compute_ppr(self, th=0.9, inplace=False):

        # ##################
        # Select data matrix

        # Get distance matrix
        # Compute a sparse distance matrix from a list of affinity values
        D = self._computeD()

        # ############################################
        # Local graph analysis algorithms

        # Apply the clustering algorithm
        if (not hasattr(self, 'edge_class')
                or self.edge_class == 'undirected'):
            G = nx.from_scipy_sparse_matrix(D)
        else:
            G = nx.from_scipy_sparse_matrix(D, create_using=nx.DiGraph())

        # See also pagerank_numpy and pagerank_scipy
        p_max = 0.99
        p_base = (1.0 - p_max) / (self.n_nodes - 1)
        p = {i: p_base for i in range(self.n_nodes)}

        new_out = []
        for i in range(self.n_nodes):
            print(f'-- -- Node {i+1} out of {self.n_nodes}')
            pers = copy.copy(p)
            pers[i] = p_max
            C = nxalg.pagerank(
                G, alpha=0.85, personalization=pers, max_iter=100,
                tol=1e-06, nstart=None, weight='weight', dangling=None)

            new_out += [((i, j), C[i]) for j in range(self.n_nodes)
                        if C[i] > th]

        new_edges, new_weights = zip(*new_out)

        # Create pandas structures for edges.
        # Each node is a project, indexed by field REFERENCIA.
        orig_REF = [self.nodes[m] for m, n in new_edges]
        dest_REF = [self.nodes[n] for m, n in new_edges]
        df_edges = pd.DataFrame(
            list(zip(orig_REF, dest_REF, self.weights)),
            columns=['Source', 'Target', 'Weight'])
        df_edges['Type'] = 'Directed'

        if inplace:
            self.edge_ids = new_edges
            self.weights = new_weights
            self.df_edges = df_edges

        return new_edges, new_weights

    def compute_eq_nodes(self, name='eq_class'):
        """
        Computes a new node attribute indicating the equivalence classes. Each
        attribute value will identify all nodes with the same feature vector

        The self graph must containg a T-matrix, self.T

        Parameters
        ----------
        name : str, optional (default='eq_class')
            Name of the new node attribute

        Returns
        -------
        cluster_ids : list
        Teq : array
            A reduced feature matrix containing one row per equivalence class
        """

        # Time control
        t0 = time()

        # ###########################
        # Computing equivalence nodes

        sg = SimGraph(self.T)
        sg.cluster_equivalent_nodes(reduceX=True)

        # Add equivalence class as an attribute of the snode
        self.add_attributes(name, sg.cluster_ids)

        # ###############
        # Update metadata
        tm = time() - t0
        if 'local_features' not in self.metadata:
            self.metadata['local_features'] = {}
        self.lg_report = {'feature_name': name,
                          'No. of equivalence classes': sg.n_clusters,
                          'No. of embedded unit-weight edges': sg.n_edges}
        self.metadata['local_features'].update({name: self.lg_report})
        self.metadata['local_features'][name].update({'time': tm})

        logging.info(f"-- -- {sg.n_clusters} equivalence classes computed in "
                     f"{tm} seconds")

        return sg.cluster_ids, sg.Xeq

    def detectCommunities(self, alg='louvain', ncmax=None, label='Comm'):
        """
        Applies a Community Detection algorithm to the current self graph given
        by the edges in self.edges and the weights in self.weights.

        This is a simplified versio of self.computeClusters(), that is
        restricted to algorith based on affinity matrices, and not on feature
        vectors.

        Cluster labels are stored in attribute self.cluster_labels

        Parameters
        ----------
        alg : str {'louvain', fastgreedy', 'walktrap', 'infomap', \
             'labelprop', 'spectral'}, optional (default='louvain')
        ncmax : int or None, optional (default=None)
            Number of communities.
        label : str, optional (default='Comm')
             Label for the cluster indices in the output dataframe
        """

        # Start clock
        t0 = time()

        # Check if the label name is allowed
        if self.n_nodes > 0 and label in list(self.df_nodes):
            logging.warning(f"The label name {label} has been used. "
                            f"Existing data will be removed")

        # Parameters
        r = 1e100     # Resolution Only used by the Louvaing algorihm

        # ####################
        # Initializa CD report

        self.CD_report = {'algorithm': alg,
                          'ncmax': ncmax}
        # Apply the clustering algorithm
        if alg == 'louvain':
            # Louvain-specific parameters
            self.CD_report.update({'resolution': r})

        # ##############################
        # Community detection algorithmm
        self.cluster_labels, self.cluster_sizes = self.CD.detect_communities(
            self.edge_ids, self.weights, n_nodes=self.n_nodes, alg=alg,
            ncmax=ncmax, resolution=r)

        # ######################
        # Order clusters by size

        # To facilitate the visualization of the main clusters in gephi,
        # smallest indices are assigned to clusters with highest size

        # Update number of clusters
        self.nc[label] = len(self.cluster_sizes)
        self.CD_report.update({'n_communities': self.nc[label]})
        # int is required because self.cluster_sizes[0] is numpy.float(),
        # which causes problems to yalm.safe_dump()
        self.CD_report.update({'largest_comm': int(self.cluster_sizes[0])})

        # ########################
        # Store model in dataframe
        self.add_attributes(label, self.cluster_labels)

        # ###############
        # Update metadata
        tm = time() - t0
        if 'communities' not in self.metadata:
            self.metadata['communities'] = {}
        self.metadata['communities'].update({label: self.CD_report})
        self.metadata['communities'][label].update({'time': tm})

        # End message
        nc = self.metadata['communities'][label]['n_communities']
        logging.info(f'-- -- {nc} communities computed in {tm} seconds')

        return

    def compareCommunities(self, comm1, comm2, method='vi', remove_none=False):
        """
        Compare community structures comm1 and comm2 with a given method.

        Parameters
        ----------
        comm1 : str or list
            - if str: name of the community in the self snode
            - if list: membership list with the same size than the number
                       of nodes in the snode
        comm2 : str or list
            - if str: name of the community in the self snode
            - if list: membership list with the same size than the number
                       of nodes in the snode
        method : str {'vi', 'meila', 'nmi', 'danon', 'rand', 'adjusted_rand', \
                 'split-join', 'split-join-proj'}, optional (default='vi')
            Measure to use. Options are:
            'vi' | 'meila',  variation of information metric [Meila];
            'nmi' | 'danon', normalized mutual information [Danon];
            'rand', Rand index [Rand];
            'adjusted_rand', means the adjusted Rand index [Hubert];
            'split-join', split-join distance [van Dongen];
            'split-join-proj', assymetric split-join distance [van Dongen];
        remove_none : bool, optional (default=False)
            Whether to remove None entries from the membership lists.
        """

        if isinstance(comm1, str):
            clabels1 = self.df_nodes[comm1].tolist()
        else:
            clabels1 = comm1
        if isinstance(comm2, str):
            clabels2 = self.df_nodes[comm2].tolist()
        else:
            clabels2 = comm2

        # Call community comparator with the given metric
        d = self.CD.compare_communities(clabels1, clabels2, method=method,
                                        remove_none=False)

        # Print results
        logging.info(f'-- -- The value of {method.upper()} between '
                     f'communities is {d}')

        return

    def local_graph_analysis(self, parameter, label=None):
        """
        Computes local graph parameter for all nodes in the self snode.

        Parameters
        ----------
        parameter : str {'centrality', 'degree', 'betweenness', 'closeness', \
                    'centrality', 'pageRank', 'cluster_coef', 'katz',
                    'abs_in_degree', 'abs_out_degree'}
            Local parameter to compute
        label : str or None, optional (default=None)
            Name of the node attribute that will contain the local parameter
        """

        # Start
        t0 = time()       # Time control
        logging.info(f'-- Computing {parameter} for all nodes in the graph')
        if label is None:
            label = parameter  # Default label value

        # Check if the label name is allowed
        # if self.df_nodes is not None and label in list(self.df_nodes):
        #     logging.info(f"The label name {label} has been reserved to " +
        #                  "another field of the graph structure. Changed " +
        #                  f"to {0}_2")
        #    label = label + '_2'

        # ##################
        # Select data matrix

        # Get distance matrix
        # Compute a sparse distance matrix from a list of affinity values
        D = self._computeD()

        # ############################################
        # Local graph analysis algorithms

        # Apply the clustering algorithm
        if parameter in ['centrality', 'degree', 'betweenness', 'closeness',
                         'pageRank', 'cluster_coef', 'katz', 'abs_in_degree',
                         'abs_out_degree']:

            if (not hasattr(self, 'edge_class')
                    or self.edge_class == 'undirected'):
                G = nx.from_scipy_sparse_matrix(D)
            else:
                G = nx.from_scipy_sparse_matrix(D, create_using=nx.DiGraph())

            if parameter == 'centrality':
                # C = nxalg.eigenvector_centrality(
                #     G, max_iter=200, tol=1e-07, nstart=None, weight='weight')
                #    G, max_iter=100, tol=1e-06, nstart=None, weight='weight')
                print("Computing numpy centrality")
                C = nxalg.eigenvector_centrality_numpy(
                    G, weight='weight', max_iter=100, tol=0)

            elif parameter == 'degree':
                if self.edge_class == 'directed':
                    C = nxalg.out_degree_centrality(G)
                else:
                    C = nxalg.degree_centrality(G)

            elif parameter == 'abs_in_degree':
                C = dict(G.in_degree())

            elif parameter == 'abs_out_degree':
                C = dict(G.out_degree())

            elif parameter == 'betweenness':
                C = nxalg.betweenness_centrality(
                    G, k=3, normalized=True, weight='weight', endpoints=False,
                    seed=None)

            elif parameter == 'closeness':
                logging.warning('   Be patient. This may take some time...')
                C = nxalg.centrality.closeness_centrality(
                    G, u=None, distance=None, wf_improved=True)

            elif parameter == 'cluster_coef':
                logging.warning('   Be patient. This may take some time...')
                C = nxalg.clustering(G, nodes=None, weight='weight')

            elif parameter == 'pageRank':
                # See also pagerank_numpy and pagerank_scipy
                C = nxalg.pagerank(
                    G, alpha=0.85, personalization=None, max_iter=100,
                    tol=1e-06, nstart=None, weight='weight', dangling=None)

            elif parameter == 'katz':
                # This causes memory overflow for large graphs
                # alpha = 0.5 / max(nx.adjacency_spectrum(G))
                # This may fail if the largest eigenvector is < 0.1
                alpha = 0.1
                C = nxalg.katz_centrality(
                    G, alpha=alpha, beta=1.0, max_iter=1000, tol=1e-06,
                    nstart=None, normalized=True, weight='weight')

            # ########################
            # Store model in dataframe
            self.add_attributes(parameter, C.values())

            # ###############
            # Update metadata
            if 'local_features' not in self.metadata:
                self.metadata['local_features'] = {}
            self.lg_report = {'feature_name': parameter}
            self.metadata['local_features'].update({label: self.lg_report})
            self.metadata['local_features'][label].update(
                {'time': time() - t0})

            # End message
            tm = self.metadata['local_features'][label]['time']
            logging.info(f"-- {parameter} computed in {tm} seconds")

        else:

            logging.info(f"-- Parameter {parameter} not available. No action "
                         "taken")

        return

    def community_metric(self, cd_alg, parameter):
        """
        Computes global graph parameter for all nodes in the self snode.

        Parameters
        ----------
        cd_alg : str
            Name of the community detection algorithm
        parameter : str
            Name of the global parameter to compute

        Returns
        -------
        q :: float
            Value of the metric. This value is saved into the self.metadata
            attribute
        """

        # Start
        t0 = time()       # Time control
        logging.info(
            f'-- Computing {parameter} of {cd_alg} community in the graph')

        # ############################################
        # Local graph analysis algorithms

        clabels = self.df_nodes[cd_alg].values
        q = self.CD.community_metric(self.edge_ids, self.weights, clabels,
                                     parameter)

        # ###############
        # Update metadata

        # WARNING: float() is required to convert np.float64() into float.
        #          Otherwise, yaml will raise an error when metadata was saved
        self.metadata['communities'][cd_alg][parameter] = float(q)
        logging.info(f"-- {parameter} is {q}")

        # End message
        tm = time() - t0
        logging.info(f"-- {parameter} computed in {tm} seconds")

        return q

    def graph_layout(self, alg='fa2', color_att=None, gravity=1,
                     save_gexf=True):
        """
        Compute the layout of the given graph

        Parameters
        ----------
        alg : str {'fa2', 'fr'}, optional (default=fa2)
            Layout algorithm. Ootions are: 'fa2' (forze atlas 2), 'fr'
            (Fruchterman-Reingold).
        color_att : str or None, optional (default=None)
            Name of the attribute in self.df_nodes to use as color index
        gravity: int, optional (default=1)
            Gravity parameter (only for force atlas 2)
        save_gexf : bool, optional (default=True)
            If True, the graph layout is exported to a gexf file
            It False,  the node positions are stord as columns in self.df_nodes
            (note that positions are not saved in self.df_nodes if
            save_gexf=True)
        """

        # Start clock
        t0 = time()

        # ##################################
        # Transform graph to networkx format

        # Compute a sparse affinity matrix from a list of affinity values
        K = self._computeK(diag=False)
        # Convert matrix into graph
        G = nx.from_scipy_sparse_matrix(K)

        # ############
        # Graph layout

        # Compute positions using layout algorithm
        if alg == 'fa2':
            layout = ForceAtlas2(
                # Behavior alternatives
                outboundAttractionDistribution=False,  # Dissuade hubs
                linLogMode=False,  # NOT IMPLEMENTED
                adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                edgeWeightInfluence=1.0,
                # Performance
                jitterTolerance=1.0,  # Tolerance
                barnesHutOptimize=True,
                barnesHutTheta=1.2,
                multiThreaded=False,  # NOT IMPLEMENTED
                # Tuning
                scalingRatio=2.0,
                strongGravityMode=False,
                gravity=gravity,
                # Log
                verbose=True)
            positions = layout.forceatlas2_networkx_layout(
                G, pos=None, iterations=2000)
        elif alg == 'fr':
            positions = nx.drawing.layout.spring_layout(
                G, k=None, pos=None, fixed=None, iterations=5,
                threshold=0.0001, weight='weight', scale=1, center=None, dim=2,
                seed=None)

        # Get attributes to map to colors
        if color_att:
            node_attribs = self.df_nodes[color_att].tolist()
            uniq_attribs = list(set(node_attribs))  # List of unique attributes

            # Get a palette of rgb colors as large as the list of unique
            # attributes
            n_colors = len(uniq_attribs)
            palette = sns.color_palette(n_colors=n_colors)
            # Map palette of [0,1]-components to a paletter of 0-255 comps.
            palette = [[int(255.999 * x) for x in row] for row in palette]

            # Map attribute values to rgb colors
            attrib_2_idx = dict(zip(uniq_attribs, palette))

            # Map node attribute to rgb colors
            node_colors = [attrib_2_idx[a] for a in node_attribs]

        # Get list of degrees
        degrees = [val for (node, val) in G.degree()]

        # Insert graphical node attributes in graph structure
        for n in range(self.n_nodes):
            G.nodes[n]['viz'] = {'size': degrees[n],
                                 'position': {'x': positions[n][0],
                                              'y': positions[n][1],
                                              'z': 0}}
            if color_att:
                G.nodes[n]['viz']['color'] = {'r': node_colors[n][0],
                                              'g': node_colors[n][1],
                                              'b': node_colors[n][2],
                                              'a': 1}

        # ###############
        # Update metadata
        tm = time() - t0
        self.metadata['graph_layout'] = {'algorithm': 'Force Atlas 2',
                                         'time': tm}
        # ######
        # Saving

        if save_gexf:
            # Save graph in file
            if not self.path2graph.is_dir():
                self.path2graph.mkdir()

            path = self.path2graph / (self.label + '.gexf')
            nx.write_gexf(G, path)
        else:
            # Store positions in nodes dataframe
            # This might be unnecessary if the graph will be save in gexf
            x, y = list(zip(*positions.values()))
            self.add_attributes(['x', 'y'], [x, y])

        # End message
        logging.info(f'-- -- Graph layout computed in {tm} seconds')

        return

    # ##########
    # Save graph
    # ##########
    def saveGraph(self):
        """
        Save the datagraph in csv files
        """

        # Save nodes
        if self.n_nodes > 0:
            if len(self.df_nodes) == self.n_nodes:   # Cautionary test
                self.save_nodes()
            else:
                # This should never happen, but it is checked to avoid
                # catastrofic errors
                logging.error(
                    "-- -- Mismatch error: the number of nodes in metadata "
                    "is not equal to the length of the node list")
                exit()

            # Save edges
            if self.n_edges > 0:
                if len(self.df_edges) == self.n_edges:    # Cautionary test
                    self.df_edges.to_csv(self.path2edges, index=False,
                                         columns=self.df_edges.columns,
                                         sep=',', encoding='utf-8')
                else:
                    # This should never happen, but it is checked to avoid
                    # catastrofic errors
                    logging.error(
                        "-- -- Mismatch error: the number of edges in "
                        "metadata is not equal to the length of the edge list")
                    exit()

        if self.save_T:
            self.save_feature_matrix()

        self.save_metadata()

        return

    def save_feature_matrix(self):
        """
        Save feature matrix
        """

        # Save equivalent feature matrix
        if self.T is not None:
            if issparse(self.T):
                save_npz(self.path2T, self.T)
            else:
                np.savez(self.path2T, T=self.T)

        return

    def save_metadata(self):
        """
        Save metadata dictionary in yml file.
        """

        # Save metadata
        with open(self.path2mdata, 'w') as f:
            yaml.safe_dump(self.metadata, f, default_flow_style=False)

        return

    def save_nodes(self):

        if not self.path2graph.is_dir():
            self.path2graph.mkdir()

        # Save nodes
        self.df_nodes.to_csv(self.path2nodes, index=False,
                             columns=self.df_nodes.columns,
                             sep=',', encoding='utf-8')

        return

