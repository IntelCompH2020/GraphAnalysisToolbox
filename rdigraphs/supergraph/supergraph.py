#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Python libraries
import numpy as np
import pandas as pd
import scipy.sparse as scsp
import logging
import copy
import os
import shutil
import collections

# Local Imports
from rdigraphs.supergraph.snode import DataGraph
from rdigraphs.supergraph.sedge import SEdge

EPS = np.finfo(float).tiny


class SuperGraph(object):

    """
    Generic class defining a supergraph.

    A supergraph is a set of supernodes and super-edges connecting them.

    A supernode is itself a graph.

    A superedge connecting supernodes M and N is a directed bipartite graph
    whose nodes are the nodes of M and N, and edges are directed links from
    nodes in M to nodes in N.

    The class provides methods to create a supergraph and add nodes and edges
    using data taken from an SQL database.

    """

    def __init__(self, snode=None, path=None, path2snodes=None,
                 path2sedges=None, label="sg", keep_active=False):
        """
        Initializes an empty supergraph with an empty dictionary of supernodes
        and an empty dictionary of superlinks

        Parameters
        ----------
        snode : str or None, optional (default=None)
            If None, an empty supergraph is created.
            Otherwise, the given snode is added to the supergraph structure
        path : str or None, optional (default=None)
            Path to the supergraph folder
        path2snodes : str or None, optional (default=None)
            Path to the snodes folder. If None, snodes will be located in
            folder 'snodes' from the input path
        path2sedges : str or None, optional (default=None)
            Path to the sedges folder. If None, sedges will be located in
            folder 'sedges' from the input path
        label : str, optional (default='sg')
            Name of the supegraph
        keep_active : bool, optional (default=False)
            If True, all supergraph components loaded from memory or generated
            by some method are preserved in memory.
            If False, some methods may deactivate snodes or sedges to free some
            memory. This is useful for large graphs
        """

        # Name assigned to the supergraph
        self.label = label

        # Initialize dictionary of supernodes and superedges
        self.snodes = {}
        self.sedges = {}

        # Activation state variable
        self.keep_active = keep_active

        # Metagraph variables.
        # The metagraph is the graph describing the supergraph structure of
        # snodes and sedges
        self.path = path     # Location of the metagraph files.
        self.metagraph = DataGraph(label="metagraph", path=path)

        # Location of snodes:
        if path is not None and path2snodes is None:
            self.path2snodes = os.path.join(path, 'snodes')   # Default path
        else:
            self.path2snodes = path2snodes

        # Location of sedges
        if path is not None and path2sedges is None:
            self.path2sedges = os.path.join(path, 'sedges')   # Default path
        else:
            self.path2sedges = path2sedges

        # Add node if provided.
        if snode is not None:
            self.addSuperNode(snode)

        self.clean_up_metagraph()

        return

    # #################
    # Supergrap edition
    # #################
    def clean_up_metagraph(self):
        """
        Removes from the metagraph all nodes or edges that do no longer exist
        in the supergraph database.
        """

        for label in self.metagraph.nodes:
            snode_folder = os.path.join(self.path2snodes, label)
            if not os.path.isdir(snode_folder):
                logging.warning(f'---- Graph {label} does no longer exist. '
                                f' Removed from the supergraph structure')
                self.drop_snode(label)

        for s, t in self.metagraph.edge_ids:
            source = self.metagraph.nodes[s]
            target = self.metagraph.nodes[t]
            label = source + '_2_' + target

            sedge_folder = os.path.join(self.path2sedges, label)
            if not os.path.isdir(sedge_folder):
                logging.warning(f'---- Bigraph {label} does no longer exist.'
                                f' Removed from the supergraph structure')
                self.drop_sedge(label)

        return

    def addSuperNode(self, snode, attributes={}):
        """
        Add a snode to the supergraph structure.
        The node is an object (graph) from class DataGraph.

        Parameters
        ----------
        snode : object
            snode (an instance of class DataGraph)

        attributes : dict, optional (default={})
            Dictionary of snode attributes
        """

        label = snode.label
        self.snodes[label] = snode

        # Update metagraph
        self.metagraph.add_single_node(label, attributes)

        return

    def addSuperEdge(self, sedge, weight=1, attributes={}):
        """
        Add a superedge (sedge) to the supergraph structure.
        The sedge is an object (graph) from class SEdge, which is an extension
        of class DataGraph.

        Parameters
        ----------
            sedge : object
                Superedge (an object of class SEdge)
            weight : float, optional (default=1)
                Weight of the sedge (in the supergraph)
            attributes : dict, optional (default={})
                A dictionary of attributes
        """

        label = sedge.label_source + '_2_' + sedge.label_target

        if (sedge.label_source not in self.snodes
                or sedge.label_target not in self.snodes):
            logging.error('The source and target snodes of the new sedge '
                          'should be loaded in the supergraph')
        else:
            self.sedges[label] = sedge

        # Update metagraph
        attributes_and_label = dict(attributes, **{'label': label})
        self.metagraph.add_single_edge(
            sedge.label_source, sedge.label_target, weight=weight,
            attributes=attributes_and_label)

        return

    def makeSuperNode(self, label, out_path=None, nodes=None, T=None,
                      attributes={}, edge_class='undirected', save_T=False):
        """
        Make a new snode for the supergraph structure.
        The snode is created as an object (graph) from class DataGraph, with
        the input data in the args.

        Parameters
        ----------
        label : str
            Name os the supernode
        out_path : str or None, optional (default=None)
            Output path
        nodes : list or None, optional (default=None)
            List of nodes
        T : array or None, optional (default=None)
            Feature matrix, one row per node
        attributes : dict, optional (default={})
            Attributes of the supernode. Note that these are not attributes of
            the nodes, but of the whole supernode, that will be stored in the
            snode metagraph
        save_T : bool, optional (default=False)
            If True, the feature matrix T is saved into an  npz file.
         """

        if label in self.snodes or self.is_snode(label):
            logging.warning(
                f'-- -- An snode named {label} already exists in the '
                'supergraph.')
            logging.warning(
                '-- -- The old snode will be removed. Note that sedges '
                'starting or ending in the old snode could not be consistent '
                'with the new one')
            self.drop_snode(label)

        if out_path is None:
            # Default path: a folder with the name in label in the standard
            # location in the folder of snodes
            out_path = os.path.join(self.path2snodes, label)

        self.snodes[label] = DataGraph(label=label, path=out_path,
                                       edge_class=edge_class)
        self.snodes[label].set_nodes(nodes, T, save_T=save_T)

        # Update metagraph
        self.metagraph.add_single_node(label, attributes)

        return

    def drop_snode(self, label):
        """
        Removes snode from the supergraph. Note that this does not remove the
        related sedges, and the resulting supergraphg might be inconsistent.
        Future version of this method should accomplish edge removal.

        Parameters
        ----------
        label : str
            Name of the snode to be removed
        """

        # Deactivate snode
        self.deactivate_snode(label)

        # Drop snode from the metagraph
        self.metagraph.drop_single_node(label)

        # Delete snode container
        path = os.path.join(self.path2snodes, label)
        if os.path.isdir(path):
            shutil.rmtree(path)
        logging.info(f'-- -- -- snode {label} deleted')

        return

    def drop_sedge(self, label):
        """
        Removes sedges from the supergraph

        Parameters
        ----------
        label : str
            Name of the sedge to be removed
        """

        self.deactivate_sedge(label)

        # Drop edge from the metagraph
        link = label.split("_2_")

        self.metagraph.disconnect_nodes(link[0], link[1])

        # Delete sedge container
        path = os.path.join(self.path2sedges, label)
        if os.path.isdir(path):
            shutil.rmtree(path)
        logging.info(f'-- -- -- sedge {label} deleted')

        return

    def duplicate_snode(self, xlabel, ylabel, out_path=None):
        """
        Creates a copy of a given snode with another name.

        Parameters
        ----------
        xlabel : str
            Name of the snode to be duplicated
        ylabel : str
            Name of the new snode
        out_path : str or None, optional (default=None)
            Output path of the duplicate
        """

        if not self.is_active_snode(xlabel):
            self.activate_snode(xlabel)

        # Create new snode object
        self.makeSuperNode(ylabel, out_path=None)

        # Copy main snode attributes from the original snode
        self.snodes[ylabel].df_nodes = copy.copy(self.snodes[xlabel].df_nodes)
        self.snodes[ylabel].df_edges = copy.copy(self.snodes[xlabel].df_edges)
        self.snodes[ylabel].T = copy.copy(self.snodes[xlabel].T)
        self.snodes[ylabel].Teq = copy.copy(self.snodes[xlabel].Teq)
        self.snodes[ylabel].save_T = self.snodes[xlabel].save_T
        self.snodes[ylabel].metadata = copy.copy(self.snodes[xlabel].metadata)

        self.snodes[ylabel]._df_nodes_2_atts()
        self.snodes[ylabel]._df_edges_2_atts()
        self.snodes[ylabel].update_metadata()

        return

    # ############################
    # Read/change supergraph state
    # ############################
    def is_snode(self, label):
        """
        Checks if the snode given by label exists in the supergraph

        Parameters
        ----------
        label : str
            Name of the snode

        Returns
        -------
        b : boolean
            True if snode exists, False otherwise
        """
        return label in self.metagraph.nodes

    def is_sedge(self, e_label):
        """
        Checks if the sedge given by e_label exists in the supergraph

        Parameters
        ----------
        e_label : str
            Name of the sedge

        Returns
        -------
        b : boolean
            True if sedge exists, False otherwise
        """

        return ((self.metagraph.n_edges > 0)
                and (e_label in self.metagraph.df_edges['label'].tolist()))

    def is_active_snode(self, label):
        """
        Checks if the snode given by label is active

        Parameters
        ----------
        label : str
            Name of the snode

        Returns
        -------
        b : boolean
            True if snode is active, False otherwise
        """
        return label in self.snodes

    def is_active_sedge(self, e_label):
        """
        Checks if a given s_edge is active

        Parameters
        ----------
        e_label : str
            Name of the sedge

        Returns
        -------
        b : boolean
            True if sedge is active, False otherwise
        """
        return e_label in self.sedges

    def activate_snode(self, label):
        """
        Loads snode into the dictionary of active snodes

        Parameters
        ----------
        label : str
            Name of the snode
        """

        # Check if sedge named e_label do exists
        if self.is_snode(label):
            path = os.path.join(self.path2snodes, label)
            self.snodes[label] = DataGraph(label=label, path=path)

        else:
            logging.error(
                f"-- -- snode {label} does not exist in the supergraph ")

        return

    def activate_sedge(self, label):
        """
        Loads sedge into the dictionary of active sedges.

        Parameters
        ----------
        label : str
            Name of the sedge
        """

        if self.is_sedge(label):
            path = os.path.join(self.path2sedges, label)
            self.sedges[label] = SEdge(label=label, path=path)

        else:
            logging.error(
                f"-- -- sedge {label} does not exist in the supergraph ")

        return

    def activate_all(self):
        """
        Activates all snodes and sedges in the supergraph
        """

        for node in self.metagraph.nodes:
            if not self.is_active_snode(node):
                self.activate_snode(node)

        if 'label' in self.metagraph.df_edges:
            for edge in self.metagraph.df_edges['label'].tolist():
                if not self.is_active_sedge(edge):
                    self.activate_sedge(edge)

        return

    def deactivate_snode(self, label):
        """
        Remove snode from the dictionary of snodes. Note that this does not
        supresses the snode from the supergraph. It is only removed from
        memory.

        Parameters
        ----------
        label : str
            Name of the snode
        """

        if label in self.snodes:
            del self.snodes[label]

        return

    def deactivate_sedge(self, label):
        """
        Remove sedge from the dictionary of sedge. Note that this does not
        supresses the sedge from the supergraph. It is only removed from
        memory.

        Parameters
        ----------
        label : str
            Name of the sedge
        """

        if label in self.sedges:
            del self.sedges[label]

        return

    def deactivate(self):
        """
        Removes all snodes and sedges from the dictionaries of active snodes
        and sedges. This removes them from memory, but not from the supergraph
        structure. It is used to clean memory space
        """

        self.snodes = {}
        self.sedges = {}

        return

    # ######################
    # Supergraph information
    # ######################
    def get_terminals(self, e_label):
        """
        Returns the name of the source and target snodes of a given sedge

        Parameters
        ----------
        e_label : str
            Name of the sedge

        Returns
        -------
        s_label : str
            Name of the source snode
        t_label : str
            Name of the target snode
        """

        # Read sedge if not in memory
        in_memory = self.is_active_sedge(e_label)

        if in_memory:
            # Get names of the terminal snodes from sedge in memory
            s_label, t_label = self.sedges[e_label].get_terminals()
        else:
            # Get names from sedge in file.
            metadata = self.get_metadata(e_label, is_node_name=False)
            s_label = metadata['graph']['source']
            t_label = metadata['graph']['target']

        return s_label, t_label

    def get_metadata(self, label, is_node_name=True):
        """
        Returns the metadata of a given snode or sedge

        Parameters
        ----------
        label : str
            Name of the snode or sedge
        is_node : bool, optional (default=True)
            If True, label is a snode. If False, label is a sedge

        Returns
        -------
        md : dict
            Metadata dictionary.
        """

        # Check if sedge named e_label do exists
        if is_node_name:
            if self.is_active_snode(label):
                # Read metadata from in-memory snode
                md = self.snodes[label].metadata
            elif self.is_snode(label):
                # Read metadata from file.
                path = os.path.join(self.path2snodes, label)
                snode = DataGraph(label=label, path=path, load_data=False)
                md = snode.metadata
            else:
                logging.error(
                    f"-- -- snode {label} does not exist in the supergraph ")
                exit()

        else:
            if self.is_active_sedge(label):
                # Read metadata from in-memory snode
                md = self.sedges[label].metadata
            elif self.is_sedge(label):
                # Read metadata from file.
                path = os.path.join(self.path2sedges, label)
                sedge = SEdge(label=label, path=path, load_data=False)
                md = sedge.metadata
            else:
                logging.error(
                    f"-- -- sedge {label} does not exist in the supergraph ")
                exit()

        return md

    def get_attributes(self, label, is_snode_name=True):
        """
        Returns the attributes of a given snode or sedge

        Parameters
        ----------
        label : str
            Name of the snode or sedge
        is_node : bool, optional (default=True)
            If True, label is a snode. If False, label is a sedge

        Returns
        -------
        atts : list of str
            List of attributes of the given snode

        Notes
        -----
        If the snode is active, the attributes are not read from file, but from
        memory. Thus, if any other method has modified the attributes without
        updating in-memory data, the attribute list might be not updated.
        """

        if is_snode_name:

            if self.is_active_snode(label):
                # Get attribute names from memory
                atts = self.snodes[label].get_attributes()
            else:
                # Get attribute names from metadata file
                metadata = self.get_metadata(label)
                atts = metadata['nodes']['attributes']

        else:

            if self.is_active_sedge(label):
                # Get attribute names from memory
                atts = self.sedges[label].get_attributes()
            else:
                # Get attribute names from metadata file
                metadata = self.get_metadata(label)
                atts = metadata['nodes']['attributes']

        return atts

    # ###############
    # Snode inference
    # ###############
    def snode_from_atts(self, source, attrib, target=None, path_snode=None,
                        path_sedge=None, e_label=None, att_size=True):
        """
        Generate a new snode and a new sedge from a given snode in the
        supergraph and one of its attributes.

        The nodes of the new snode will consist of the attribute values of the
        snode.

        Each node in the source snode will be connected to the node in the
        target snode containing its attribute value.

        Parameters
        ----------
        source : str
            Name of the source snode in the supergraph
        attrib : str
            The attribute in snode containing the target nodes
        target : str or None, optional (default=None)
            Name of the target node
        path_snode : str or None, optional (default=None)
            Output path to save the target snode
        path_sedge : str or None, optional (default=None)
            Output path to save the sedge
        e_label : str or None, optional (default=None)
            Name of the new s_edge
        att_size : bool, optional (defautl=False)
            If True, adds attribute to the target node containing the size
            of the node measued by the number of neighbors in the sedge
        """

        if not self.is_active_snode(source):
            self.activate_snode(source)

        # #############
        # Default paths
        if path_snode is None:
            path_snode = os.path.join(self.path2snodes, target)
        if path_sedge is None:
            path_sedge = os.path.join(self.path2sedges, e_label)

        # Default name of the target snode and the sedge
        if target is None:
            target = attrib
        if e_label is None:
            e_label = f"{source}_2_{target}"

        # ############
        # Source snode
        s0 = self.snodes[source]

        # ############
        # Target snode

        # Check if each attribute is a list or not.
        # IF not, each attribute value will become a target node.
        # If yes, eath value in each list will become a target node
        if s0.n_nodes > 0:
            atts_in_list = type(s0.df_nodes.iloc[0][attrib]) == list
        else:
            atts_in_list = False

        # Create target snodes
        if atts_in_list:
            print("UNTESTED: CHECK THE CASE OF ATTRIBUTE LISTS")
            breakpoint()
            nodes1 = list(set(
                [a for att_list in s0.df_nodes[attrib] for a in att_list]))
        else:
            nodes1 = list(set(s0.df_nodes[attrib]))
        # Convert type of the node names to string
        # This is required because attributes can be not strings (e.g. int)
        nodes1 = list(map(str, nodes1))

        self.makeSuperNode(target, out_path=path_snode, nodes=nodes1)
        s1 = self.snodes[target]
        # s1 = DataGraph(label=target, path=path_snode)
        # s1.set_nodes(nodes1)
        # self.snodes[target] = s1

        # #########
        # Superedge

        e01 = SEdge(label=e_label, path=path_sedge,
                    label_source=s0.label, label_target=s1.label)
        e01.set_nodes(s0.nodes, s1.nodes)

        # Add edges
        source_nodes = s0.df_nodes[s0.REF].tolist()
        target_nodes = s0.df_nodes[attrib].tolist()
        if type(target_nodes[0]) == list:
            print("UNTESTED: CHECK THE CASE OF ATTRIBUTE LISTS")
            breakpoint()
            edges = [(s, t) for (s, t_list) in zip(source_nodes, target_nodes)
                     for t in t_list]
            source_nodes, target_nodes = zip(*edges)

        # Convert type of the node names to string
        # This is required because attributes can be not strings (e.g. int)
        target_nodes = list(map(str, target_nodes))

        e01.set_edges(source_nodes, target_nodes)

        # Metadata for the new sedge
        e01.metadata['graph']['subcategory'] = 'graph from attributes'

        self.sedges[e_label] = e01

        # Add attribute to target node containing the size of each node,
        # measured as the number of in-links in the sedge.
        if att_size:
            node_sizes_dict = collections.Counter(target_nodes)
            node_sizes = [node_sizes_dict[n] for n in s1.nodes]
        s1.add_attributes('NSize', node_sizes)

        # Update metagraph
        s1_attrib = {'category': attrib}
        self.metagraph.add_single_node(s1.label, attributes=s1_attrib)
        e01_attrib = {'category': 'snode_from_atts',
                      'Type': 'directed',
                      'label': e01.label}

        self.metagraph.add_single_edge(s0.label, s1.label, weight=1,
                                       attributes=e01_attrib)

        return

    def snode_from_eqs(self, source, target=None, path_snode=None,
                       path_sedge=None, e_label=None):
        """
        Generate a new snode and a new sedge from a given snode in the
        supergraph.

        The nodes of the new snode will consist of the equivalence classes
        of the snode.

        An equivalence class is the set of all nodes fully connected by links
        with unit weight

        All nodes from the same equivalence class at the source snode will be
        connected to the same equivalent-class node in the target snode

        Parameters
        ----------
        source: str
            Name of the source snode in the supergraph
        target: str or None
            Name of the target node. If None, default name eq_{source} is
            usedm where {source} is the source name
        path_snode: str or None
            Output path to save the target snode. If None a defautl path is
            used
        path_sedge: str or None
            Output path to save the sedge. If None, a default path is used
        e_label: str or None
            Name of the sedge connecting the source and target snodes. If None,
            a default name {source}_2_{target} is used
        """

        if not self.is_active_snode(source):
            self.activate_snode(source)

        # #############
        # Default paths
        if path_snode is None:
            path_snode = os.path.join(self.path2snodes, target)
        if path_sedge is None:
            path_sedge = os.path.join(self.path2sedges, e_label)

        # ############
        # Source snode
        s0 = self.snodes[source]

        # ############
        # Target snode

        # Default name of the target snode
        if target is None:
            target = f'eq_{source}'
        cluster_ids, Teq = s0.compute_eq_nodes(name=target)

        # Create target snodes
        # We assign numerical names to nodes.
        # nodes1 = list(range(Teq.shape[0]))     # Old (should be equivalent)
        nodes1 = sorted(list(set(cluster_ids)))

        # Compute list of attributes containing the size of each cluster
        cluster_sizes = collections.Counter(cluster_ids)
        atts1 = [cluster_sizes[n] for n in nodes1]

        # Node names must be string, so we convert them to str. Thus,
        # node i will have label 'i'
        nodes1 = list(map(str, nodes1))

        # This is just to check possible errors
        if Teq.shape[0] != len(nodes1):
            logging.error("-- -- The number of clusters is not equal to the "
                          "number of rows in Teq. This is unexpected and "
                          "will likely cause errors")

        s1 = DataGraph(label=target, path=path_snode)
        # Note that we add the feature matrix here.
        s1.set_nodes(nodes1, Teq, save_T=True)
        s1.add_attributes('cardinality', atts1)
        self.snodes[target] = s1

        # #########
        # Superedge

        if e_label is None:
            e_label = s0.label + '_2_' + s1.label
        e01 = SEdge(label=e_label, path=path_sedge,
                    label_source=s0.label, label_target=s1.label)
        e01.set_nodes(s0.nodes, s1.nodes)

        # Add edges
        edge_ids = [(i, c) for i, c in enumerate(cluster_ids)]
        source_ids, target_ids = zip(*edge_ids)

        source_nodes = [s0.nodes[i] for i in source_ids]
        target_nodes = [nodes1[i] for i in target_ids]

        # Convert type of the node names to string
        # This is required because attributes can be not strings (e.g. int)
        target_nodes = list(map(str, target_nodes))

        e01.set_edges(source_nodes, target_nodes)

        # Metadata for the new sedge
        e01.metadata['graph']['subcategory'] = 'graph from attributes'
        self.sedges[e_label] = e01

        # Update metagraph
        s1_attrib = {'category': 'eq_class'}
        self.metagraph.add_single_node(s1.label, attributes=s1_attrib)
        e01_attrib = {'category': 'snode_from_atts',
                      'Type': 'directed',
                      'label': e01.label}

        self.metagraph.add_single_edge(s0.label, s1.label, weight=1,
                                       attributes=e01_attrib)

        return

    def snode_from_edges(self, source, edges, target=None, path_snode=None,
                         path_sedge=None, e_label=None):
        """
        Generate a new snode and a new sedge from a given snode in the
        supergraph and and a list of edges to the new snode

        This method is similar to snode_from_atts. The difference is that
        snode_from_atts takes the edges from an snode attribute, while
        snode_from_edges takes the edges as an input argument.

        Parameters
        ----------
        source: str
            Name of the source snode in the supergraph
        edges: list
            List of edges
        target: str or None
            Name of the target node. If None, default name A_{source} is
            usedm where {source} is the source name
        path_snode: str or None
            Output path to save the target snode. If None a defautl path is
            used
        path_sedge: str or None
            Output path to save the sedge. If None, a default path is used
        e_label: str or None
            Name of the sedge connecting the source and target snodes. If None,
            a default name {source}_2_{target} is used
        """

        if not self.is_active_snode(source):
            self.activate_snode(source)

        # #############
        # Default paths
        if path_snode is None:
            path_snode = os.path.join(self.path2snodes, target)
        if path_sedge is None:
            path_sedge = os.path.join(self.path2sedges, e_label)

        # ############
        # Source snode
        s0 = self.snodes[source]

        # ###############
        # Filtering edges

        # This is to remove edges from nodes not in s0

        # This can be don in one line...
        #     edges = [e for e in edges if e[0] in set(s0.nodes)]
        # ... but it is too much slow. The following is much faster
        logging.info('---- Filtering edges.')
        # Ordered list (with repetitions) of source nodes from the edge list
        sources = [e[0] for e in edges]
        # Select only those source nodes that exist in s0.
        common = set(s0.nodes).intersection(sources)
        # Build a dictionary node: value, where value is 1 only for the
        # selected nodes
        marked_edges = {s: 0 for s in sources}
        for c in common:
            marked_edges[c] = 1

        # Use the dictionary to filter the list of edges
        edges = [e for e in edges if marked_edges[e[0]]]
        logging.info(
            f'---- Only {len(common)} out of {len(s0.nodes)} source nodes '
            f'have some edge.')
        logging.info('---- Isolated source nodes will be ignored')

        # ############
        # Target snode

        # Default name of the target snode
        if target is None:
            target = 'A_' + source

        # Create target snodes
        nodes1 = [x[1] for x in edges]
        source_nodes, target_nodes = list(zip(*edges))
        # Convert type of the node names to string
        # This is required because targets can be not strings (e.g. int)
        target_nodes = list(map(str, target_nodes))
        nodes1 = list(set(target_nodes))

        s1 = DataGraph(label=target, path=path_snode)
        s1.set_nodes(nodes1)
        self.snodes[target] = s1

        # #########
        # Superedge

        if e_label is None:
            e_label = s0.label + '_2_' + s1.label
        e01 = SEdge(label=e_label, path=path_sedge,
                    label_source=s0.label, label_target=s1.label)
        e01.set_nodes(s0.nodes, s1.nodes)

        # Add edges
        e01.set_edges(source_nodes, target_nodes)

        # Metadata for the new sedge
        e01.metadata['graph']['subcategory'] = 'graph from edges'

        self.sedges[e_label] = e01

        # Update metagraph
        s1_attrib = {'category': target}
        self.metagraph.add_single_node(s1.label, attributes=s1_attrib)
        e01_attrib = {'category': 'snode_from_edges',
                      'Type': 'directed',
                      'label': e01.label}

        self.metagraph.add_single_edge(s0.label, s1.label, weight=1,
                                       attributes=e01_attrib)

        return

    def computeSimGraph(self, label, R=None, n_edges=None, n_gnodes=None,
                        similarity='He2', g=1, th_gauss=0.1, rescale=False,
                        blocksize=25_000, useGPU=False, tmp_folder=None,
                        save_every=1e300, verbose=True):
        """
        Calls the snode method to compute a similarity graph

        Parameters
        ----------
        label : str
            Name of the snode where the similarity graph will be computed.
        R : float or None, optional (default=None)
            Radius. Edges link all data pairs at distance lower than R
            This is to forze a sparse graph.
        n_edges : int or None, optional (default=None)
            Target number of edges. n_edges is an alternative to radius. Only
            one of both must be specified
        n_gnodes : int or None, optional (default=None)
            Number of nodes in the source subgraph.
            If None, all nodes are used
            If n_gnodes < no. of rows in self.T, a random subsample is taken.
        similarity : str, optional (default='JS')
            Similarity measure used to compute affinity matrix.
            Available options are:
            (1) 'JS' (1 minus Jensen-Shannon (JS) divergence (too slow));
            (2) 'l1' (1 minus l1 distance);
            (3) 'He' (1 minus squared Hellinger's distance (sklean-based));
            (4) 'He2' (Same as He, but based on a proper implementation);
            (5) 'Gauss' (An exponential function of the squared l2 distance);
            (6) 'l1->JS' (Same as JS, but the graph is computed after
            preselecting edges using l1 distances and a theoretical bound);
            (7) 'He->JS' (Same as JS, but the graph is computed after pre-
            selecting edges using Hellinger's distances and a theoretical bound
            (8) 'He2->JS':Same as He-Js, but using a self implementation of He
        g : int or float, optional (default=1)
            Exponent for the affinity mapping (not used for 'Gauss')
        th_gauss : float, optional (default=0.1)
            Similarity threshold All similarity values below this threshold are
            set to zero. This is only for the gauss method, the rest of them
            compute the threshold automatically from R).
        rescale: bool, optional (default=False)
            If True, affinities are computed from distances by rescaling values
            so that the minimum is zero and maximum is 1.
        blocksize : int, optional (default=25_000)
            Size of each block for affinity computations
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
        """

        if not self.is_active_snode(label):
            self.activate_snode(label)

        self.snodes[label].computeSimGraph(
            R=R, n_gnodes=n_gnodes, n_edges=n_edges, similarity=similarity,
            g=g, th_gauss=th_gauss, rescale=rescale, blocksize=blocksize,
            useGPU=useGPU, tmp_folder=tmp_folder, save_every=save_every,
            verbose=verbose)

        # Update metagraph
        df = self.metagraph.df_nodes    # This is just to abbreviate.
        REF = self.metagraph.REF        # This is just to abbreviate.
        df.loc[df[REF] == label, 'category'] = 'similarity'

        return

    def computeSimBiGraph(self, s_label, t_label, e_label=None, R=None,
                          n_edges=None, n_gnodesS=None, n_gnodesT=None,
                          similarity='He2', g=1, rescale=False,
                          blocksize=25_000, useGPU=False, tmp_folder=None,
                          save_every=1e300, verbose=True):
        """
        Calls the snode method to compute a similarity bipartite graph.

        It assumes that the source and target snodes with their corresponding
        feature matrices already exists in the supergraphs structure.

        A new sedge connecting the source and target snodes will be created.
        If it already exists

        Parameters
        ----------
        s_label : str
            Name of the source snode
        t_label : str
            Name of the target snode
        e_label : str or None, optional (default=None)
            Name of the new s_edge
        R : float or None, optional (default=None)
            Radius. Edges link all data pairs at distance lower than R.
            This is to forze a sparse graph.
        n_edges : int or None, optional (default=None)
            Target number of edges. n_edges is an alternative to radius. Only
            one of both must be specified
        n_gnodesS : int or None, optional (default=None)
            Number of nodes in the source subgraph.
            If None, all nodes are used
            If n_gnodesS < no. of rows in self.Xs, a random subsample is taken.
        n_gnodesT : int or None, optional (default=None)
            Number of nodes in the target subgraph.
            If None, all nodes are used.
            If n_gnodesT < no. of rows in self.Xt, a random subsample is taken.
        similarity : str {'He2', 'He2->JS'}, optional (default='JS')
            Similarity measure used to compute affinity matrix.
            Available options are:
            (1) 'He2' (Same as He, but based on a proper implementation);
            (2) 'He2->JS': 1 minus Jensen-Shannon (JS) divergence
        g : int or float, optional (default=1)
            Exponent for the affinity mapping
        rescale: bool, optional (default=False)
            If True, affinities are computed from distances by rescaling values
            so that the minimum is zero and maximum is 1.
        blocksize : int, optional (default=25_000)
            Size of each block for affinity computations
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
            If False, block-by-block messaging is omitted
        """

        if not self.is_active_snode(s_label):
            self.activate_snode(s_label)
        if not self.is_active_snode(t_label):
            self.activate_snode(t_label)

        # ################
        # Create superedge

        if e_label is None:
            e_label = s_label + '_2_' + t_label
        if self.is_sedge(e_label):
            # Drop existing sedge
            logging.warning(f"-- -- An sedge named {e_label} already exists. "
                            "It will be removed to create the new one.")
            self.drop_sedge(e_label)

        path2sedge = os.path.join(self.path2sedges, e_label)
        self.sedges[e_label] = SEdge(
            label=e_label, path=path2sedge,
            label_source=s_label, label_target=t_label)
        s_nodes = self.snodes[s_label].nodes
        t_nodes = self.snodes[t_label].nodes
        self.sedges[e_label].set_nodes(s_nodes, t_nodes,
                                       Xs=self.snodes[s_label].T,
                                       Xt=self.snodes[t_label].T)
        self.sedges[e_label].computeSimBiGraph(
            R=R, n_gnodesS=n_gnodesS, n_gnodesT=n_gnodesT, n_edges=n_edges,
            similarity=similarity, g=g, rescale=rescale, blocksize=blocksize,
            useGPU=useGPU, tmp_folder=tmp_folder, save_every=save_every,
            verbose=verbose)

        # Update metagraph
        e_attrib = {'category': 'similarity',
                    'Type': 'undirected',
                    'label': e_label}
        self.metagraph.add_single_edge(s_label, t_label, weight=1,
                                       attributes=e_attrib)

        return

    def compute_ppr(self, s_label, t_label=None, th=0.9, inplace=False):
        """
        Calls the snode method to compute a similarity graph

        Parameters
        ----------
        label : str
            Name of the snode where the similarity graph will be computed.
        th : float, optional (default=0.9)
            Threshold over the ppr to create a link
        inplace : bool, optional (default=True)
            If true, the new graph overrides the original graph
        """

        if not self.is_active_snode(s_label):
            self.activate_snode(s_label)

        edges, weights = self.snodes[s_label].compute_ppr(th=th)

        # Create new s_node
        if inplace is False:
            if t_label is None:
                t_label = f'{s_label}_ppr'

            new_nodes = self.snodes[s_label].nodes
            self.makeSuperNode(t_label, nodes=new_nodes, attributes={},
                               edge_class='directed')

            s_node_ids, t_node_ids = zip(*edges)
            source_nodes = [new_nodes[n] for n in s_node_ids]
            target_nodes = [new_nodes[n] for n in t_node_ids]
            self.snodes[t_label].set_edges(source_nodes, target_nodes, weights)

        else:
            t_label = s_label

        # Update metagraph
        df = self.metagraph.df_nodes    # This is just to abbreviate.
        REF = self.metagraph.REF        # This is just to abbreviate.
        df.loc[df[REF] == t_label, 'category'] = 'PPR'

        return

    def graph_layout(self, snode_label, attribute, gravity=1):
        """
        Compute the layout of the given graph

        Parameters
        ----------
        snode_label : str
            Name of the snode
        gravity: int, optional (default=1)
            Gravity parameter of the graph layout method (only for force atlas
            2)
        attribute: str
            Snode attribute used to color the graph
        """

        if not self.is_active_snode(snode_label):
            self.activate_snode(snode_label)

        self.snodes[snode_label].graph_layout(
            color_att=attribute, gravity=gravity)

        return

    def sub_snode(self, xlabel, ynodes, ylabel=None, sampleT=False,
                  save_T=False):
        """
        Subsample snode X using a given subset of nodes.

        The list of nodes may contain nodes that are not in X. These nodes will
        be included in the new graph, with no edges.

        Parameters
        ----------
        xlabel : str
            Name of the snode to be sampled
        ynodes : int or list
            If list, list of nodes of the output subgraph.
            If int, number of nodes to sample. The list of nodes is taken
            at random without replacement from the graph nodes
        ylabel : str or None, optional (default=None)
            Name of the new snode.
            If None, the sampled snode replaces the original one
        sampleT : bool, optional (defaul=False)
            If True, the feature matrix is also sampled, if it exists.
        save_T : bool, optional (default=False)
            If True, the feature matrix T is saved into an npz file.
        """

        if not sampleT and save_T:
            logging.warning(
                "-- save_T=True overrides sampleT, which must be True too."
                "Otherwise the graph and the feature matrix would become "
                "inconsistent.")
            sampleT = True

        if not self.is_active_snode(xlabel):
            self.activate_snode(xlabel)

        # If no ylabel is given, the subsampled snode replaces the original
        if ylabel is None:
            ylabel = xlabel

        # Separate nodes already in x and new nodes
        if type(ynodes) == list:
            xnodes = self.snodes[xlabel].nodes
            ynodes_new = list(set(ynodes) - set(xnodes))
            ynodes_in_x = list(set(ynodes) - set(ynodes_new))
        else:
            ynodes_new = []
            ynodes_in_x = ynodes

        # Subsample snode
        subgraph = self.snodes[xlabel].sub_graph(ynodes_in_x, sampleT=sampleT)
        df_edges = subgraph['edges']

        # Get list of nodes in the same order that the dataframe
        ref = self.snodes[xlabel].REF
        ynodes_in_x = subgraph['nodes'][ref].tolist()

        # Create new snode with the list of edges
        self.makeSuperNode(ylabel, nodes=ynodes_in_x, T=subgraph['T'],
                           save_T=save_T)

        # Add node attributes
        att_names = [x for x in subgraph['nodes'] if x != ref]
        if len(att_names) > 0:
            values = [subgraph['nodes'][x].tolist() for x in att_names]
            self.snodes[ylabel].add_attributes(att_names, values)

        if len(df_edges) > 0:
            source_nodes = df_edges['Source'].tolist()
            target_nodes = df_edges['Target'].tolist()
            weights = df_edges['Weight'].tolist()
            self.snodes[ylabel].set_edges(source_nodes, target_nodes, weights)

        # Add new nodes.
        if len(ynodes_new) > 0:
            self.snodes[ylabel].add_new_nodes(ynodes_new)

        return

    def sub_snode_by_value(self, xlabel, att, value, ylabel=None):
        """
        Subsample snode by the value of a single attribute

        Parameters
        ----------
        xlabel : str
            Name of the snode to be sampled
        att : str
            Name of the attribute to select nodes by value
        value : int or str or list, optional
            Value of the attribute. Only nodes taking this value will be
            selected.
            If value is a list, all nodes taking a value in the list are
            selected.
        ylabel : str or None, optional (default=None)
            Name of the new snode.
            If None, the sampled snode replaces the original one

        Notes
        -----
        Note that the feature matrix, if it exists, is not sampled.
        """

        if not self.is_active_snode(xlabel):
            self.activate_snode(xlabel)

        # If no ylabel is given, the subsampled snode replaces the original
        if ylabel is None:
            ylabel = xlabel
        else:
            self.duplicate_snode(xlabel, ylabel)
        self.snodes[ylabel].filter_nodes_by_value(att, value)

        # Log result
        n_nodes = self.snodes[ylabel].n_nodes
        n_edges = self.snodes[ylabel].n_edges
        logging.info(f"-- -- Nodes with attribute {att} taking a not allowed "
                     f"value removed. {ylabel} now has {n_nodes} nodes and "
                     f"{n_edges} edges")
        return

    def sub_snode_by_novalue(self, xlabel, att, value, ylabel=None,
                             sampleT=False):
        """
        Subsample snode by removing all nodes without a given value of the
        given attribute

        Parameters
        ----------
        xlabel : str
            Name of the snode to be sampled
        att : str
            Name of the attribute to select nodes by value
        value : int or str, optional
            Value of the attribute. Only nodes NOT taking this value will be
            selected
        ylabel : str or None, optional (default=None)
            Name of the new snode.
            If None, the sampled snode replaces the original one
        sampleT : bool, optional (defaul=False)
            If True, the feature matrix is also sampled, if it exists.
        """

        if not self.is_active_snode(xlabel):
            self.activate_snode(xlabel)

        # If no ylabel is given, the subsampled snode replaces the original
        if ylabel is None:
            ylabel = xlabel

        # Select nodes by value
        ynodes = self.snodes[xlabel].get_nodes_by_novalue(att, value)

        # Subsample graph
        self.sub_snode(xlabel, ynodes, ylabel=ylabel, sampleT=sampleT)

        # Log result
        n_nodes = self.snodes[ylabel].n_nodes
        n_edges = self.snodes[ylabel].n_edges
        logging.info(f"-- -- Nodes with value {value} in attribute {att} "
                     f"removed. {xlabel} now has {n_nodes} nodes "
                     f"and {n_edges} edges")
        return

    def sub_snode_by_threshold(self, xlabel, att, th, bound='lower',
                               ylabel=None):
        """
        Subsample snode by the removing all nodes whose value of a given
        attribute is below or above a given threshold

        Parameters
        ----------
        xlabel : str
            Name of the snode to be sampled
        att : str
            Name of the attribute to select nodes by value
        th : int or float
            Value of the attribute. Only nodes taking this value will be
            selected
        bound : str {'lower', 'upper'}, optional (default='lower')
            States if the threshold is a lower (default) or an upper bound.
            If "lower", all nodes with attribute less than the bound are
            removed
        ylabel : str or None, optional (default=None)
            Name of the new snode.
            If None, the sampled snode replaces the original one
        sampleT : bool, optional (defaul=False)
            If True, the feature matrix is also sampled, if it exists.
        """

        if not self.is_active_snode(xlabel):
            self.activate_snode(xlabel)

        # If no ylabel is given, the subsampled snode replaces the original
        if ylabel is None:
            ylabel = xlabel
        else:
            self.duplicate_snode(xlabel, ylabel)
        self.snodes[ylabel].filter_nodes_by_threshold(att, th, bound=bound)

        # Log result
        n_nodes = self.snodes[ylabel].n_nodes
        n_edges = self.snodes[ylabel].n_edges
        aux = {'lower': 'above', 'upper': 'below'}
        logging.info(f"-- -- Nodes with attribute {att} {aux[bound]} {th} "
                     f"removed. {ylabel} now has {n_nodes} nodes and "
                     f"{n_edges} edges")
        return

    def filter_edges_from_snode(self, label, th, rescale=False):
        """
        Removes edges below a given threshold from a given snode

        Parameters
        ----------
        label : str
            Name of the snode
        th : int or float
            Threshold
        rescale : bool, optional (default=False)
            If True, rescale non-zero weights to span from 0 to 1
        """

        if not self.is_active_snode(label):
            self.activate_snode(label)

        self.snodes[label].filter_edges(th, rescale)

        return

    def filter_edges_from_sedge(self, label, th, rescale=False):
        """
        Removes edges below a given threshold from a given snode

        Parameters
        ----------
        label : str
            Name of the snode
        th : int or float
            Threshold
        rescale : bool, optional (default=False)
            If True, rescale non-zero weights to span from 0 to 1
        """

        if not self.is_active_snode(label):
            self.activate_snode(label)

        self.sedges[label].filter_edges(th, rescale)

        return

    def transduce(self, xylabel, n=1, normalize=True, keep_active=None):
        """
        Given snode X and sedge X-Y, compute a graph for Y based on the
        connectivity betwen nodes in Y through edges from Y to X (strored
        in X-Y) and edges in X.

        Parameters
        ----------
        xylabel : str
            Name of the sedge (bipartite graph) X-Y.
            The names of snodes X and Y will be taken from the metadata of the
            bipartite graph
        n : int, optional (default=1)
            Order for K. A positive integer.
            The affinity matrix is a normalized version of F·K^n·F'
        normalize : bool, optional (default=True)
            If True the graph is normalized so that each node has similarity 1
            to itself.
        keep_active : bool, optinoal (default=False)
            If True, snodes and sedge are not deactivated before return.
            If False, X and X-Y are deactivated. Y remains active, otherwise
            changes would be lost.
            If None, the defaul value in self.keep_active is used

        Notes
        -----
        The new graph is stored in the edges of snode Y.
        """

        logging.info("-- Computing transductive graph")

        if keep_active is None:
            keep_active = self.keep_active

        # Make sure that both terminal snodes and the sedge are active
        if not self.is_active_sedge(xylabel):
            self.activate_sedge(xylabel)
        xlabel = self.sedges[xylabel].metadata['graph']['source']
        ylabel = self.sedges[xylabel].metadata['graph']['target']
        if not self.is_active_snode(xlabel):
            self.activate_snode(xlabel)
        if not self.is_active_snode(ylabel):
            self.activate_snode(ylabel)

        # Get superedge
        xygraph = self.sedges[xylabel]

        # Get source and target snodes
        xlabel = xygraph.metadata['graph']['source']
        ylabel = xygraph.metadata['graph']['target']

        xgraph = self.snodes[xlabel]
        ygraph = self.snodes[ylabel]

        if ygraph.n_edges > 0:
            logging.warning(f"-- -- snode {ylabel} contains {ygraph.n_edges}"
                            " edges. They will be removed")

        # Compute the n-th power of the similarity matrix for graph X
        if n == 0:
            Kx = scsp.identity(xgraph.n_nodes, format='dia')
        elif n == 1:
            # The code for n>1 can be used for n=1 too, but the code below
            # avoids duplicating the affinity matrix.
            Kx = xgraph._computeK()
        else:
            K0 = xgraph._computeK()
            Kx = copy.deepcopy(K0)
            for i in range(n - 1):
                Kx = Kx.dot(K0)

        # Compute affinity matrix for the xygraph
        Kxy = xygraph._computeK()

        if Kxy.shape[0] != Kx.shape[0]:
            logging.error(
                "The feature vectors and the similarity matrix have "
                "mismatching dimensions.")

        S = Kxy.transpose().dot(Kx.dot(Kxy))

        # Normalization. This is done to make sure that the similarity of
        # a node with itself is 1.
        if normalize:
            R = scsp.diags(1.0 / np.sqrt(scsp.csr_matrix.diagonal(S)))
            S = R.dot(S.dot(R))

        # Compute lists with origin, destination and value for all edges in the
        # graph affinity matric.
        orig_id, dest_id = S.nonzero()

        # Since the graph is undirected, we select ordered pairs orig_id,
        # dest_id only
        ygraph.edge_ids = list(filter(
            lambda y: y[0] < y[1], zip(orig_id, dest_id)))

        # Compute list of affinity values
        ygraph.weights = [S[y] for y in ygraph.edge_ids]

        # Create pandas structures for edges.
        # Each node is a project, indexed by field REFERENCIA.
        orig_REF = [ygraph.nodes[y[0]] for y in ygraph.edge_ids]
        dest_REF = [ygraph.nodes[y[1]] for y in ygraph.edge_ids]
        ygraph.df_edges = pd.DataFrame(zip(orig_REF, dest_REF, ygraph.weights),
                                       columns=['Source', 'Target', 'Weight'])
        ygraph.df_edges['Type'] = 'Undirected'
        ygraph.n_edges = len(ygraph.weights)

        # Update metadata
        ygraph.metadata['graph'].update({'subcategory': 'transductive',
                                         'order': n,
                                         'normalize': normalize})
        ygraph.update_metadata()

        logging.info(f"-- -- Transductive graph generated with "
                     f"{ygraph.n_nodes} nodes and {ygraph.n_edges} edges")

        # Update metadata
        REF = self.metagraph.REF
        self.metagraph.df_nodes.loc[
            self.metagraph.df_nodes[REF] == ygraph.label, 'category'] = (
                'transductive')

        if not keep_active:
            # Deactivate the snode and sedge that have not changed. Note that
            # ygraph is not deactivated because changes would be lost.
            self.deactivate_snode(xlabel)
            self.deactivate_sedge(xylabel)

        return

    def transitive_graph(self, e_label, xmlabel, mylabel, path_sedge=None,
                         keep_active=None):
        """
        Construct a new superedge XY connecting suprenodes X and Y that are
        linked by an intermediate supernode M (through superedges XM
        and MY)

        To do so, we replace connections x-m-y by connections x-y

        Parameters
        ----------
        e_label : str
            Label of the new superedge
        xmlabel : str
            Label of superedge x-m
        mylabel : str
            Label of superedge m-y
        path_sedge : str
            Path to the ne sedge
        keep_active : bool, optinoal (default=False)
            If True, snodes and sedge are not deactivated before return.
            If False, XM and MY are deactivated. XY remains active, otherwise
            changes would be lost.
            If None, the defaul value in self.keep_active is used
        """

        if keep_active is None:
            keep_active = self.keep_active

        if not self.is_active_sedge(xmlabel):
            self.activate_sedge(xmlabel)
        if not self.is_active_sedge(mylabel):
            self.activate_sedge(mylabel)

        # ################################
        # Check if sedges must be reversed

        # Get origin and target nodes:
        xmlabel0 = self.sedges[xmlabel].metadata['graph']['source']
        xmlabel1 = self.sedges[xmlabel].metadata['graph']['target']
        mylabel0 = self.sedges[mylabel].metadata['graph']['source']
        mylabel1 = self.sedges[mylabel].metadata['graph']['target']

        # Identify the connecting snode (i.e., the 'm' snode).
        mlabel = (set([xmlabel0, xmlabel1]) & set([mylabel0, mylabel1])).pop()
        reverse_xm = (xmlabel0 == mlabel)
        reverse_my = (mylabel1 == mlabel)

        # #############
        # Default paths
        if path_sedge is None:
            path_sedge = os.path.join(self.path2sedges, e_label)

        # ############
        # Input sedges
        exm = self.sedges[xmlabel]
        emy = self.sedges[mylabel]

        # ########################################
        # Source and target nodes of the new sedge
        if reverse_xm:
            source_nodes = exm.get_target_nodes()
            m_nodes_0 = exm.get_source_nodes()
            s_label = xmlabel1
        else:
            source_nodes = exm.get_source_nodes()
            m_nodes_0 = exm.get_target_nodes()
            s_label = xmlabel0
        # Remove prefixes
        source_nodes = [x[2:] for x in source_nodes]
        m_nodes_0 = [x[2:] for x in m_nodes_0]

        if reverse_my:
            target_nodes = emy.get_source_nodes()
            m_nodes_1 = exm.get_target_nodes()
            t_label = mylabel0
        else:
            target_nodes = emy.get_target_nodes()
            m_nodes_1 = exm.get_source_nodes()
            t_label = mylabel1
        # Remove prefixes
        target_nodes = [x[2:] for x in target_nodes]
        m_nodes_1 = [x[2:] for x in m_nodes_1]

        # Check consistency of the sets of intermediate nodes in both sedges
        if set(m_nodes_0) != set(m_nodes_1):
            logging.error(
                f'---- Sedges {xmlabel} and {mylabel} have different sets of'
                f'intermediate nodes')
        elif m_nodes_0 != m_nodes_1:
            logging.error(
                f'---- Intermediate nodes in {xmlabel} and {mylabel} have '
                f'different ordering. Order-independent transitive graphs '
                f'are not available yet')

        # ############
        # Target sedge

        exy = SEdge(label=e_label, path=path_sedge,
                    label_source=s_label, label_target=t_label)
        exy.set_nodes(source_nodes, target_nodes)

        # Compute affinity matrix
        if reverse_xm:
            Kxm = exm._computeK().T
        else:
            Kxm = exm._computeK()
        if reverse_my:
            Kmy = emy._computeK().T
        else:
            Kmy = emy._computeK()

        # Compute affinity matrix for the xygraph
        Kxy = Kxm.dot(Kmy)

        # Compute lists with origin, destination and value for all edges in the
        # graph affinity matric.
        orig_id, dest_id = Kxy.nonzero()
        # List of edges. Each edge is a pair of indices in matrix coordinate
        edge_ids_unshifted = list(zip(orig_id, dest_id))
        # Compute list of affinity values
        exy.weights = [Kxy[e] for e in edge_ids_unshifted]

        # Transform matrix coordinates into node indices. This is because the
        # list of nodes in exy contains source nodes first and target nodes
        # after them.
        dest_id = [x + exy.n_source for x in dest_id]
        # List of edges. Each edge is a pair of indices relative to the node
        # location in the object list of nodes.
        exy.edge_ids = list(zip(orig_id, dest_id))

        # Create pandas structures for edges.
        # Each node is a project, indexed by field REFERENCIA.
        orig_REF = [exy.nodes[e[0]] for e in exy.edge_ids]
        dest_REF = [exy.nodes[e[1]] for e in exy.edge_ids]
        exy.df_edges = pd.DataFrame(zip(orig_REF, dest_REF, exy.weights),
                                    columns=['Source', 'Target', 'Weight'])

        exy.df_edges['Type'] = 'Directed'
        exy.n_edges = len(exy.weights)

        # Update metadata
        exy.metadata['graph'].update({'subcategory': 'transitive'})
        exy.update_metadata()

        # Add new sedge to the supergraph structure
        self.sedges[e_label] = exy

        logging.info(f"---- Transitive graph generated with "
                     f"{exy.n_nodes} nodes and {exy.n_edges} edges")

        # Update metagraph
        e_attrib = {'category': 'transitive',
                    'Type': 'directed',
                    'label': e_label}
        self.metagraph.add_single_edge(s_label, t_label, weight=1,
                                       attributes=e_attrib)

        if not keep_active:
            # Deactivate the snode and sedge that have not changed. Note that
            # ygraph is not deactivated because changes would be lost.
            self.deactivate_sedge(xmlabel)
            self.deactivate_snode(mylabel)

        return

    # def reverse_bigraph(self, xy_label, yx_label=None, path_sedge=None):
    #     """
    #     Given bigraph x->y, generate a reversed graph y->x by converting
    #     source nodes into target nodes and viceversa

    #     Parameters
    #     ----------
    #     xy_label : str
    #         Name of the bigraph
    #     yx_label : str or None, optional (default=None)
    #         Name of the reversed graph. If None, the name is built from the
    #         names of the source and targe nodes.
    #     """

    #     # WARNING: THIS METHOD IS UNDER CONSTRUCTION
    #     breakpoint()

    #     if not self.is_active_sedge(xy_label):
    #         self.activate_sedge(xy_label)
    #     x_label = self.sedges[xy_label].metadata['graph']['source']
    #     y_label = self.sedges[xy_label].metadata['graph']['target']

    #     if yx_label is None:
    #         yx_label = f"{y_label}_2_{x_label}"

    #     # #############
    #     # Default paths
    #     if path_sedge is None:
    #         path_sedge = os.path.join(self.path2sedges, yx_label)

    #     # ############
    #     # Input sedges
    #     eyx = SEdge(label=yx_label, path=path_sedge,
    #                 label_source=y_label, label_target=x_label)

    #     exy = self.sedges[xy_label]

    #     # This code mmay be wrong. Check it.
    #     nodes_orig = [x[2:] for x in exy.nodes if x[0] == "t"]
    #     nodes_dest = [x[2:] for x in exy.nodes if x[0] == "s"]
    #     eyx.set_nodes(nodes_orig, nodes_dest)

    #     nodes = [f"s_{x[2:]}" if x[0] == "t" else f"t_{x[2:]}"
    #              for x in exy.df_nodes[self.REF]]

    #     df_edges = copy.copy(self.df_edges)

    #     target_nodes = exy.df_edges.Source

    #     return

    # #############################
    # Snode handling and processing
    # #############################
    def add_snode_attributes(self, label, att, att_values):
        """
        Parameters
        ----------
        label : str
            Graph to add the new attribute
        att : str or list
            Name of the attribute
            If att_values is a pandas dataframe, 'att' contains the name of
            the column in att_values that will be used as key for merging
        values : list or pandas dataframe or dict
            If list: it contains the attribute values. If names is list, it
            is a list of lists. The order of the values must correspond with
            the order of nodes in the self node.
            If pandas dataframe, dataframe containing the new attribute values.
            This dataframe will be merged with the dataframe of nodes. In
            such case, name
            If dict, the keys must refer to values in the reference column of
            the snode.
        att_values : dataframe containing the attribute values
        """

        if not self.is_active_snode(label):
            self.activate_snode(label)

        self.snodes[label].add_attributes(att, att_values)

        return

    def remove_snode_attributes(self, label, att_names):
        """
        Parameters
        ----------
        label : str
            Graph to add the new attribute
        att_names : str or list
            Name or names of the attributes to remove
        """

        if not self.is_active_snode(label):
            self.activate_snode(label)

        self.snodes[label].remove_attributes(att_names)

        return

    def detectCommunities(self, label, alg='louvain', ncmax=None,
                          comm_label='Comm'):
        """
        Applies the selected community detection algorithm to a given node

        Parameters
        ----------
        label : str
            Name of the snode
        alg : str, opional (default='louvain')
            Community detection algorithm
        ncmax : int or None, optional (default=None)
            Number of communities.
        label : str, optional (default='Comm')
            Label for the cluster indices in the output dataframe
        """

        if not self.is_active_snode(label):
            self.activate_snode(label)

        self.snodes[label].detectCommunities(
            alg=alg, ncmax=ncmax, label=comm_label)

        return

    def remove_isolated_nodes(self, label):
        """
        Removes all isolated nodes in a given snode

        Parameters
        ----------
        label : str
            Name of the snode
        """

        # Activate snode
        if not self.is_active_snode(label):
            self.activate_snode(label)

        self.snodes[label].remove_isolated()

        return

    # ##############
    # Snode analysis
    # ##############
    def cosine_sim(self, xlabel, ylabel):
        """
        Computes the cosine similarity between two supernodes, X and Y.
        The cosine similarity isdefined as follows:

            sim = trace(X' Y) / ||X|| ||Y||

        where ||·|| is the Frobenius norm

        Parameters
        ----------
        xlabel : str
            Name of one supernode
        ylabel : str
            Name of the other supernode

        Returns
        -------
        score : float
            Value of the cosine similarity
        """

        if not self.is_active_snode(xlabel):
            self.activate_snode(xlabel)
        if not self.is_active_snode(ylabel):
            self.activate_snode(ylabel)

        # Check if X and Y have identical lists of nodes
        xnodes = self.snodes[xlabel].nodes
        ynodes = self.snodes[ylabel].nodes

        if xnodes != ynodes:
            # Check if the sets of nodes are also diferent.
            if set(xnodes) == set(ynodes):
                # Align node ordering in both snodes
                self.snodes[xlabel].sort_nodes()
                self.snodes[ylabel].sort_nodes()
            else:
                raise Exception("Both supernodes must have the same nodes")

        # Compute graph matrices
        Kx = self.snodes[xlabel]._computeK(diag=False)
        Ky = self.snodes[ylabel]._computeK(diag=False)

        score = (Kx.multiply(Ky).sum()
                 / (scsp.linalg.norm(Kx) * scsp.linalg.norm(Ky) + EPS))

        return score

    def local_snode_analysis(self, label, parameter):
        """
        Compute local features of nodes in the given snode

        Parameters
        ----------
        label : str
            Name of the snode
        parameter : str
            Name of the local feature
        """

        if not self.is_active_snode(label):
            self.activate_snode(label)

        self.snodes[label].local_graph_analysis(
            parameter=parameter, label=parameter)

        return

    # #######
    # Storage
    # #######
    def save_metagraph(self):

        self.metagraph.saveGraph()

        return

    def save_supergraph(self):
        """
        Saves all active snodes and sedges.
        This means that it will save all snodes and sedges that have been
        uploaded to self.snodes and self.sedges
        """

        for s in self.snodes:
            self.snodes[s].saveGraph()
        for s in self.sedges:
            self.sedges[s].saveGraph()

        self.save_metagraph()

        return

    def export_2_halo(self, e_label, s_att1, s_att2, t_att, t_att2=None):
        """
        Export sedge, with selected attributes, into a csv file, for
        visualization with Halo.

        Parameters
        ----------
        path2sedge : str
            Path to the bipartite graph
        s_att1 : str
            Name of the first attribute of the source node
        s_att2 : str
            Name of the second attribute of the source node
        t_att : str
            Name of the attribute of the target node
        t_att2 : str or None, optional (default=None)
            Name of the second attribute of the target node
            If None, t_att2 is taken equal to t_att

        Returns
        -------
        label_map : dict
            Dictionary of correspondences label_in_graph : label_in_halo
        """

        # ##########
        # LOAD GRAPH

        # Load source snode
        s_label, t_label = self.get_terminals(e_label)

        # Activate sedge and snodes
        if not self.is_active_sedge(e_label):
            self.activate_sedge(e_label)
        if not self.is_active_snode(s_label):
            self.activate_snode(s_label)
        if not self.is_active_snode(t_label):
            self.activate_snode(t_label)

        # #####################################
        # MAKE BIGRAPH DATAFRAME IN HALO FORMAT

        # Read 'Source', 'Target' and 'Weight'
        df_edges = self.sedges[e_label].df_edges
        # Remove prefixes
        df_edges['Source'] = [x[2:] for x in df_edges['Source']]
        df_edges['Target'] = [x[2:] for x in df_edges['Target']]

        # Add key column
        df_edges['KEY'] = list(range(len(df_edges)))

        # Read source attributes 1 and 2
        s_REF = self.snodes[s_label].REF
        df_s_nodes = self.snodes[s_label].df_nodes[
            [s_REF, s_att1, s_att2]].set_index(s_REF)

        # Read target attributes
        t_REF = self.snodes[t_label].REF
        if t_att2 is None:
            df_t_nodes = self.snodes[t_label].df_nodes[
                [t_REF, t_att]].set_index(t_REF)
        else:
            df_t_nodes = self.snodes[t_label].df_nodes[
                [t_REF, t_att, t_att2]].set_index(t_REF)

        # Load attributes into halo dataframe
        df_edges[s_att1] = [
            df_s_nodes.loc[x, s_att1] for x in df_edges['Source']]
        df_edges[s_att2] = [
            df_s_nodes.loc[x, s_att2] for x in df_edges['Source']]
        df_edges[t_att] = [
            df_t_nodes.loc[x, t_att] for x in df_edges['Target']]
        if t_att2 is not None:
            df_edges[t_att2] = [
                df_t_nodes.loc[x, t_att2] for x in df_edges['Target']]

        # Remove commas from data
        for label in [s_att2, t_att, t_att2]:
            df_edges[label] = df_edges[label].apply(
                lambda x: x if type(x) != str else x.replace(' ', '_'))
            df_edges[label] = df_edges[label].apply(
                lambda x: x if type(x) != str else x.replace(',', ''))
            df_edges[label] = df_edges[label].apply(
                lambda x: x if type(x) != str else x.replace('(', '_'))
            df_edges[label] = df_edges[label].apply(
                lambda x: x if type(x) != str else x.replace(')', '_'))
            df_edges[label] = df_edges[label].apply(
                lambda x: x if type(x) != str else x.replace(
                    'Universidad', 'Univ'))

        # Rename columns:
        label_map = {'Source': 'SOURCE_ID', s_att1: 'SOURCE_NM',
                     s_att2: 'SOURCE_CAT', 'Target': 'TARGET_ID',
                     t_att: 'TARGET_CAT', 'Weight': 'WEIGHT'}
        if t_att2 is not None:
            label_map.update({t_att2: 'TARGET_NM'})
        df_edges.rename(columns=label_map, inplace=True)

        # Duplicate some columns
        if t_att2 is None:
            df_edges[['TARGET_NM']] = df_edges[['TARGET_ID']]

        # #################
        # SECURE FORMATTING

        # The following is to make sure that there are no float data that is
        # expected to be str.
        # Make sure that everything is a string
        for label in ['SOURCE_CAT', 'TARGET_CAT']:
            if any(type(x) == float for x in df_edges[label]):
                df_edges[label] = df_edges[label].fillna(999)
                df_edges[label] = df_edges[label].apply(int)
        df_edges.fillna(value="--", inplace=True)

        # ###############
        # Save halo graph

        # Save edges
        if len(df_edges) > 0:
            fpath = os.path.join(self.sedges[e_label].path2graph,
                                 f'halo_{e_label}.csv')
            df_edges.to_csv(fpath, index=False, columns=df_edges.columns,
                            sep=',', encoding='utf-8')
        else:
            logging.warning('-- -- Bipartite graph is empty. Not saved')

        # Clean memory:
        self.deactivate()

        return label_map

    # # #######################################################################
    # # FROM HERE, THE OLD SUPERGRAPH CODE

    # def project(self, sedge_label):
    #     """
    #     Remove from the superlink (orig, dest) all components that do not
    #     link nodes from supernode orig to nodes in supernode dest

    #     Projects graph G over the self graph. This means that all nodes in
    #     the self graph that are not present in G are removed.
    #     """

    #     # The following variables are defined as abbreviations only
    #     g = self.sedges[sedge_label]          # Link graph
    #     Ograph = self.snodes[sedge_label[0]]  # Origin graph
    #     Dgraph = self.snodes[sedge_label[1]]  # Destination graph

    #     # Remove links with unknown origin
    #     g.project(Ograph, orig=Ograph.REF, imag=g.REForig)

    #     # Remove links with unknown destination
    #     g.project(Dgraph, orig=Dgraph.REF, imag=g.REFdest)

    #     # # Remove links with unknown origin
    #     # g.df_nodes = g.df_nodes[
    #     #     g.df_nodes[g.REForig].isin(Ograph.df_nodes[Ograph.REF])]

    #     # # Remove links with unknown destinations
    #     # g.df_nodes = g.df_nodes[
    #     #     g.df_nodes[g.REFdest].isin(Ograph.df_nodes[Dgraph.REF])]

    #     # # Rebuild node list
    #     # g.nodes = g.nodes[g.REF].tolist()

    # def removeSparseEdges(self, sxy_name, mode, n=1):
    #     """
    #     Remove all xy edges for all nodes in x (if mode == 'Source') or y
    #     (if mode == 'Target') with n or less edges
    #     """

    #     if n > 0:
    #         # Count the number of edges per links
    #         gXY = self.sedges[sxy_name]     # Just to abbreviate

    #         if mode == 'Source':
    #             REF = gXY.REForig
    #         elif mode == 'Target':
    #             REF = gXY.REFdest
    #         else:
    #             exit("removeSingleEdges: unknown mode")

    #         # Add new column with the count of links.
    #         # The name of the column is strange to minimize collision
    #         # probability with an existing name
    #         gXY.df_nodes['freq314159'] = (
    #             gXY.df_nodes.groupby(REF)[REF].transform('count'))

    #         # Remove links with counts up to n
    #         gXY.df_nodes = gXY.df_nodes[gXY.df_nodes['freq314159'] > n - 1]

    #         # Remove auxiliary column
    #         gXY.df_nodes.drop('freq314159', axis=1, inplace=True)
    #         gXY.nodes = gXY.df_nodes[gXY.REF].tolist()


