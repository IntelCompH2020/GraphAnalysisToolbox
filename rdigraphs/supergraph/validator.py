import os
import random
import numpy as np
import scipy.sparse as scsp
import pandas as pd
import copy
from pathlib import Path

import time
import logging
import configparser
import itertools

# # import configparser
from collections import Counter


# # Set matplotib parameters to allow remote executions
# # import matplotlib
# # matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Local imports
from rdigraphs.supergraph.supergraph import SuperGraph


class Validator(object):
    """
    Task Manager for the graph analyzer.

    The behavior of this class depends on the state of the project, in
    dictionary self.state, with the followin entries:

    - 'isProject'   : If True, project created. Metadata variables loaded
    - 'configReady' : If True, config file succesfully loaded. Datamanager
                      activated.
    """

    _dir_struct = {'snodes': Path('graphs'),
                   'metagraph': Path('metagraph')}

    def __init__(self, corpus_name, DM, model2val, path2val, path2out,
                 epn=100, ref_graph_nodes_init=100_000, ref_graph_prefix='RG',
                 ref_graph_nodes_target=20_000, ref_graph_epn=100,
                 blocksize=25_000, useGPU=False, models_fname='modelo.npz'):
        """
        Initializes the Validator object

        Parameters
        ----------
        corpus_name: str {'S2', 'Crunchbase'}
            Name of the corpus to validate
        DM : DataManager object
            It will provide access to data repositories.
        model2val : dict
            A dictionary of model attributes. Each dictionary must contain the
            following entries: ref_col (the column in the db containing the
            node names), path2nodename (the path to the node names) and
            path2models (the path to the folder containing the models)
        path2val : str or pathlib.Path
            Path to the folder to save validation results
        path2out : str or pathlib.Path, optional (default='output')
            Name of the output folder in the validation path
        epn : int, optional (default=100)
            Average number of edges per node in the graphs used for validation
        ref_graph_prefix : str, optional (default='RG')
            Prefix for the names of the reference graph and their subsampled
            versions
        ref_graph_nodes_init : int, optional (default=100_000)
            Size of the initial set of nodes (only for Crunch)
        ref_graph_nodes_target : int, optional (default=20_000)
            Target numer of nodes in the reference graph (only for CrunchEver)
        ref_graph_epn : int, optional (default=100)
            Target average number of edges per node in the reference graph
        blocksize : int, optional (default=25_000)
            Size of blocks for the computation of similarity graphs. 25_000 is
            ok for computation in a standard PC. Larger values may cause large
            processing times caused by memory swapping.
        useGPU : boolean, optional (defautl=False)
            If True, cupy library will be used to carry out some matrix
            multiplications
        models_fname : str, optional (default='modelo.npz')
            Name of the files containing the topic models to validate
        """

        # Name of the corpus to validate
        self.corpus_name = corpus_name
        # Path to project
        self.path2val = path2val
        self.path2out = path2out
        self.path2rgs = self.path2out / 'ref_graph_sim'
        self.path2var = self.path2out / 'variability'
        self.path2sca = self.path2out / 'scalability'
        self.path2sub = self.path2out / 'subtrained_models'

        # Supergraph object
        self.SG = None

        # Datamanager handler
        self.DM = DM

        # Dictionary of available models
        self.model = model2val
        # Name of the files containing the topic models to validate
        self.models_fname = models_fname
        # Prefix for the names of the reference graph and their subsampled
        # versions
        self.ref_graph_prefix = ref_graph_prefix

        # ###################################
        # Parameters for the reference graphs

        # Size of the initial set of nodes taken from the DB to compute the
        # reference graph (only for Crunch)
        self.n_nodes_db = ref_graph_nodes_init
        # Target number of nodes in the reference graph
        self.n_nodes_rg = ref_graph_nodes_target
        # Target average number of edges per node
        self.epn = epn
        if ref_graph_epn is None:
            ref_graph_epn = epn
        self.ref_graph_epn = ref_graph_epn

        # Size of blocks for the computation of similarities
        self.blocksize = blocksize
        # True if cuda is available
        self.useGPU = useGPU

        # Load supergraph
        p2sn = self.path2val / 'graphs'
        p2mg = self.path2val / 'metagraph'
        self.SG = SuperGraph(path=p2mg, path2snodes=p2sn)

        return

    def readCoordsFromFile(self, fpath, fields=['thetas'], sparse=False,
                           path2nodenames=None, ref_col='corpusid',
                           path2params=None):
        """
        Reads a data matrix from a given path.
        This method assumes a particular data structure of the project

        Parameters
        ----------
        fpath : str or pathlib.Path
            Path to the file that contains the topic model.
        fields : str or list, optional (default=['thetas'])
            Name of the field or fields containing the doc-topic matrix
        sparse : bool, optional (default=False)
            If True, the doc-topic matrix is sparse, otherwise dense
        path2nodenames : str or None, optional (default=None)
            path to file containing metadata (in particular node names). If
            None, the file is assumed to be in fpath, with name
            docs_metadata.csv
        ref_col : str, optional (default='corpusid')
            Name of the column in the metadata file (given by path2nodenames)
            that contains the doc id.

        Returns
        -------
        data_out : dict
            Output data dictionary
        df_nodes : dataframe
            Dataframe of nodes
        """

        # #############
        # DATA LOCATION

        # If there is only one file with extension npz, take it
        fnpz = [f for f in os.listdir(fpath) if f.split('.')[-1] == 'npz']
        if len(fnpz) == 1:
            path2topics = os.path.join(fpath, fnpz[0])
        # otherwise, take the one with the specified names
        else:
            path2topics = os.path.join(fpath, self.models_fname)

        if path2nodenames is None:
            path2nodenames = os.path.join(fpath, 'docs_metadata.csv')

        if path2params is None:
            path2params = os.path.join(fpath, 'train.config')

        # ##############
        # LOADING TOPICS

        data = np.load(path2topics, mmap_mode=None, allow_pickle=True,
                       fix_imports=True, encoding='ASCII')
        logging.info(f'-- -- Topic model loaded from {path2topics}')

        # Since data is and
        data_out = {}
        for field in fields:
            if field == 'thetas' and field not in data:
                # If here, it means that the topic matrix is sparse.
                # We must build it up in the sparse.csr format.
                if 'thetas_data' in data:
                    data_out['thetas'] = scsp.csr_matrix(
                        (data['thetas_data'], data['thetas_indices'],
                         data['thetas_indptr']), shape=data['thetas_shape'])
                else:
                    data_out['thetas'] = scsp.csr_matrix(
                        (data['data'], data['indices'], data['indptr']),
                        shape=data['shape'])
            else:
                data_out[field] = data[field]

        # #############
        # LOADING NAMES

        if os.path.isfile(path2nodenames):
            # I don't like this try, but I have found cases where using
            # lineterminator='\n' the column name is taken with a final '\r'
            # which causes a mismatch error with the expected ref_col name.
            try:
                # Note that node names are loaded as string type.
                df_nodes = pd.read_csv(path2nodenames, usecols=[ref_col],
                                       lineterminator='\n', dtype=str)
            except Exception:
                df_nodes = pd.read_csv(path2nodenames, usecols=[ref_col],
                                       dtype=str)

        else:
            logging.info(f'-- -- File {path2nodenames} with node names does '
                         'not exist. No dataframe of nodes is returned')
            df_nodes = None

        del data

        # ##################
        # LOADING PARAMETERS

        # Read parameters from a config file
        # The config file has no heading section, so we must add it.
        with open(path2params, 'r') as f:
            config_string = '[sec]\n' + f.read()
        cf = configparser.ConfigParser()
        cf.read_string(config_string)

        if os.path.isfile(path2params):
            # We load all parameters, but only some of them will be used
            params = {
                'input': cf.get('sec', 'input'),
                'num-topics': cf.get('sec', 'num-topics'),
                'alpha': cf.get('sec', 'alpha'),
                'optimize-interval': cf.get('sec', 'optimize-interval'),
                'num-threads': cf.get('sec', 'num-threads'),
                'num-iterations': cf.get('sec', 'num-iterations'),
                'doc-topics-threshold': cf.get('sec', 'doc-topics-threshold')}
        else:
            params = None

        return data_out, df_nodes, params

    def compute_reference_graph(self):
        """
        Computes a reference graph for a given corpus, based on metadata.
        """

        # Parameters from the corpus
        corpus_data = self.model
        # The column in the database containing the node identifiers
        ref_col = corpus_data['ref_col']
        # Path to the file containing the metadata, including the node names,
        # which is common to all models
        path2nodenames = corpus_data['path2nodenames']
        path2models = corpus_data['path2models']

        # Path to topic models
        logging.info('-- Reading nodes from the topic model folders')

        # ###############################
        # Read data from the topic models

        # We will compute a reference graph for a subset of docs from the db.
        # We cannot take a random subset of docs from the db, because some
        # docs in the db are not in the topic models.
        # To select docs from the db that exist in the topic models, we need
        # to known what docs are there in the topic models

        # Get an arbitrary topic model to get the list of nodes
        folder_names = os.listdir(path2models)
        folder_names = [
            x for x in folder_names if x[:4] == self.corpus_name[:4]]

        if len(folder_names) == 0:
            logging.error("-- There are no topic models to validate, "
                          "you need to create them first")
            return

        fpath = path2models / folder_names[0]

        df_nodes_tm = self.readCoordsFromFile(
            fpath=fpath, sparse=True, path2nodenames=path2nodenames,
            ref_col=ref_col)[1]
        # This is the complete list of nodes in the topic models

        nodes_tm = df_nodes_tm[ref_col].tolist()
        logging.info(f'-- -- {len(nodes_tm)} nodes in the topic models')

        # Read db table
        if self.corpus_name == 'Crunch':

            # #############################
            # REFERENCE GRAPH FOR COMPANIES
            # #############################

            # Load all nodes and its category attribute from database
            logging.info('-- Reading nodes from database')
            df = self.DM['Crunch'].readDBtable(
                'CompanyCat', limit=None, selectOptions=None,
                filterOptions=None, orderOptions=None)
            # df = self.DM['Crunch'].readDBtable(
            #     'CompanyGroupCat', limit=None, selectOptions=None,
            #     filterOptions=None, orderOptions=None)
            att = 'categoryID'

            # ####################
            # Build feature matrix

            # The goal of this process is to build a binary feature matrix T
            # with size = (number of companies x number of categories).
            # Each row is a company, each column a category. A nonzero value in
            # T[5, 7] means company 5 belongs to category 7.

            # Original dataset
            nodes_db = sorted(list(set(df['companyID'])))
            n_nodes_db = len(nodes_db)
            # Inverse index of nodes
            node_db2ind = dict(zip(nodes_db, range(n_nodes_db)))
            logging.info(f'-- -- Dataset loaded with {n_nodes_db} nodes and '
                         f'{len(df)} attribute assignments')

            row_inds = [node_db2ind[x] for x in df['companyID']]  # .tolist()]
            col_inds = df[att].tolist()
            data = [1] * len(row_inds)
            T = scsp.csr_matrix((data, (row_inds, col_inds)))
            # Feature matrix should be normalized so that each row sums up to 1
            T = scsp.csr_matrix(T / np.sum(T, axis=1))

            # ########################
            # Compute similarity graph

            # Initialize graph with the full feature matrix and all nodes
            graph_name = self.ref_graph_prefix
            self.SG.makeSuperNode(label=graph_name, nodes=nodes_db, T=T)
            # Select nodes that are in the topic models only.
            self.SG.sub_snode(graph_name, nodes_tm, sampleT=True)
            # Subsample graph for the validation analysis. This is to reduce
            # the size of the graphs in order to reduce computation time
            n_gnodes = self.n_nodes_db
            self.SG.sub_snode(graph_name, n_gnodes, sampleT=True)

            # Compute similarity graph based on attributes
            n_edges = self.ref_graph_epn * n_nodes_db
            logging.info('-- -- Computing similarity graph (this may take a '
                         'while...)')
            self.SG.computeSimGraph(
                graph_name, n_edges=n_edges, n_gnodes=n_gnodes, g=1,
                rescale=False, blocksize=self.blocksize, useGPU=self.useGPU,
                tmp_folder=None, save_every=20_000_000, verbose=False)

        elif self.corpus_name == 'S2':

            # ####################################
            # REFERENCE GRAPH FOR SEMANTIC SCHOLAR
            # ####################################

            logging.info('-- Reading nodes from database')
            # Table: S2papers
            # ---- Attributes: paperID, S2paperID, title, lowertitle,
            #      paperAbstract, entities, fieldsOfStudy, s2PdfUrl, pdfUrls,
            #      year, journalVolume, journalPages, isDBLP, isMedline, doi,
            #      doiUrl, pmid, langid, LEMAS
            # ---- No. of rows: 13126503
            # Table: citations
            # ---- Attributes: citationID, S2paperID1, S2paperID2
            # ---- No. of rows: 86260982
            # Load nodes (and the relevant attributes only)
            df_nodes_db = self.DM['S2'].readDBtable(
                'S2papers', limit=None, selectOptions='paperID, S2paperID',
                filterOptions=None, orderOptions=None)
            logging.info(f'-- -- {len(df_nodes_db)} nodes in the database')

            # Select nodes that are in the topic models only.
            df_nodes_db = df_nodes_db[df_nodes_db['paperID'].isin(nodes_tm)]

            # Subsample nodes to reduce computation time during validation
            df_nodes_db = df_nodes_db.sample(n=self.n_nodes_rg, random_state=3)
            logging.info(
                f'-- -- Random sample of {self.n_nodes_rg} nodes selected')

            # Read edges from database
            logging.info('-- Reading citation data from database')
            df_edges = self.DM['S2'].readDBtable(
                'citations', limit=None, filterOptions=None,
                orderOptions=None)
            logging.info(f'-- -- {len(df_edges)} edges in the database')

            # Select edges with a citing paper in the list of nodes only.
            nodes_S2 = df_nodes_db['S2paperID'].tolist()
            df_edges = df_edges[df_edges['S2paperID1'].isin(nodes_S2)]
            edges_orig = df_edges['S2paperID1'].tolist()
            edges_dest = df_edges['S2paperID2'].tolist()
            n_edges = len(edges_orig)
            logging.info(f'-- -- {n_edges} edges from the selected nodes')

            # and, viceversa: select nodes with a cite in df_edges.
            # This is the definite list of nodes for the large reference graph
            df_nodes_db = (
                df_nodes_db[df_nodes_db['S2paperID'].isin(edges_orig)])
            nodes = df_nodes_db['paperID'].tolist()

            # This is the list of S2 ids used for the edges
            nodes_S2 = df_nodes_db['S2paperID'].tolist()
            n_nodes = len(nodes)
            logging.info(
                f'-- -- Selecting {n_nodes} nodes with at least one reference')

            # Get list of unique cited papers
            atts = list(set(df_edges['S2paperID2']))
            n_atts = len(atts)

            # ####################
            # Build feature matrix

            # The goal of this process is to build a binary feature matrix T
            # with size = (number of papers x number of cited papers).
            # Each row is a paper, each column a cited paper. A nonzero value
            # in T[5, 7] means paper 5 cites paper 7.

            # Inverse index of nodes
            nodeS2_2ind = dict(zip(nodes_S2, range(n_nodes)))
            att2ind = dict(zip(atts, range(n_atts)))
            logging.info(
                f'-- Dataset loaded with {n_nodes} nodes, {n_atts} '
                f'attributes and {n_edges} attribute assignments')

            row_inds = [nodeS2_2ind[x] for x in edges_orig]  # .tolist()]
            col_inds = [att2ind[x] for x in edges_dest]  # .tolist()]

            # Compute normalized feature matrix
            data = [1] * len(row_inds)
            T = scsp.csr_matrix((data, (row_inds, col_inds)))
            sumT = scsp.diags(1 / T.sum(axis=1).A.ravel())
            T = scsp.csr_matrix(sumT * T)

            # ########################
            # Compute similarity graph

            # Create datagraph with the full feature matrix
            graph_name = self.ref_graph_prefix
            self.SG.makeSuperNode(label=graph_name, nodes=nodes, T=T)
            # self.SG.sub_snode(graph_name, self.n_nodes_rg, ylabel=graph_name,
            #                   sampleT=True)

            n_edges = self.ref_graph_epn * n_nodes
            self.SG.computeSimGraph(
                graph_name, n_edges=n_edges, n_gnodes=None, g=1, rescale=False,
                blocksize=self.blocksize, useGPU=self.useGPU, tmp_folder=None,
                save_every=20_000_000, verbose=False)

        else:
            logging.error('-- Unknown db')

        # #####################
        # SHOW AND SAVE RESULTS

        # Log some results
        # md = dg.metadata['graph']
        md = self.SG.snodes[graph_name].metadata
        logging.info(f"-- -- Number of nodes: {md['nodes']['n_nodes']}")
        logging.info(f"-- -- Number of edges: {md['edges']['n_edges']}")
        logging.info(f"-- -- Average neighbors per node: "
                     f"{md['edges']['neighbors_per_sampled_node']}")
        logging.info(f"-- -- Density of the similarity graph: "
                     f"{100 * md['edges']['density']} %")

        # Get connected components
        self.SG.snodes[graph_name].detect_cc(label='cc')

        cc = self.SG.snodes[graph_name].df_nodes['cc'].tolist()
        rr = Counter(cc)
        logging.info(f"-- Largest connected component with {rr[0]} nodes")
        ylabel = f'{self.ref_graph_prefix}_{self.n_nodes_rg}'
        self.SG.sub_snode_by_value(graph_name, 'cc', 0)

        self.SG.sub_snode(graph_name, self.n_nodes_rg, ylabel=ylabel)

        # Save graph: nodes and edges
        self.SG.save_supergraph()
        logging.info(f'Reference graph {ylabel} saved, with {self.n_nodes_rg} '
                     f'nodes and {self.SG.snodes[ylabel].n_edges} edges')

        # Reset snode. This is to save memory.
        self.SG.deactivate()

        return

    def subsample_reference_graph(self):
        """
        Computes a reference graph for a given corpus, based on metadata.
        """

        # ##########
        # Parameters

        # Parameters from the corpus
        corpus_data = self.model
        # The column in the database containing the node identifiers
        ref_col = corpus_data['ref_col']
        # Path to the file containing the metadata, including the node names,
        # which is common to all models
        path2nodenames = corpus_data['path2nodenames']
        path2models = corpus_data['path2models']

        # ####################
        # Load reference graph

        logging.info('-- Loading reference graph')
        label_RG = self.ref_graph_prefix
        self.SG.activate_snode(label_RG)

        # ###############################
        # Read data from the topic models

        # We need to read the topic models because some nodes in the reference
        # graph might not exist in the topic models.
        # Thus, we will select from the reference graph the nodes in the topic
        # models only.

        # Get an arbitrary topic model to get the list of nodes
        logging.info('-- Loading topic models')
        fpath = None
        for folder in os.listdir(path2models):
            p = path2models / folder
            if os.path.isdir(p) and self.models_fname in os.listdir(p):
                fpath = p
                break
        if fpath is None:
            logging.error(f"-- No topic models found in {path2models}. ")
            return

        # Read topic model nodes
        df_nodes_tm = self.readCoordsFromFile(
            fpath=fpath, sparse=True, path2nodenames=path2nodenames,
            ref_col=ref_col)[1]

        # This is the complete list of nodes in the topic models
        nodes_tm = df_nodes_tm[ref_col].tolist()
        logging.info(f'-- -- {len(nodes_tm)} nodes in the topic models')

        # ###############
        # Filtering nodes

        # Select nodes that are in the topic models only
        nodes_RG = self.SG.snodes[label_RG].nodes
        nodes_common = sorted(list(set(nodes_RG).intersection(nodes_tm)))
        logging.info(f'-- -- {len(nodes_RG)} nodes in the reference graph')
        logging.info(f'-- -- {len(nodes_common)} common nodes')
        if len(nodes_common) < len(nodes_RG):
            # Remove node out of the topic model.
            self.SG.sub_snode(
                label_RG, nodes_common, sampleT=True, save_T=True)

        # ###########
        # Subsampling

        # Subsample nodes
        ylabel = f'{label_RG}_{self.n_nodes_rg}'
        self.SG.sub_snode(label_RG, self.n_nodes_rg, ylabel=ylabel)

        # #############
        # Save and exit

        # Save graph: nodes and edges
        self.SG.save_supergraph()
        logging.info(f'Reference graph {ylabel} saved, with {self.n_nodes_rg} '
                     f'nodes and {self.SG.snodes[ylabel].n_edges} edges')

        # Reset snodes. This is to save memory.
        self.SG.deactivate()

        return

    def _compute_all_sim_graphs(
            self, path2models, path2nodenames, corpus, label_RG, epn=10,
            ref_col='corpusid'):
        """
        Computes all similarity graphs from the available topic models for a
        given corpus, and save them in a supergraph structure, to be used
        later in validation processes.

        Parameters
        ----------
        path2models : str
            Path specifying the class of models to validate.
        path2nodenames : str
            Path to the file containing the node names
        corpus : str
            Name of the corpus
        label_RG : str
            Name of the reference graph
        epn : float or int, optional (default10)
            Target number of edges per node in the sparse graph.
        col_ref : str, optional (default='corpusid')
            Name of the column in the metadata files containing the node names
        """

        # ################
        # Corpus selection

        # Paths to the models to analyze. The name of each model is also the
        # name of the folder that contains it.
        models = [f for f in os.listdir(path2models)
                  if f.split('_')[0] == corpus and 'interval' in f]

        # ####################
        # Load reference graph

        # Load reference graph from DB or from a file in the supergraph.
        logging.info('-- -- Loading reference graph...')
        breakpoint()
        if label_RG not in self.SG.metagraph.nodes:
            logging.error("-- Reference graph does not exist. Use "
                          "self.compute_reference_graph() first")
            exit()

        # If the reference graph already exists, just load it from file.
        self.SG.activate_snode(label_RG)
        # Take the list of nodes.
        # The semantic graphs will be computed using the same nodes
        nodes_ref = self.SG.snodes[label_RG].nodes
        n_gnodes = len(nodes_ref)
        # We do not need the reference graph for anything else here, so
        # we can remove it from memory.
        self.SG.deactivate_snode(label_RG)

        # #########################
        # Compute similarity graphs

        for i, model in enumerate(models):

            logging.info(f"-- Model {i} ({model}) out of {len(models)}")

            # Select topic model
            path = os.path.join(path2models, model)

            # #####################
            # Document-topic matrix

            # Load doc-topic matrix for the whole corpus
            data, df_nodes, params = self.readCoordsFromFile(
                fpath=path, sparse=True, path2nodenames=path2nodenames,
                ref_col=ref_col)
            T = data['thetas']                     # Doc-topic matrix
            nodes_model = df_nodes[ref_col].astype(str).tolist()
            # nodes = df_nodes[ref_col].tolist()     # List of node ids

            # ###########
            # Subsampling

            # Create graph
            self.SG.makeSuperNode(label=model, nodes=nodes_model, T=T)
            # Take the samples in the reference graph only
            self.SG.sub_snode(model, nodes_ref, sampleT=True, save_T=True)

            n_edges_t = int(epn * n_gnodes)
            self.SG.computeSimGraph(model, n_edges=n_edges_t,
                                    n_gnodes=n_gnodes, verbose=False)

            # Add params of the topic model to the graph metadata
            self.SG.snodes[model].metadata['source_features'] = params

            # Save similarity graph in file
            self.SG.snodes[model].saveGraph()
            # Remove similarity graph from memory
            self.SG.deactivate_snode(model)

            # Save metagraph, which is basicaly a structure describing the
            # collection of available graphs
            # We could do this only once, out of the loop, but it is done
            # here to prevent catastrophic failures
            self.SG.save_metagraph()

        return

    def compute_all_sim_graphs(self):
        """
        Computes all similarity graphs from the available topic models for a
        given corpus, and saves them in a supergraph structure, to be used
        later in validation processes.

        Parameters
        ----------
        ref_graph: str or None, optional (default=None)
            Name of the reference graph. If None, a standard name is computed
            from the corpus name and the number of nodes
        """

        logging.info("Computing all similarity graphs for validation...")
        corpus_data = self.model
        path2nodenames = corpus_data['path2nodenames']
        path2models = corpus_data['path2models']
        ref_col = corpus_data['ref_col']

        # Parameters
        print(f"Number of edges per node: {self.epn}")

        # Validate modesl, one by one..
        label_RG = f'{self.ref_graph_prefix}_{self.n_nodes_rg}'
        self._compute_all_sim_graphs(
            path2models, path2nodenames, self.corpus_name, label_RG,
            epn=self.epn, ref_col=ref_col)

        return

    def _validate_topic_models(self, path2models, path2nodenames, corpus,
                               label_RG, epn=10, ref_col='corpusid',
                               drop_snodes=True):
        """
        Analyzes the influence of the topic model and the similarity graph
        parameters on the quality of the graph.

        The similarity graph is validated using a reference graph.

        Parameters
        ----------
        path2models : str
            Path specifying the class of models to validate.
        path2nodenames : str
            Path to the file containing the node names
        corpus : str
            Name of the corpus
        label_RG : str
            Name of the reference graph
        epn : float or int, optional (default=10)
            Target number of edges per node in the sparse graph.
        col_ref : str, optional (default='corpusid')
            Name of the column in the metadata files containing the node names
        drop_snodes : boolean, optional (default=True)
            If true, similarity graphs are deleted after being processed.
            If false, similarity graphs are saved to be used (for instance,
            for the variability analysis)
        """

        # ################
        # Corpus selection
        # ################

        # Paths to the models to analyze. The name of each model is also the
        # name of the folder that contains it.
        models = [f for f in os.listdir(path2models)
                  if f.split('_')[0] == corpus and 'interval' in f]

        # Load reference graph from DB or from a file in the supergraph
        logging.info('-- -- Loading reference graph...')
        if label_RG not in self.SG.metagraph.nodes:
            logging.error("-- Reference graph does not exist. Use "
                          "self.compute_reference_graph() first")
            exit()

        # If the reference graph already exists, just load it from file.
        self.SG.activate_snode(label_RG)
        nodes_ref = self.SG.snodes[label_RG].nodes
        n_gnodes = len(nodes_ref)

        # #######################
        # Model-by-model analysis
        # #######################

        # Initialize output variables
        n_topics, n_nodes, n_edges, comp_time = [], [], [], []
        cc, cmax, rmax, c_rel, RG_sim = [], [], [], [], []
        alpha, interval, n_run = [], [], []

        run_counter = {}

        for i, model in enumerate(models):

            logging.info(f"-- Model {i} ({model}) out of {len(models)}")

            # Select topic model
            path = os.path.join(path2models, model)

            # #####################
            # Document-topic matrix

            # Load doc-topic matrix for the whole corpus
            data, df_nodes, params = self.readCoordsFromFile(
                fpath=path, sparse=True, path2nodenames=path2nodenames,
                ref_col=ref_col)
            T = data['thetas']                     # Doc-topic matrix
            nodes_model = df_nodes[ref_col].astype(str).tolist()
            # nodes = df_nodes[ref_col].tolist()     # List of node ids

            metadata_i = model.split('_')
            alpha_i = metadata_i[4]
            interval_i = metadata_i[6]
            n_run_i = metadata_i[8]

            # Take metadata information about the model from the model name
            # alpha_i = params['alpha']
            # interval_i = params['optimize-interval']

            # Update execution counter
            if (alpha_i, interval_i) in run_counter:
                run_counter[(alpha_i, interval_i)] += 1
            else:
                # We start counting from 0.
                run_counter[(alpha_i, interval_i)] = 0
            n_run_i = run_counter[(alpha_i, interval_i)]

            # Take metadata information about the model from the model name
            # model_metadata = model.split('_')
            # n_run_i = model_metadata[8]

            # ###########
            # Subsampling

            # # Create graph
            t0 = time.time()
            if not self.SG.is_snode(model):
                self.SG.makeSuperNode(label=model, nodes=nodes_model, T=T)
                # Take the samples in the reference graph only
                self.SG.sub_snode(
                    model, nodes_ref, sampleT=True, save_T=True)

                n_edges_t = int(epn * n_gnodes)
                self.SG.computeSimGraph(model, n_edges=n_edges_t,
                                        n_gnodes=n_gnodes, verbose=False)
            else:
                self.SG.activate_snode(model)
                logging.info('-- -- Semantic graph exists. Loaded from file')
                logging.info('      Thus, computation time measures loading'
                             'time only')
            dt = time.time() - t0

            # Compute connected components
            # self.SG.snodes[model].detect_cc(label='cc')
            self.SG.detectCommunities(model, alg='cc', comm_label='cc')

            # Get output parameters
            md = self.SG.snodes[model].metadata
            n_topics.append(T.shape[1])
            n_nodes.append(n_gnodes)
            n_edges.append(md['edges']['n_edges'])
            rmax.append(md['edges']['R'])
            cc.append(md['communities']['cc']['n_communities'])
            cmax.append(md['communities']['cc']['largest_comm'])
            c_rel.append(md['communities']['cc']['largest_comm'] / n_gnodes)
            comp_time.append(dt)
            alpha.append(alpha_i)
            interval.append(interval_i)
            n_run.append(n_run_i)

            # Compute similarity with the citations graph
            score = self.SG.cosine_sim(label_RG, model)
            RG_sim.append(score)

            # Remove snode, because it is no longer required
            if drop_snodes:
                self.SG.drop_snode(model)
            else:
                self.SG.snodes[model].saveGraph()
                self.SG.deactivate_snode(model)

        # ############
        # Save results

        # Sort result variables by number of topics
        # We need to save the original ordering of the number of topics to
        # sort the cd metrics afterwards.
        (n_topics, models, n_nodes, n_edges, comp_time, cc, cmax, rmax, c_rel,
            comp_time, RG_sim, alpha, interval, n_run) = tuple(zip(*sorted(
                zip(n_topics, models, n_nodes, n_edges, comp_time, cc, cmax,
                    rmax, c_rel, comp_time, RG_sim, alpha, interval,
                    n_run))))

        # Create summary table
        df = pd.DataFrame({'Model': models,
                           'No. of topics': n_topics,
                           'alpha': alpha,
                           'interval': interval,
                           'n_run': n_run,
                           'Radius': rmax,
                           'Time': comp_time,
                           'Number of nodes': n_nodes,
                           'Number of edges': n_edges,
                           'Connected components': cc,
                           'Largest component': cmax,
                           'Relative max CC': c_rel,
                           'Ref. graph similarity': RG_sim})

        print("Summary of results:")
        print(df)

        # Save summary table
        preffix = f'{corpus}_{n_gnodes}_{epn}'
        fname = f'{preffix}.xls'
        if not os.path.exists(self.path2rgs):
            os.makedirs(self.path2rgs)
        out_path = os.path.join(self.path2rgs, fname)
        df.to_excel(out_path)

        # IF snodes have not been removed, the supergraph structure is saved
        if not drop_snodes:
            self.SG.save_supergraph()

        return

    def validate_topic_models(self):
        """
        Analyzes the influence of the topic model on te quality of the
        similarity graphs

        The similarity graph is validated using a citations graph.
        """

        ref_graph = f'{self.corpus_name}_ref_{self.n_nodes_rg}'
        corpus_data = self.model
        ref_col = corpus_data['ref_col']
        path2nodenames = corpus_data['path2nodenames']
        path2models = corpus_data['path2models']

        # Parameters
        print(f"Number of edges per node: {self.epn}")

        # Validate modesl, one by one..
        self._validate_topic_models(
            path2models, path2nodenames, self.corpus_name,
            ref_graph, epn=self.epn, ref_col=ref_col, drop_snodes=False)

        return

    def _analyze_variability(self, path2models, corpus, ref_col='corpusid'):
        """
        Analyzes the variability of the semantic graphs computed from the
        topic models.

        The similarity graph is validated using a reference graph.

        Parameters
        ----------
        path2models : str
            Path specifying the class of models to validate.
        corpus : str
            Name of the corpus
        ref_col : str, optional (default='corpusid')
            Name of the column in the metadata files containing the node names
        """

        # ################
        # Corpus selection
        # ################

        # Paths to the models to analyze. The name of each model is also the
        # name of the folder that contains it.
        models = [f for f in os.listdir(path2models)
                  if f.split('_')[0] == corpus and 'interval' in f]

        # Get list of unique parameterizations.
        model_struct = {}
        for model in models:

            # Take metadata information about the model from the model name
            model_metadata = model.split('_')
            model_group = '_'.join(model_metadata[0:8])

            if model_group in model_struct:
                model_struct[model_group].append(model)
            else:
                model_struct[model_group] = [model]

        # #######################
        # Model-by-model analysis
        # #######################

        # Initialize output variables
        n_topics, epn, alpha, interval, n_run = [], [], [], [], []
        w00, w50, w80, w90 = [], [], [], []
        c00, c50, c80, c90 = [], [], [], []
        groups = []

        for i, (model_group, models_g) in enumerate(model_struct.items()):

            logging.info(
                f"-- Group {i} ({model_group}) out of {len(model_struct)}")

            # Take metadata information about the group from the group name
            groups.append(model_group)
            metadata_i = model_group.split('_')
            n_topics.append(metadata_i[2])
            alpha.append(metadata_i[4])
            interval.append(metadata_i[6])
            # Number of runs
            n_run.append(len(models_g))

            # Loop over the graphs in model_group
            for j, model in enumerate(models_g):

                self.SG.activate_snode(model)
                n_nodes = self.SG.snodes[model].n_nodes
                Wj = self.SG.snodes[model].get_matrix()
                n_e = self.SG.snodes[model].n_edges
                n_n = self.SG.snodes[model].n_nodes
                print(f"-- Number of nodes: {n_n}")
                print(f"-- Number of edges: {n_e}")
                print(f"-- Number of edges (II): {len(Wj.data)}")
                print(f"-- Min-weight: {min(Wj.data)}")
                print(f"-- Max-weight: {max(Wj.data)}")
                if j == 0:
                    epn_i = [n_e / n_n]
                    th_50 = np.percentile(Wj.data, 50)
                    th_80 = np.percentile(Wj.data, 80)
                    th_90 = np.percentile(Wj.data, 90)
                    Wi = copy.copy(Wj)
                else:
                    epn_i.append(n_e / n_n)
                    th_50 = max(th_50, np.percentile(Wj.data, 50))
                    th_80 = max(th_80, np.percentile(Wj.data, 80))
                    th_90 = max(th_90, np.percentile(Wj.data, 90))
                    # Compute the component-wise minimum of all Wj
                    Wi = Wi.minimum(Wj)
                self.SG.deactivate_snode(model)

            # Get parameters aggregating data from all graphs in the group
            epn.append(np.mean(epn_i))
            data = Wi.data
            w00.append(np.sum(data))
            c00.append(np.count_nonzero(data))
            data = data * (data > th_50)
            # data = data * (data > 0.5)
            w50.append(np.sum(data))
            c50.append(np.count_nonzero(data))
            data = data * (data > th_80)
            # data = data * (data > 0.8)
            w80.append(np.sum(data))
            c80.append(np.count_nonzero(data))
            data = data * (data > th_90)
            # data = data * (data > 0.9)
            w90.append(np.sum(data))
            c90.append(np.count_nonzero(data))

        # ############
        # Save results

        # Sort result variables by number of topics
        # We need to save the original ordering of the number of topics to
        # sort the cd metrics afterwards.
        (n_topics, models, epn, alpha, interval, n_run,
            w00, c00, w50, c50, w80, c80, w90, c90) = tuple(zip(*sorted(
                zip(n_topics, models, epn, alpha, interval, n_run,
                    w00, c00, w50, c50, w80, c80, w90, c90))))

        # Create summary table
        df = pd.DataFrame({'Model': groups,
                           'No. of topics': n_topics,
                           'Average edges per node': epn,
                           'alpha': alpha,
                           'interval': interval,
                           'n_run': n_run,
                           'w00': w00, 'c00': c00,
                           'w50': w50, 'c50': c50,
                           'w80': w80, 'c80': c80,
                           'w90': w90, 'c90': c90})

        print("Summary of results:")
        print(df)

        # Save summary table
        preffix = f'{corpus}_{n_nodes}_{self.epn}'
        fname = f'{preffix}.xls'
        if not os.path.exists(self.path2var):
            os.makedirs(self.path2var)
        out_path = os.path.join(self.path2var, fname)
        df.to_excel(out_path)

        return

    def analyze_variability(self):
        """
        Analyzes the influence of the topic model on te quality of the
        similarity graphs

        The similarity graph is validated using a citations graph.

        Parameters
        ----------
        corpus : str {'S2', 'Crunch'}
            Corpus (Pu: Semantic Scholar, or Co: Crunchbase data)
        """

        corpus_data = self.model
        ref_col = corpus_data['ref_col']
        path2models = corpus_data['path2models']

        # Parameters
        print(f"Number of edges per node: {self.epn}")

        # Validate modesl, one by one..
        self._analyze_variability(
            path2models, self.corpus_name, ref_col=ref_col)

        return

    def show_validation_results(self):
        """
        Shows the results of the topic model validation in
        self.validate_topic_models()

        Parameters
        ----------
        path: str
            Path to data
        """

        # ###############
        # Read data files
        # ###############

        # Read the file names in the folder containing the xls reports
        data_dir = self.path2rgs
        data_files = sorted(os.listdir(data_dir))

        # Read all result files
        # sim, rescale, n_nodes, n_edges, ig, tm_class = [], [], [], [], [], []
        n_nodes, epn = [], []
        alpha, n_run, interval = [], [], []

        params, df_dict = {}, {}
        # fname_struct = ['corpus', 'sim', 'rescale', 'n_nodes', 'n_edges',
        #                 'ig', 'tm_class']
        no_files = True
        for f in data_files:
            if f.endswith('.xls'):
                fname = os.path.splitext(f)[0]
                fname_parts = fname.split('_')

                # Read parameters from the file name
                n_nodes_f = fname_parts[1]
                epn_f = fname_parts[2]

                n_nodes.append(n_nodes_f)
                epn.append(epn_f)

                # Read report from file
                fpath = os.path.join(data_dir, f)
                df_dict[fname] = pd.read_excel(fpath)
                params[fname] = {'n': n_nodes_f, 'e': epn_f}

                alpha += df_dict[fname]['alpha'].tolist()
                n_run += df_dict[fname]['n_run'].tolist()
                interval += df_dict[fname]['interval'].tolist()

                no_files = False

        if no_files:
            logging.warning('-- -- No result files available')

        # ############
        # Plot results
        # ############

        # Ordered list of parameters.
        n_nodes = sorted(list(set(n_nodes)))
        epn = sorted(list(set(epn)))
        alpha = sorted(list(set(alpha)))
        n_run = sorted(list(set(n_run)))
        interval = sorted(list(set(interval)))

        # Get the list of all possiblo dataframe columns to visualize
        cols = []
        for name, df in df_dict.items():
            cols += df.columns.tolist()

        # Remove columns that will not become y-coordinates
        cols = list(set(cols) - {'Unnamed: 0', 'Model', 'No. of topics',
                                 'Number of nodes'})

        # Dictionary of abreviations (for the file names)
        cols2plot = list(set(cols) - {'alpha', 'interval', 'n_run'})
        abbreviation = {x: x for x in cols2plot}
        abbreviation.update({'Radius': 'Rad',
                             'Time': 't',
                             'Number of edges': 'ne',
                             'Connected components': 'cc',
                             'Largest component': 'lc',
                             'Relative max CC': 'rc',
                             'Ref. graph similarity': 'cs'})

        # The following nested loop is aimed to make multiple plots from xls
        # files inside the same directory. It is actually not needed if all
        # xls files in the given folder have the same parameter values.
        for n in n_nodes:
            for e in epn:

                fnames = [x for x in df_dict if
                          params[x]['n'] == n and params[x]['e'] == e]

                if len(fnames) == 0:
                    continue

                for var, ab in abbreviation.items():

                    # #################################################
                    # Plot figure for the current value of e, n and var
                    fig, ax = plt.subplots()
                    print(fnames)
                    for fname in fnames:
                        df_f = df_dict[fname]
                        for a in alpha:
                            df_a = df_f[df_f['alpha'] == a]
                            for i in interval:
                                df = df_a[df_a['interval'] == i]
                                x = df['No. of topics']
                                y = df[var]

                                base_line, = ax.plot(x, y, '.')

                                df_av = df.groupby('No. of topics').mean()
                                x = df_av.index
                                y = df_av[var]
                                ax.plot(x, y, '.-', label=f'a={a}, i={i}',
                                        color=base_line.get_color())

                    ax.set_xlabel('No. of topics')
                    ax.set_ylabel(var)
                    ax.set_title(f'Nodes: {n}, Edges per node: {e}, ')
                    ax.legend()
                    ax.grid()
                    plt.show(block=False)

                    # Save figure
                    out_dir = os.path.join(self.path2rgs, 'figs')
                    tag = '_'.join(fname_parts[0:5])
                    fname = f'{tag}_{ab}.png'
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    out_path = os.path.join(out_dir, fname)
                    plt.savefig(out_path)

        return

    def show_variability_results(self):
        """
        Shows the results of the topic model validation in
        self.validate_topic_models()

        Parameters
        ----------
        path: str
            Path to data
        """

        # ###############
        # Read data files
        # ###############

        # Read the file names in the folder containing the xls reports
        data_dir = self.path2var
        data_files = sorted(os.listdir(data_dir))

        # Read all result files
        n_nodes, epn = [], []
        alpha, interval = [], []

        params, df_dict = {}, {}
        # fname_struct = ['corpus', 'n_nodes', 'epn']
        for f in data_files:
            if f.endswith('.xls'):
                fname = os.path.splitext(f)[0]
                fname_parts = fname.split('_')

                # Read parameters from the file name
                n_nodes_f = fname_parts[1]
                epn_f = fname_parts[2]

                n_nodes.append(n_nodes_f)
                epn.append(epn_f)

                # Read report from file
                fpath = os.path.join(data_dir, f)
                df_dict[fname] = pd.read_excel(fpath)
                params[fname] = {'n': n_nodes_f, 'e': epn_f}

                alpha += df_dict[fname]['alpha'].tolist()
                interval += df_dict[fname]['interval'].tolist()

        # ############
        # Plot results
        # ############

        # Ordered list of parameters.
        n_nodes = sorted(list(set(n_nodes)))
        epn = sorted(list(set(epn)))
        alpha = sorted(list(set(alpha)))
        interval = sorted(list(set(interval)))

        # Get the list of all possiblo dataframe columns to visualize
        cols = []
        for name, df in df_dict.items():
            cols += df.columns.tolist()

        # Remove columns that will not become y-coordinates
        cols = list(set(cols) - {'Unnamed: 0', 'Model', 'No. of topics',
                                 'Number of nodes'})

        # Dictionary of abreviations (for the file names)
        cols2plot = list(set(cols) - {'alpha', 'interval', 'n_run'})
        abbreviation = {x: x for x in cols2plot}
        # abbreviation.update({'Radius': 'Rad',
        #                      'Time': 't',
        #                      'Number of edges': 'ne',
        #                      'Connected components': 'cc',
        #                      'Largest component': 'lc',
        #                      'Relative max CC': 'rc',
        #                      'Ref. graph similarity': 'cs'})

        # Index(['Unnamed: 0', 'Model', 'No. of topics',
        # 'Average edges per node', 'alpha', 'interval', 'n_run',
        # 'w00', 'c00', 'w50', 'c50', 'w80', 'c80', 'w90', 'c90'],

        # The following nested loop is aimed to make multiple plots from xls
        # files inside the same directory. It is actually not needed if all
        # xls files in the given folder have the same parameter values.
        for n in n_nodes:
            for e in epn:

                fnames = [x for x in df_dict if
                          params[x]['n'] == n and params[x]['e'] == e]

                if len(fnames) == 0:
                    continue

                for var, ab in abbreviation.items():

                    # #################################################
                    # Plot figure for the current value of e, n and var
                    fig, ax = plt.subplots()
                    print(fnames)
                    for fname in fnames:
                        df_f = df_dict[fname]
                        for a in alpha:
                            df_a = df_f[df_f['alpha'] == a]
                            for i in interval:
                                df = df_a[df_a['interval'] == i]
                                x = df['No. of topics']
                                y = df[var]

                                base_line, = ax.plot(x, y, '.')

                                df_av = df.groupby('No. of topics').mean()
                                x = df_av.index
                                y = df_av[var]
                                ax.plot(x, y, '.-', label=f'a={a}, i={i}',
                                        color=base_line.get_color())

                    ax.set_xlabel('No. of topics')
                    ax.set_ylabel(var)
                    ax.set_title(f'Nodes: {n}, Edges per node: {e}, ')
                    ax.legend()
                    ax.grid()
                    plt.show(block=False)

                    # Save figure
                    out_dir = str(self.path2var / 'figs')
                    tag = '_'.join(fname_parts[0:5])
                    fname = f'{tag}_{ab}.png'
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    out_path = os.path.join(out_dir, fname)
                    plt.savefig(out_path)

        return

    def _analyze_scalability(self, path2models, path2nodenames, corpus,
                             epn=10, ref_col='corpusid'):
        """
        Analyzes the influence of the topic model and the similarity graph
        parameters on the quality of the graph.

        The similarity graph is validated using a reference graph.

        Parameters
        ----------
        path2models : str
            Path specifying the class of models to validate.
        path2nodenames : str
            Path to the file containing the node names
        corpus : str
            Name of the corpus
        epn : float or int, optional (default=10)
            Target number of edges per node in the sparse graph.
        col_ref : str, optional (default='corpusid')
            Name of the column in the metadata files containing the node names
        """

        # ################
        # Corpus selection
        # ################

        # Paths to the models to analyze. The name of each model is also the
        # name of the folder that contains it.
        models = [f for f in os.listdir(path2models)
                  if f.split('_')[0] == corpus and 'interval' in f]

        # Selected models for the analysis:
        sel_n_topics = ['25', '40']
        sel_alpha = ['1.0', '5.0']
        sel_interval = ['0', '10']
        sel_run = ['0']
        # r_nodes = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        r_nodes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        # Filter models
        selected_models = []
        for model in models:
            # Extract model parameters
            metadata_i = model.split('_')

            # Reject model if parameters do not belong to the selected set:
            if ((metadata_i[2] in sel_n_topics)
                    and (metadata_i[4] in sel_alpha)
                    and (metadata_i[6] in sel_interval)
                    and (metadata_i[8] in sel_run)):
                selected_models.append(model)
        logging.info(f"Selected models: {selected_models}")

        # #######################
        # Model-by-model analysis
        # #######################

        # Initialize output variables
        model_name, n_topics, alpha, interval, run = [], [], [], [], []
        n_nodes, n_edges, comp_time, rmax = [], [], [], []

        for i, model in enumerate(selected_models):

            logging.info(
                f"-- Model {i} ({model}) out of {len(selected_models)}")

            # Select topic model
            path = os.path.join(path2models, model)

            # #####################
            # Document-topic matrix

            # Load doc-topic matrix for the whole corpus
            data, df_nodes, params = self.readCoordsFromFile(
                fpath=path, sparse=True, path2nodenames=path2nodenames,
                ref_col=ref_col)
            T = data['thetas']                     # Doc-topic matrix
            nodes_model = df_nodes[ref_col].astype(str).tolist()
            n_nodes_all = len(nodes_model)
            # nodes = df_nodes[ref_col].tolist()     # List of node ids

            # Extract model parameters
            metadata_i = model.split('_')
            n_topics_i = metadata_i[2]
            alpha_i = metadata_i[4]
            interval_i = metadata_i[6]
            run_i = metadata_i[8]

            # Take metadata information about the model from the model name
            # alpha_i = params['alpha']
            # interval_i = params['optimize-interval']

            # ###########
            # Subsampling

            # Create graph
            self.SG.makeSuperNode(label=model, nodes=nodes_model, T=T)

            for r in r_nodes:

                t0 = time.time()
                n_gnodes = int(r * n_nodes_all)
                n_edges_t = int(epn * n_gnodes)
                self.SG.sub_snode(model, n_gnodes, ylabel='subgraph',
                                  sampleT=True, save_T=False)
                self.SG.computeSimGraph('subgraph', n_edges=n_edges_t,
                                        n_gnodes=n_gnodes, verbose=False)
                dt = time.time() - t0

                # ########################
                # Store relevant variables

                # Store model parameters
                model_name.append(model)
                n_topics.append(n_topics_i)
                alpha.append(alpha_i)
                interval.append(interval_i)
                run.append(run_i)

                # Store graph parameters
                n_nodes.append(n_gnodes)
                md = self.SG.snodes['subgraph'].metadata
                n_edges.append(md['edges']['n_edges'])

                # Store output variables
                rmax.append(md['edges']['R'])
                comp_time.append(dt)

                # Remove snode, because it is no longer required
                self.SG.drop_snode('subgraph')

            self.SG.drop_snode(model)

        # ############
        # Save results

        # Sort result variables by number of topics
        # We need to save the original ordering of the number of topics to
        # sort the cd metrics afterwards.
        (n_nodes, n_edges, model_name, n_topics, alpha, interval, run, rmax,
            comp_time) = tuple(zip(*sorted(
                zip(n_nodes, n_edges, model_name, n_topics, alpha, interval,
                    run, rmax, comp_time))))

        # Create summary table
        df = pd.DataFrame({'Model': model_name,
                           'Topics': n_topics,
                           'alpha': alpha,
                           'i': interval,
                           'run': run,
                           'Radius': rmax,
                           'Time': comp_time,
                           'Nodes': n_nodes,
                           'Edges': n_edges})

        print("Summary of results:")
        print(df)

        # Save summary table
        preffix = f'{corpus}_{epn}'
        fname = f'{preffix}.xls'
        if not os.path.exists(self.path2sca):
            os.makedirs(self.path2sca)
        out_path = self.path2sca / fname
        df.to_excel(out_path)

        return

    def analyze_scalability(self):
        """
        Analyzes the scalability of the graph generatio nprocess

        Parameters
        ----------
        corpus : str {'S2', 'Crunch'}
            Corpus (Pu: Semantic Scholar, or Co: Crunchbase data)
        """

        logging.info("-- Scalability of similarity graph computations")
        corpus_data = self.model
        path2nodenames = corpus_data['path2nodenames']
        path2models = corpus_data['path2models']
        ref_col = corpus_data['ref_col']

        # Parameters
        print(f"Number of edges per node: {self.epn}")

        # Validate modesl, one by one...
        self._analyze_scalability(
            path2models, path2nodenames, self.corpus_name, epn=self.epn,
            ref_col=ref_col)

        exit()

        return

    def show_scalability_results(self):
        """
        Shows the results of the scalability analysis done in
        self.analyze_scalability()
        """

        # #######################
        # Configurable parameters
        # #######################

        cols2legend = ['Topics', 'alpha', 'i', 'run']
        cols2y = ['Radius', 'Time', 'Edges']
        cols2x = ['Nodes']

        # Variables to be taken from the file names, and the part of the fname
        # containing the variable
        fname2fig = {'corpus': 0, 'epn': 1}
        # NOT USED.
        # fname2legend = {}
        # fname2y = {}
        # fname2x = {}

        # Dictionary of abreviations (for the file names). Abbreviatios are
        # used to reduce the size of legends and other figure text elements
        abbreviation = {x: x for x in cols2y}
        abbreviation.update({'Radius': 'Rad',
                             'Time': 't',
                             'Edges': 'ne'})

        # ###############
        # Read data files
        # ###############

        # Read the file names in the folder containing the xls reports
        data_dir = self.path2sca
        data_files = sorted(os.listdir(data_dir))

        data2legend = {x: [] for x in cols2legend}
        data2axis = {x: [] for x in cols2x + cols2y}
        metadata = {x: [] for x in fname2fig}

        params, df_dict = {}, {}
        # fname_struct = ['corpus', 'sim', 'rescale', 'n_nodes', 'n_edges',
        #                 'ig', 'tm_class']
        for f in data_files:
            if f.endswith('.xls'):
                fname = os.path.splitext(f)[0]
                fname_parts = fname.split('_')

                # Read parameters from the file name
                metadata = {x: metadata[x] + [fname_parts[loc]]
                            for x, loc in fname2fig.items()}
                params[fname] = {x: fname_parts[loc]
                                 for x, loc in fname2fig.items()}

                # Read report from file
                fpath = os.path.join(data_dir, f)
                df_dict[fname] = pd.read_excel(fpath)

                for x in data2legend:
                    data2legend[x] += df_dict[fname][x].tolist()

                for x in data2axis:
                    data2axis[x] += df_dict[fname][x].tolist()

        # ############
        # Plot results
        # ############

        # Get the unique values of all variables that will not be in x- or y-
        # axes
        for var in cols2legend:
            data2legend[var] = sorted(list(set(data2legend[var])))
        for var in metadata:
            metadata[var] = sorted(list(set(metadata[var])))

        # The following nested loop is aimed to make multiple plots from xls
        # files inside the same directory. It is actually not needed if all
        # xls files in the given folder have the same parameter values.
        for md in itertools.product(*list(metadata.values())):

            # ################################
            # Plot figures for parameter set x

            # List of all files matching the parameter assignment in x
            fnames = [f for f in df_dict
                      if list(params[f].values()) == list(md)]
            if len(fnames) == 0:
                continue
            logging.info(f"-- -- Plotting figures from files {fnames}")

            for var, ab in abbreviation.items():

                # #############################
                # Plot figures for variable var

                for col2x in cols2x:

                    # ###################################
                    # Plot variable var vs variable col2x

                    fig, ax = plt.subplots()

                    for fname in fnames:
                        df_f = df_dict[fname]

                        for p in itertools.product(
                                *list(data2legend.values())):

                            df = copy.copy(df_f)
                            for n, param in enumerate(p):
                                df = df[df[cols2legend[n]] == param]

                            x = df[col2x]
                            y = df[var]

                            base_line, = ax.plot(x, y, '.')

                            df_av = df.groupby(col2x).mean()
                            x = df_av.index
                            y = df_av[var]

                            label = ', '.join(
                                f'{name}={value}'
                                for name, value in zip(cols2legend, p))
                            ax.plot(x, y, '.-', label=label,
                                    color=base_line.get_color())

                    ax.set_xlabel(col2x)
                    ax.set_ylabel(var)
                    title = ', '.join(
                        f'{name}: {value}'
                        for name, value in zip(fname2fig, md))
                    ax.set_title(title)
                    ax.legend()
                    ax.grid()
                    plt.show(block=False)

                # Save figure
                out_dir = os.path.join(self.path2sca, 'figs')
                tag = '_'.join(fname_parts)
                fname = f'{tag}_{ab}.png'
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                out_path = os.path.join(out_dir, fname)
                plt.savefig(out_path)

        return

    def _compute_all_subtrain_simgraphs(self, path2models, corpus,
                                        n_gnodes=20_000, epn=10, reset=False):
        """
        Computes all similarity graphs from the available topic models for a
        given corpus, and save them in a supergraph structure, to be used
        later in validation processes.

        Parameters
        ----------
        path2models : str
            Path specifying the class of models to validate.
        corpus : str
            Name of the corpus
        n_gnodes : int, optional (default 20_000)
            Number of nodes of the subsampled graphs
        epn : float or int, optional (default 10)
            Target number of edges per node in the sparse graph.
        reset : boolean, optional (default=False)
            If True, the simgraph is computed no matter if a previous version
            exists.
        """

        # ################
        # Corpus selection

        # Paths to the models to analyze. The name of each model is also the
        # name of the folder that contains it.
        models = [f for f in os.listdir(path2models)
                  if f.split('_')[0] == 'ntpc']

        # #########################
        # Compute similarity graphs
        for i, model in enumerate(models):

            logging.info(f"-- Model {i} ({model}) out of {len(models)}")

            graph_name = f'{model}_{int(n_gnodes)}_{int(epn)}'

            if reset or not self.SG.is_snode(graph_name):

                # Select topic model
                path = os.path.join(path2models, model)

                # #####################
                # Document-topic matrix

                # Load doc-topic matrix for the whole corpus
                data, df_nodes, params = self.readCoordsFromFile(
                    fpath=path, sparse=True)
                T = data['thetas']                     # Doc-topic matrix

                if i == 0:
                    n_nodes_all = T.shape[0]
                    n_gnodes = min(n_gnodes, n_nodes_all)
                    nodes = sorted(random.sample(range(n_nodes_all), n_gnodes))
                T = T[nodes, :]

                # ###########
                # Subsampling

                # Create graph
                self.SG.makeSuperNode(label=graph_name, nodes=nodes, T=T)
                n_edges_t = int(epn * n_gnodes)
                self.SG.computeSimGraph(
                    graph_name, n_edges=n_edges_t, n_gnodes=n_gnodes,
                    useGPU=True, verbose=False)

                # Add params of the topic model to the graph metadata
                self.SG.snodes[graph_name].metadata['source_features'] = params

                # Save similarity graph in file
                self.SG.snodes[graph_name].saveGraph()
                # Remove similarity graph from memory
                self.SG.deactivate_snode(graph_name)

                # Save metagraph, which is basicaly a structure describing the
                # collection of available graphs
                # We could do this only once, out of the loop, but it is done
                # here to prevent catastrophic failures
                self.SG.save_metagraph()

            else:
                logging.info(f'-- -- Graph {graph_name} already exsits.')

        return

    def _validate_subtrain_models(self, path2models, corpus, label_RG,
                                  epn=10, n_gnodes=20_000, drop_snodes=True):
        """
        Analyzes the influence of the topic model and the similarity graph
        parameters on the quality of the graph.

        The similarity graph is validated using a reference graph.

        Parameters
        ----------
        path2models : str
            Path specifying the class of models to validate.
        corpus : str
            Name of the corpus
        label_RG : str
            Name of the reference graph
        epn : float or int, optional (default=10)
            Target number of edges per node in the sparse graph.
        n_gnodes : int
            Number of nodes in the graphs
        drop_snodes : boolean, optional (default=True)
            If true, similarity graphs are deleted after being processed.
            If false, similarity graphs are saved to be used (for instance,
            for the variability analysis)
        """

        # ################
        # Corpus selection
        # ################
        ratios = [0.02, 0.05, 0.10, 0.25, 1]
        n_gnodes = int(n_gnodes)
        epn = int(epn)

        label_parts = label_RG.split('_')
        ntpc = label_parts[1]
        label_RG = f'{label_RG}_{n_gnodes}_{epn}'

        # Paths to the models to analyze. The name of each model is also the
        # name of the folder that contains it.
        graphs = [f for f in self.SG.metagraph.nodes
                  if (len(f.split('_')) >= 6)
                  and (f.split('_')[0: 3] == ['ntpc', str(ntpc), 'percidx'])
                  and (f.split('_')[4: 6] == [str(n_gnodes), str(epn)])]
        graphs = [g for g in graphs if g != label_RG]

        # Load reference graph from DB or from a file in the supergraph
        if label_RG not in self.SG.metagraph.nodes:
            logging.error("-- Reference graph does not exist. Use "
                          "self.compute_reference_graph() first")
            exit()
        logging.info(f'-- -- Detected graphs: {graphs}')

        # If the reference graph already exists, just load it from file.
        self.SG.activate_snode(label_RG)
        nodes_ref = self.SG.snodes[label_RG].nodes
        n_gnodes = len(nodes_ref)

        # #######################
        # Model-by-model analysis
        # #######################

        # Initialize output variables
        models, n_topics, n_nodes, n_edges, sf = [], [], [], [], []
        rmax, cc, cmax, c_rel, RG_sim = [], [], [], [], []

        for i, graph_name in enumerate(graphs):

            logging.info(f"-- Model {i} ({graph_name}) out of {len(graphs)}")

            metadata_i = graph_name.split('_')
            ratio_i = ratios[int(metadata_i[3])]
            model_i = '_'.join(metadata_i[:4])

            # ###########
            # Subsampling

            # # Create graph
            self.SG.activate_snode(graph_name)

            # Compute connected components
            self.SG.snodes[graph_name].detect_cc(label='cc')

            # Get output parameters
            md = self.SG.snodes[graph_name].metadata
            models.append(model_i)
            n_topics.append(ntpc)
            n_nodes.append(n_gnodes)
            n_edges.append(md['edges']['n_edges'])
            sf.append(ratio_i)
            rmax.append(md['edges']['R'])
            cc.append(md['communities']['cc']['n_communities'])
            cmax.append(md['communities']['cc']['largest_comm'])
            c_rel.append(md['communities']['cc']['largest_comm'] / n_gnodes)

            # Compute similarity with the citations graph
            score = self.SG.cosine_sim(label_RG, graph_name)
            RG_sim.append(score)

            # Remove snode, because it is no longer required
            if drop_snodes:
                self.SG.drop_snode(graph_name)
            else:
                self.SG.snodes[graph_name].saveGraph()
                self.SG.deactivate_snode(graph_name)

        # ############
        # Save results

        # Sort result variables by number of topics
        # We need to save the original ordering of the number of topics to
        # sort the cd metrics afterwards.
        (sf, models, n_topics, n_nodes, n_edges, rmax, cc, cmax, c_rel,
            RG_sim) = tuple(zip(*sorted(
                zip(sf, models, n_topics, n_nodes, n_edges, rmax, cc, cmax,
                    c_rel, RG_sim))))

        # Create summary table
        df = pd.DataFrame({'Model': models,
                           'Topics': n_topics,
                           'Sampling factor': sf,
                           'Nodes': n_nodes,
                           'Edges': n_edges,
                           'Radius': rmax,
                           'Connected components': cc,
                           'Largest component': cmax,
                           'Relative max CC': c_rel,
                           'Ref. graph similarity': RG_sim})

        print("Summary of results:")
        print(df)

        # Save summary table
        preffix = f'{corpus}_ntpc_{ntpc}_{n_gnodes}_{epn}'
        fname = f'{preffix}.xls'
        if not os.path.exists(self.path2sub):
            os.makedirs(self.path2sub)
        out_path = os.path.join(self.path2sub, fname)
        df.to_excel(out_path)

        # IF snodes have not been removed, the supergraph structure is saved
        if not drop_snodes:
            self.SG.save_supergraph()

        return

    def validate_subtrain_models(self):
        """
        Analyzes the influence of the topic model on the quality of the
        similarity graphs

        The similarity graph is validated using a citations graph.
        """

        epn = self.epn
        epn = 2_000
        n_gnodes = 4_000

        logging.info("-- Computing all similarity graphs for validation...")
        logging.info(f"-- -- Number of edges per node: {epn}")

        path2models = Path('/home/jarenas/github/TM/small_corpus_test')

        # Compute graphs...
        self._compute_all_subtrain_simgraphs(
            path2models, self.corpus_name, n_gnodes=n_gnodes, epn=epn)

        # Validate modesl, one by one..
        ref_graph = 'ntpc_120_percidx_5'
        self._validate_subtrain_models(
            path2models, self.corpus_name, ref_graph, epn=epn,
            n_gnodes=n_gnodes, drop_snodes=False)

        # Validate modesl, one by one..
        ref_graph = 'ntpc_50_percidx_5'
        self._validate_subtrain_models(
            path2models, self.corpus_name, ref_graph, epn=epn,
            n_gnodes=n_gnodes, drop_snodes=False)

        return

    def show_subtrain_results(self):
        """
        Shows the results of the scalability analysis done in
        self.analyze_scalability()
        """

        # #######################
        # Configurable parameters
        # #######################

        cols2legend = ['Topics']
        cols2y = ['Radius', 'Edges', 'Connected components',
                  'Largest component', 'Relative max CC',
                  'Ref. graph similarity']
        cols2x = ['Sampling factor']

        # Variables to be taken from the file names, and the part of the fname
        # containing the variable
        fname2fig = {'corpus': 0, 'Nodes': 3}
        fname2legend = {'epn': 4}
        # fname2fig = {'corpus': 0, 'epn': 4}
        # fname2legend = {'Nodes': 3}

        # Dictionary of abreviations (for the file names). Abbreviatios are
        # used to reduce the size of legends and other figure text elements
        abbreviation = {x: x for x in cols2y}
        abbreviation.update({'Radius': 'Rad',
                             'Edges': 'ne',
                             'Connected components': 'cc',
                             'Largest component': 'largest_cc',
                             'Ref. graph similarity': 'RGS'})

        # ###############
        # Read data files
        # ###############

        # Read the file names in the folder containing the xls reports
        data_dir = self.path2sub
        data_files = sorted(os.listdir(data_dir))

        data2legend = {x: [] for x in cols2legend}
        data2axis = {x: [] for x in cols2x + cols2y}
        metadata = {x: [] for x in fname2fig}

        params, df_dict = {}, {}
        # fname_struct = ['corpus', 'sim', 'rescale', 'n_nodes', 'n_edges',
        #                 'ig', 'tm_class']
        for f in data_files:
            if f.endswith('.xls'):
                fname = os.path.splitext(f)[0]
                fname_parts = fname.split('_')

                # Read parameters from the file name
                metadata = {x: metadata[x] + [fname_parts[loc]]
                            for x, loc in fname2fig.items()}
                params[fname] = {x: fname_parts[loc]
                                 for x, loc in fname2fig.items()}

                # Read report from file
                fpath = os.path.join(data_dir, f)
                df_dict[fname] = pd.read_excel(fpath)

                for x in data2legend:
                    data2legend[x] += df_dict[fname][x].tolist()

                for x in data2axis:
                    data2axis[x] += df_dict[fname][x].tolist()

        # ############
        # Plot results
        # ############

        # Get the unique values of all variables that will not be in x- or y-
        # axes
        for var in cols2legend:
            data2legend[var] = sorted(list(set(data2legend[var])))
        for var in metadata:
            metadata[var] = sorted(list(set(metadata[var])))

        # The following nested loop is aimed to make multiple plots from xls
        # files inside the same directory. It is actually not needed if all
        # xls files in the given folder have the same parameter values.
        for md in itertools.product(*list(metadata.values())):

            # ################################
            # Plot figures for parameter set x

            # List of all files matching the parameter assignment in x
            fnames = [f for f in df_dict
                      if list(params[f].values()) == list(md)]
            if len(fnames) == 0:
                continue
            logging.info(f"-- -- Plotting figures from files {fnames}")

            for ylabel, ab in abbreviation.items():

                # ###############################
                # Plot figures for variable ylabel

                for xlabel in cols2x:

                    # ###################
                    # Plot ylabel vs xlabel

                    fig, ax = plt.subplots()

                    for fname in fnames:
                        # Preffix for the legends
                        fname_parts = fname.split('_')
                        legend0 = ', '.join(
                            f'{name}={fname_parts[v]}'
                            for name, v in fname2legend.items())

                        df_f = df_dict[fname]

                        for p in itertools.product(
                                *list(data2legend.values())):

                            df = copy.copy(df_f)
                            for n, param in enumerate(p):
                                df = df[df[cols2legend[n]] == param]

                            if len(df) > 0:
                                x = df[xlabel]
                                y = df[ylabel]

                                base_line, = ax.plot(x, y, '.')

                                df_av = df.groupby(xlabel).mean()
                                x = df_av.index
                                y = df_av[ylabel]

                                legend1 = ', '.join(
                                    f'{name}={value}'
                                    for name, value in zip(cols2legend, p))
                                if len(legend0) > 0 and len(legend1) > 0:
                                    label = legend0 + ', ' + legend1
                                else:
                                    label = legend0 + legend1
                                ax.semilogx(x, y, '.-', label=label,
                                            color=base_line.get_color())

                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(ylabel)
                    title = ', '.join(
                        f'{name}: {value}'
                        for name, value in zip(fname2fig, md))
                    ax.set_title(title)
                    ax.legend()
                    ax.grid()
                    plt.show(block=False)

                    # Save figure
                    out_dir = os.path.join(self.path2sub, 'figs')
                    var_names = list(metadata.keys())
                    tag = '_'.join(f'{k}_{v}' for k, v in zip(var_names, md))
                    fname = f'{tag}_{ab}.png'
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    out_path = os.path.join(out_dir, fname)
                    plt.savefig(out_path)
                    plt.close()

        return


