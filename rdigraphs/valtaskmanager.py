import logging

# # Local imports
from rdigraphs.sgtaskmanager import SgTaskManager

# #####################
# PROBABLY USELESS
from pathlib import Path

import platform
# This is to solve a known incompatibility issue between matplotlib and
# tkinter on mac os.
if platform.system() == 'Darwin':     # Darwin is the system name for mac os.
    # IMPORTANT: THIS CODE MUST BE LOCATED BEFORE ANY OTHER IMPORT TO
    #            MATPLOTLIB OR TO A LIBRARY IMPORTING FROM MATPLOTLIB
    import matplotlib
    matplotlib.use('TkAgg')

# ####################################
# Imports for the VALIDATION module
from rdigraphs.supergraph.validator import Validator


class ValTaskManager(SgTaskManager):
    """
    Main class of the Everis project.

    The behavior of this class depends on the state of the project, in
    dictionary self.state, with the followin entries:

    - 'isProject'   : If True, project created. Metadata variables loaded
    - 'cfReady'     : If True, config file succesfully loaded. Datamanager
                      activated.
    - 'dbReady'     : It True, the project can connect to a database
    """

    # This is a dictionary that contains a list to all subdirectories
    # that should exist in the project folder

    _valid_corpus = ['S2', 'K']

    _dir_struct = {'val': Path('.'),
                   'valmodels': Path('models'),
                   'valoutput': Path('output')}

    _ref_col = {'S2': 'paperID', 'RG': 'pmid', 'K': 'pmid'}
    _corpus_name = 'K'
    _label_RG = 'RG'

    def __init__(self, path2project, paths2data):
        """
        Initializes the validation task manager object

        Parameters
        ----------
        path2project : str
            Path to the graph processing project
        paths2data : dict
            Paths to data sources
        """

        super().__init__(path2project, paths2data)

        self.path2project = Path(self.path2project)
        print('-- Task Manager object succesfully initialized')

        # Path to the validation folder for the given corpus
        self.path2val = self.path2project / self._dir_struct['val']
        # Output path to the given corpus
        self.path2out = self.path2project / self._dir_struct['valoutput']

        return

    def setup(self):
        """
        Sets up the project. To do so:
            - Loads the configuration file and initializes the data manager.
            - Informs on whether the DDBB structure is ready or not
        """

        # Creates Data Manager object. Upon creation the object
        # will try to connect only to the Database of the project itself

        super().setup()

        ##################
        # Validator object
        self.models_2_validate = {}

        for corpus in self._valid_corpus:
            ref_col = self._ref_col[corpus]
            self.models_2_validate[corpus] = {
                'ref_col': ref_col,
                'path2nodenames': Path(self.path2tm) / 'metadata_models.csv',
                'path2models': Path(self.path2tm)}

        # Other validation parameters
        self.val_params = {
            'epn': self.global_parameters['validate_all_models']['epn'],
            'ref_graph_prefix': self.global_parameters[
                'validate_all_models']['ref_graph_prefix'],
            'ref_graph_epn': self.global_parameters[
                'validate_all_models']['ref_graph_epn'],
            'ref_graph_nodes_init': self.global_parameters[
                'validate_all_models']['ref_graph_nodes_init'],
            'ref_graph_nodes_target': self.global_parameters[
                'validate_all_models']['ref_graph_nodes_target'],
            'blocksize': self.global_parameters['algorithms']['blocksize'],
            'useGPU': self.global_parameters['algorithms']['useGPU'] == 'True'}

        self.state['configReady'] = True

        logging.info('-- Project setup finished')

        return

    def compute_all_sim_graphs(self):
        """
        Computes all similarity graphs from the available topic models for a
        given corpus, and save them in a supergraph structure, to be used
        later in validation processes.
        """

        # Path to the topic models folder
        model2val = self.models_2_validate[self._corpus_name]
        V = Validator(self._corpus_name, self.DM, model2val, self.path2val,
                      self.path2out, **self.val_params)
        V.compute_all_sim_graphs()

        return

    def compute_reference_graph(self):
        """
        Computes a reference graph for a given corpus, based on metadata.
        """

        # Path to the topic models folder
        model2val = self.models_2_validate[self._corpus_name]
        V = Validator(self._corpus_name, self.DM, model2val, self.path2val,
                      self.path2out, **self.val_params)
        V.compute_reference_graph()

        return

    def subsample_reference_graph(self):
        """
        Computes a reference graph for a given corpus, based on metadata.
        """

        # Path to the topic models folder
        model2val = self.models_2_validate[self._corpus_name]
        V = Validator(self._corpus_name, self.DM, model2val, self.path2val,
                      self.path2out, **self.val_params)
        V.subsample_reference_graph()

        return

    def validate_topic_models(self):
        """
        Analyzes the influence of the topic model on te quality of the
        similarity graphs

        The similarity graph is validated using a citations graph.
        """

        # Path to the topic models folder
        model2val = self.models_2_validate[self._corpus_name]
        V = Validator(self._corpus_name, self.DM, model2val, self.path2val,
                      self.path2out, **self.val_params)
        V.validate_topic_models()

        return

    def show_validation_results(self):
        """
        Shows the results of the topic model validation in
        self.validate_topic_models()
        """

        # Path to the topic models folder
        model2val = self.models_2_validate[self._corpus_name]
        V = Validator(self._corpus_name, self.DM, model2val, self.path2val,
                      self.path2out, **self.val_params)
        V.show_validation_results()

        return

    def analyze_variability(self):
        """
        Analyzes the influence of the topic model on te quality of the
        similarity graphs

        The similarity graph is validated from the analisys of the variability
        of node relationships in the graph
        """

        # Path to the topic models folder
        model2val = self.models_2_validate[self._corpus_name]
        V = Validator(self._corpus_name, self.DM, model2val, self.path2val,
                      self.path2out, **self.val_params)
        V.analyze_variability()

        return

    def show_variability_results(self):
        """
        Shows the results of the topic model validation in
        self.validate_topic_models()
        """

        # Path to the topic models folder
        model2val = self.models_2_validate[self._corpus_name]
        V = Validator(self._corpus_name, self.DM, model2val, self.path2val,
                      self.path2out, **self.val_params)
        V.show_variability_results()

        return

    def analyze_scalability(self):
        """
        Analyzes the influence of the topic model on te quality of the
        similarity graphs

        The similarity graph is validated using a citations graph.
        """

        # Path to the topic models folder
        model2val = self.models_2_validate[self._corpus_name]
        V = Validator(self._corpus_name, self.DM, model2val, self.path2val,
                      self.path2out, **self.val_params)
        V.analyze_scalability()

        return

    def show_scalability_results(self):
        """
        Shows the results of the topic model validation in
        self.validate_topic_models()
        """

        # Path to the topic models folder
        model2val = self.models_2_validate[self._corpus_name]
        V = Validator(self._corpus_name, self.DMs, model2val, self.path2val,
                      self.path2out, **self.val_params)
        V.show_scalability_results()

        return

    def validate_subtrain_models(self):
        """
        Validates topics models obtained using a reduced corpus, using a
        gold standard based o a large corpus
        """

        # Path to the topic models folder
        model2val = self.models_2_validate[self._corpus_name]
        V = Validator(self._corpus_name, self.DM, model2val, self.path2val,
                      self.path2out, **self.val_params)
        V.validate_subtrain_models()

        return

    def show_subtrain_results(self):
        """
        Shows the results of the topic model validation in
        self.validate_subtrain_models()
        """

        # Path to the topic models folder
        model2val = self.models_2_validate[self._corpus_name]
        V = Validator(self._corpus_name, self.DM, model2val, self.path2val,
                      self.path2out, **self.val_params)
        V.show_subtrain_results()

        return
