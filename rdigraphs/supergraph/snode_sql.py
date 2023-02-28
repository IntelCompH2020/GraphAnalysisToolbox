#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Python libraries
import pandas as pd
import numpy as np
import random
import copy
import logging
import ipdb

from ETL2.DBIndicadores import DBIndicadores

# Local imports
from rdigraphs.supergraph.snode import DataGraph


class DataGraph_sql(DataGraph):

    """
    Generic class defining a graph of data
    """

    def __init__(self, db_info, fields, max_num_nodes=None, REF='Id',
                 topics_field=None, out_path=None, label="dg"):
        """
        Stores the main attributes of a datagraph object and loads the graph
        data as a list of node attributes from a database
        """

        super(DataGraph, self).__init__(
            fields, max_num_nodes, REF, topics_field, out_path, label)

        # ##################################
        # Variables for the SQL data sources

        # Parameters for the db containing the data
        self.db_info = db_info
        if db_info is not None:
            if 'filterOptions' in db_info:
                self.db_info['filterOptions'] = db_info['filterOptions']
            else:
                self.db_info['filterOptions'] = None
            if 'orderOptions' in db_info:
                self.db_info['orderOptions'] = db_info['orderOptions']
            else:
                self.db_info['orderOptions'] = None

        # Selected fields in the db that will be node attributes.
        self.fields = fields
        self.topics_field = topics_field
        self.base_fields = None   # Fields from the main DB table
        self.sub_fields = None    # Fields to read fromo other tables
        self.db = None            # Database manager

        # Load data from the database
        if db_info is not None:
            self.importData()

        return

    def importData(self):
        """
        Reads the pandas dataframe for the datagraph from a database.
        """

        # #############
        # Read database

        # Open database
        self.db = DBIndicadores(self.db_info['server'], self.db_info['user'],
                                self.db_info['password'], self.db_info['name'])

        # Separate fields from the main table and fields to read from other
        # tables in the db.
        self.base_fields = [x for x in self.fields if not isinstance(x, tuple)]

        # Each sub-field is a tuple whose first component is the name of the
        # table containing the data dn the second component is the list of
        # attributes to load.
        self.sub_fields = [x for x in self.fields if isinstance(x, tuple)]

        # Read nodes and node attributes from database
        idREF = self.base_fields.index(self.REF)
        self.nodes, self.df_nodes = self.db2df(
            self.base_fields, self.db_info, self.max_num_nodes, idREF,
            refCol=self.REF)
        self.n_nodes = len(self.nodes)

        # It topic features have been taken from the database, compute
        # the topic matrix
        if (self.topics_field is not None and self.topics_field in
                self.base_fields):
            self.T = self.getTopics()
            self.df_nodes.drop(self.topics_field, axis=1, inplace=True)
            logging.info("-- -- -- Data loaded with {0} topics".format(
                self.T.shape[1]))

        # Read some node attributes from auxiliary tables.
        db_info = copy.deepcopy(self.db_info)
        db_info['filterOptions'] = None
        db_info['orderOptions'] = None
        for f in self.sub_fields:
            db_info['tablename'] = f[0]

            if len(f) == 3:
                # Note that we are assuming here that the first field in f[1],
                # is used as the reference field (i.e. idREF=0)
                left_on = f[1]
                right_on = f[2][0]
                f_nodes, df_fnodes = self.db2df(f[2], db_info, refCol=f[2][0])
            else:     # if len(f) == 2
                # Note that we are assuming here that the first field in f[1],
                # is used as the reference field (i.e. idREF=0)
                left_on = f[1][0]
                right_on = f[1][0]
                f_nodes, df_fnodes = self.db2df(f[1], db_info, refCol=f[1][0])

            # self.df_nodes = self.df_nodes.merge(df_fnodes, how='left',
            #                                     on=f[1][0])
            self.df_nodes = self.df_nodes.merge(
                df_fnodes, how='left', left_on=left_on, right_on=right_on)

    def db2df(self, fields, db_info, max_num_nodes=None, idREF=0, refCol='Id'):
        """
        Constructs a data frame from the field contained in the database
        specified in db_info.

        Args:
            :fields:   List of fields to read from the database
            :db_info:  Dictionary specifying the database access parameters
            :max_num_nodes: Number of entries to read from the database. If
                       this number is smaller than the db size, a subset
                       is taken by random sampling.
            :idREF:    Index value corresponding to the field in 'fields'
                       that will be used as reference value in the output
                       dataframe.
            :refCol:   Name to give to the reference column (the original
                       field name in the db can be used. This option is
                       provided to manage some particular needs)

        Returns:
            :nodes:    List of components of the reference column in the
                       output dataframe.
            :df_nodes: Output dataframe

        The number of entries taken fron the db is at most max_num_projects
        """

        # #############
        # Read database

        # Load raw data from the database.
        selectOptions = ", ".join(('`' + x + '`' for x in fields))
        rawData = self.db.queryData(
            db_info['tablename'], selectOptions=selectOptions,
            filterOptions=db_info['filterOptions'],
            orderOptions=db_info['orderOptions'])

        # Since we need random sampling, we download the whole dataset, and
        # subsample:
        if (max_num_nodes is not None and max_num_nodes < len(rawData)):
            random.seed(1)
            rawData = random.sample(rawData, max_num_nodes)
        if len(rawData) == 0:
            logging.warning("There are no elements in the database")

        # #########################
        # Create dataframe of nodes

        # Extract nodes
        # idREF = self.base_fields.index(refCol)
        nodes = map(lambda x: x[idREF], rawData)

        # Create pandas structures for nodes.
        # Each node is indexed by field REFERENCIA.
        df_nodes = pd.DataFrame(nodes, columns=[refCol])

        for n, t in enumerate(fields):

            if t != fields[idREF]:   # and t != self.topics_field:
                df_nodes[t] = map(lambda x: x[n], rawData)
                # The following is necessary to save the graph data in a csv
                # file to be read by gephi (not sure why)
                if isinstance(rawData[0][n], unicode):
                    df_nodes[t] = df_nodes[t].str.encode('utf8')

        return nodes, df_nodes

    def getTopics(self):
        """
        Extracts numerical arrays of data from a list of strings.

        Input:
            rawData :A list of lists of strings.
            idx: The element of each rawData element to get.

        Returns:
            NumData: A list of lists of numbers.
        """

        # Get the position of the topic string in rawdata
        topics_str = self.df_nodes[self.topics_field].tolist()

        # Check if all entries have a topic vector
        is_data = map(lambda d: d == '' or d is None, topics_str)
        if np.count_nonzero(is_data) > 0:
            ipdb.set_trace()
            exit("There are nodes without topic vectors")

        # Get the topic vectors
        StrData = map(lambda d: d.split(','), topics_str)
        topics = np.array([[float(c) for c in d] for d in StrData])

        return topics

    def saveModel(self, Id, tipo, nc):
        """
        Save some parameters and results related to the clustering
        algorithm.

        This method is a specific feature for the graphs constructed for the
        FECYT project.
        """

        table = 'modelos'
        keyname = 'id'
        # TIPO: ClusterGeneral, ClusterTIC, ClusterBIO, ClusterENE
        valuename = 'TIPO'
        values = [(tipo, str(Id))]
        self.db.setGenericField(table, keyname, valuename, values)

        # NTPCGRP: Number of clusters
        valuename = 'NTPCGRP'
        values = [(nc, str(Id))]
        self.db.setGenericField(table, keyname, valuename, values)

        # NAMES
        valuename = 'NAMES'
        # Provisionally, we use trivial names for the clusters.
        # Names must be a string in the form
        # 'name1//name2//name3//name4...'
        names = '//'.join(['Cluster' + str(i) for i in range(nc)])
        values = [(names, str(Id))]

        self.db.setGenericField(table, keyname, valuename, values)

        # DESCRIPCION and CONTENIDO: not used

    def exportClusters(self, field, label):

        REF = 'Id'
        data = [tuple(x) for x in self.df_nodes[[label, REF]].values]
        logging.info("-- -- Exporting cluster indices to field " + field)
        self.db.setFields(field, data)

    def saveGraph(self, extralabel="", newREF=True, mode='gnodes'):

        # Change the name of the self.REF column to 'Id'
        # This is a requirement to visualize the graph using Gephi.
        if newREF is True:
            self.df_nodes.rename(columns={self.REF: 'Id'}, inplace=True)

        if mode == 'gnodes':
            # Select only the nodes in the subgraph.
            df2_nodes = self.df_nodes.ix[self.i_to_n]
        else:
            df2_nodes = self.df_nodes

        # Save nodes
        fpath = self.out_path + self.label + extralabel + '_nodes.csv'
        df2_nodes.to_csv(fpath, index=False, columns=self.df_nodes.columns,
                         sep=',', encoding='utf-8')

        # Save edges
        if hasattr(self, 'df_edges'):
            fpath = self.out_path + self.label + extralabel + '_edges.csv'
            self.df_edges.to_csv(fpath, index=False,
                                 columns=self.df_edges.columns,
                                 sep=',', encoding='utf-8')
