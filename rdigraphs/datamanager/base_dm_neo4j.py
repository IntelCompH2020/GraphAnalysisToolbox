"""
This class provides general functionality for managing a Neo4J database

* reading specific fields (with the possibility to filter by field values)
* storing calculated values in the dataset

Created on Sep 06 2018

@author: Manu A. Vázquez

"""

import typing
import pathlib
import socket

import numpy as np
import pandas as pd
import colorama

from neo4j import GraphDatabase

# Local Imports
from . import cypher
from . import util

# color is reset at the end of every print
colorama.init(autoreset=True)

# color alias for the sake of brevity
color_info = colorama.Fore.LIGHTWHITE_EX
color_caveat = colorama.Fore.LIGHTRED_EX
color_reset = colorama.Style.RESET_ALL


class BaseDMneo4j:
	"""
	Base class for interacting with a Neo4j database.

	"""

	def __init__(self, db_server: str, db_password: str,
		         db_user: str ='neo4j') -> None:
		"""
		Initializer

		Parameters
		----------
		db_server : str
			The URL for the server
		db_password : str
			User Password
		db_user : str, optional
			User login
		"""

		self.driver = GraphDatabase.driver(db_server, auth=(db_user, db_password))

	def __del__(self):
		"""
		Tidy up stuff after before deleting the object.

		"""

		if hasattr(self, 'driver'):
			self.driver.close()

	def reset_database(self) -> None:
		"""
		Reset the database, deleting everything.

		"""

		with self.driver.session() as session:

			# several cypher statements need to be sent...
			for statement in cypher.reset_database():

				# cypher statement is run
				session.run(statement)

	def properties_of_label(self, label: str) -> typing.List[str]:
		"""
		Returns all the properties (across all the nodes) of a given label.

		Parameters
		----------
		label : str
			The label (type)

		Returns
		-------
		out: list
			A list with the properties

		"""

		labels = {k: v for k, v in self.get_db_structure().items() if v['type'] == 'node'}

		return list(labels[label]['properties'].keys())

	def properties_of_relationship(self, relationship_type: str) -> typing.List[str]:
		"""
		Returns all the properties of a given relationship.

		Parameters
		----------
		relationship_type : str
			The type of the relationship

		Returns
		-------
		out: list
			A list with the properties

		"""

		relationships = {k: v for k, v in self.get_db_structure().items() if v['type'] == 'relationship'}

		return list(relationships[relationship_type]['properties'].keys())

	def get_db_structure(self) -> dict:
		"""
		Returns meta-data.

		Returns
		-------
		out: dictionary
			Metadata

		"""

		with self.driver.session() as session:

			# a cypher statement requesting the relationship types
			statement_result = session.run(cypher.get_metadata())

		return statement_result.value()[0]

	def read_nodes(
			self, label: str, limit: int = None, select_options=None,
			filter_options=None, order_options=None) -> pd.DataFrame:
		"""
		Reads nodes from the database.

		Parameters
		----------
		label : str
			Label of the nodes
		limit : int
			Maximum number of nodes
		select_options : unused
		filter_options : unused
		order_options : unused

		Returns
		-------
		out: Pandas dataframe
			Every row contains information about a node

		"""

		# all the existing properties for this node are retrieved
		properties = self.properties_of_label(label)

		# a dictionary for the results with one entry per property
		res = {k: [] for k in properties}

		with self.driver.session() as session:

			# cypher statement is run
			statement_result = session.run(cypher.match_label(label, limit))

		# the above `statement_result` encompasses several records
		for record in statement_result:

			# this assumes the first value ("returned thing" in the cypher statement) is the one requested
			record_dic = record.value()

			# we loop through all the properties of this label...
			for k in properties:

				# if the property exists in this record, its value is stored in the corresponding entry of the
				# dictionary; it it doesn't, a `NaN` is put instead
				res[k].append(record_dic.get(k, np.nan))

		# the dictionary is used to build a dataframe, and returned afterwards
		return pd.DataFrame(res)

	def read_edges(self, relationship_type: str, limit: int = None) -> typing.Optional[pd.DataFrame]:
		"""
		Reads edges from the database.

		Parameters
		----------
		relationship_type : str
			Type of the relationship
		limit : int
			Maximum number of edges

		Returns
		-------
		out: Pandas dataframe
			Every row contains information about a single edge

		"""

		with self.driver.session() as session:

			# cypher statement is run
			statement_result = session.run(cypher.match_relationship(
				relationship_type, limit))

		# if the query doesn't yield any result...
		if not statement_result.peek():

			print(f'{color_info}the query did not produce any result..."None" returned', end='')
			return None

		# the labels of the nodes are "peeked" beforehand
		# NOTE: this assumes
		#   - the relationship only involves two labels (types of nodes)
		#   - a single relationship between the nodes (the `[0]` part)
		#   - each node has a single label
		labels = [next(iter(n.labels)) for n in statement_result.peek().value().relationships[0].nodes]

		# there should be only two labels
		assert len(labels) == 2

		# for every label, *all* the existing properties
		labels_properties = {l: self.properties_of_label(l) for l in labels}

		# *all* (across all the instances) the properties that might show up in this relationship
		relationship_properties = self.properties_of_relationship(relationship_type)

		# a `dict` to fill in with the results
		res = {(l + '_' + p): [] for l, properties in labels_properties.items() for p in properties}

		# for every property of this relationship...
		for p in relationship_properties:

			# ...an empty list is initialized
			res[p] = []

		# the above `statement_result` encompasses several records
		for record in statement_result:

			# this assumes the first value ("returned thing" in the cypher
			# statement) is the one requested
			value = record.value()

			# this assumes a single relationship between the nodes
			rel = value.relationships[0]

			# for every node involved in the relationship...
			for node in rel.nodes:

				# the first and (by assumption) only label of this node
				label = next(iter(node.labels))

				# for every property of this particular label...
				for p in labels_properties[label]:

					# the value is stored in the appropriate key (it should
					# already be in the dictionary)
					res[label + '_' + p].append(node.get(p, np.nan))

			# for every property of this relationship...
			for p in relationship_properties:

				# ...the value is stored in the appropriate key (it should
				# already be in the dictionary)
				res[p].append(rel.get(p, np.nan))

		# the dictionary is used to build a dataframe, and returned afterwards
		return pd.DataFrame(res)

	def drop_node_property(self, label: str, property: str) -> None:
		"""
		Deletes a property from nodes.

		Parameters
		----------
		label : str
			The label of the node
		property : str
			The name of the property

		Returns
		-------

		"""

		with self.driver.session() as session:

			# cypher statement is run
			session.run(cypher.remove_property(label, property))

	def drop_relationship(self, relationship_type: str) -> None:
		"""
		Deletes a relationship.

		Parameters
		----------
		relationship_type : str
			The type of the relationship

		"""

		with self.driver.session() as session:

			# cypher statement is run
			session.run(cypher.remove_relationship(relationship_type))

	def write_dataframe(
			self, df: pd.DataFrame, source: typing.Tuple[str, typing.List[str]],
			destination: typing.Tuple[str, typing.List[str]],
			edge: typing.Tuple[str, typing.List[str]]) -> None:
		"""
		Writes a dataframe in the database

		Parameters
		----------
		df : pandas Dataframe
			The data
		source : tuple of a str and a list of str
			The first element in the tuple is the label for the `source` node,
			and the second one is a list with the columns of the dataframe that
			will become the properties of the node
		destination : tuple of a str and a list of str
			The first element in the tuple is the label for the `destination`
			node, and the second one is a list withthe columns of the dataframe
			that will become the properties of the node
		edge : tuple of a str and a list of str
			The first element in the tuple is the label for the `edge` node,
			and the second one is a list with the columns of the dataframe that
			will become the properties of the node
		"""

		with self.driver.session() as session:

			# loop over the columns for source, destination and edge
			# NOTE: the "_" in each tuple would capture the index (irrelevant here)
			for (_, source_columns), (_, destination_columns), (_, edges_columns) in zip(
					df[source[1]].iterrows(), df[destination[1]].iterrows(), df[edge[1]].iterrows()):

				# NOTE: properties dictionaries are built by applying `dict` over (Pandas) Series
				statement = cypher.merge_relationship_with_properties_clause(
					origin_label=source[0], destination_label=destination[0], relationship_type=edge[0],
					origin_properties=dict(source_columns), destination_properties=dict(destination_columns),
					relationship_properties=dict(edges_columns))

				# cypher statement is run
				session.run(statement)

	def make_edges(
			self, df: pd.DataFrame, source: typing.Tuple[str, typing.Tuple[str, str]],
			destination: typing.Tuple[str, typing.Tuple[str, str]],
			relationship: typing.Tuple[str, typing.Dict[str, str]]):
		"""
		Makes edges between nodes as specified in a dataframe; if the requested nodes don't exist, they are created

		Parameters
		----------
		df : pandas Dataframe
			The data
		source : tuple of a str and a tuple of two str
			The first element is the name of the column (within this DataFrame) specifying the `source` node; the second
			is a tuple whose first element is the *label* in the graph and whose second is the property (in the nodes
			with the aforementioned label) that must match the values in the column.
		destination : tuple of a str and a tuple of two str
			The first element is the name of the column (within this DataFrame) specifying the `destination` node;
			the second is a tuple whose first element is the *label* in the graph and whose second is the property (in
			the nodes with the aforementioned label) that must match the values in the column.
		relationship : tuple of a str and a dict
			The first element is the *type* of the relationship to be created between nodes; the second is a dictionary
			that maps columns in this DataFrame to *properties* of the relationship.
		"""

		# a (sub)DataFrame including only the columns for the properties of the relationship
		relationship_properties = df[list(relationship[1].keys())]

		# columns are renamed as requested
		relationship_properties.columns = relationship[1].values()

		with self.driver.session() as session:

			# loop over the columns for source, destination and edge
			# NOTE: the "_" in the tuple would capture the index (irrelevant here)
			for source_property, destination_property, (_, relationship_cols) in zip(
					df[source[0]], df[destination[0]], relationship_properties.iterrows()):

				statement = cypher.make_relationship(
					origin_label=source[1][0], destination_label=destination[1][0], relationship_type=relationship[0],
					origin_properties={source[1][1]: source_property}, relationship_properties=dict(relationship_cols),
					destination_properties={destination[1][1]: destination_property})

				# cypher statement is run
				session.run(statement)

	def export_graph(self, label_nodes, path2nodes, col_ref_nodes, label_edges, path2edges):
		"""
		Export graph to Neo4J

		Parameters
		----------
		label_nodes : str or tuple of str
			If str, all nodes of the same type
			If tuple, one type for sources, the other for destinations
		path2nodes : str
			Path to the file of nodes: a csv file with one column for the
			node names, and possibly other columns for attributes (that may
			be numeric or str)
		col_ref_nodes : str
			Name of the column in the file of nodes containing the node names
		label_edges : str
			Type of relationship for all edges.
		path2edges : str
			Path to the edges file: a csv file with 3 columns: 
			Source, Target, Weight
		"""

		# the name of the machine running this code
		localhost = socket.gethostname()

		# paths are turned into `pathlib` objects
		nodes_file, edges_file = pathlib.Path(path2nodes), pathlib.Path(path2edges)

		# the first row in the `DataFrame` is read just to get the names of the columns
		df = pd.read_csv(nodes_file, header=0, nrows=1)

		assert col_ref_nodes in df

		# ------- index

		# a statement for adding an index for the type of node about to be added
		statement = cypher.create_indexes(label_nodes, [col_ref_nodes])[0]

		with self.driver.session() as session:

			# ...the cypher statement is run
			session.run(statement)

		# ------- nodes

		# a server for the nodes file is built ("start" and "stop" functions)
		start_nodes_file_server, stop_nodes_file_server = util.simple_http_server(port=4001, path=nodes_file.parent)

		# server is started
		start_nodes_file_server()

		# an auxiliary object to help assemble the statement for loading the nodes
		# cypher_load_table = cypher.LoadTable(nodes_file.parent)
		cypher_load_table = cypher.LoadTable(pathlib.Path('/'), data_access=f'http://{localhost}:4001/')

		# the required parameters...
		parameters = {
			'SQL columns to neo4j properties': {c: c for c in df.columns},
			'neo4j label': label_nodes, 'intermediate csv file': nodes_file.name}

		# ...are passed to that object, which returns the statement
		statement = cypher_load_table.assemble(parameters)

		with self.driver.session() as session:

			# ...the cypher statement is run
			session.run(statement)

		# server is stopped
		stop_nodes_file_server()

		# ------- edges

		start_edges_file_server, stop_edges_file_server = util.simple_http_server(port=4001, path=edges_file.parent)

		start_edges_file_server()

		# an auxiliary object to help assemble the statement for loading the edges
		# cypher_load_join_table = cypher.LoadJoinTable(edges_file.parent)
		cypher_load_join_table = cypher.LoadJoinTable(pathlib.Path('/'), data_access=f'http://{localhost}:4001/')

		# the required parameters...
		parameters = {
			'merge reciprocated relationships': False,
			'origin': {
				'neo4j label': label_nodes, 'SQL matching column': 'Source', 'neo4j matching property': col_ref_nodes
			},
			'destination': {
				'neo4j label': label_nodes, 'SQL matching column': 'Target', 'neo4j matching property': col_ref_nodes
			},
			'SQL columns to neo4j properties': {'Weight': 'Weight'}, 'relationship type': label_edges,
			'intermediate csv file': edges_file.name
		}

		# ...are passed to that object, which returns the statement
		statement = cypher_load_join_table.assemble(parameters)

		with self.driver.session() as session:

			# ...the cypher statement is run
			session.run(statement)

		stop_edges_file_server()