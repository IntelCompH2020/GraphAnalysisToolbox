import typing
import pathlib


def labels() -> str:
	"""
	Returns a statement to get all the labels (node types) in the database.

	Returns
	-------
	out: str
		Neo4j statement

	"""

	return 'CALL db.labels()'


def relationship_types() -> str:
	"""
	Returns a statement to get all the relationship types in the database.

	Returns
	-------
	out: str
		Neo4j statement

	"""

	return 'CALL db.relationshipTypes()'


def get_metadata() -> str:
	"""
	Returns a statement to get the "schema" of the database.

	Returns
	-------
	out: str
		Neo4j statement

	"""

	return 'CALL apoc.meta.schema'


def drop_null_valued_keys(d: dict):
	"""
	Convenience function to get rid of null (`None`) values in a dictionary.

	Parameters
	----------
	d : dict
		Any dictionary

	Returns
	-------
	out: dict
		The input dictionary without null values

	"""

	return {k: v for k, v in d.items() if v}

# =============================================================


def create_indexes(label: str, properties: typing.List[str]) -> typing.Union[str, typing.List[str]]:
	"""
	Returns a statement that creates indexes on a given label.

	Parameters
	----------
	label : str
		The label (node type) on which to create the index
	properties : list
		A list of properties to be indexed on the given label

	Returns
	-------
	out: str
		Neo4j statement

	"""

	return [f"CREATE INDEX ON :{label}({prop});" for prop in properties]


def create_constraint_clause(label: str, node: str = 'node') -> str:
	"""
	Returns the *part* of a statement that creates a constraint.

	Parameters
	----------
	label : str
		The label (node type) on which to create the index
	node : str, optional
		Name of the node when referred in another statement

	Returns
	-------
	out: str
		Neo4j statement

	"""

	return f"CREATE CONSTRAINT ON ({node}:`{label}`)"


def assert_uniqueness_clause(property: str, node: str = 'node') -> str:
	"""
	Returns the *part* of a statement that ensures a property of a node is unique.

	Parameters
	----------
	property : str
		Name of the mean-to-be-unique property
	node : str, optional
		Name of the node (coming from other statement)

	Returns
	-------
	out: str
		Neo4j statement

	"""

	return f"ASSERT {node}.`{property}` IS UNIQUE"


def create_unique_constraint(label: str, property: str) -> str:
	"""
	Returns a statement to creates a unique constraint.

	Parameters
	----------
	label : str
		The label of the node
	property : str
		The property of the node

	Returns
	-------
	out: str
		Neo4j statement

	"""

	return create_constraint_clause(label) + ' ' + assert_uniqueness_clause(property) + ';'

# =============================================================


def match_label(label: str, limit: int = None, dummy: str = 'node') -> str:
	"""
	Returns a statement that matches and returns a label.

	Parameters
	----------
	label : str
		The label of the node
	limit : int, optional
		The maximum number of matches
	dummy : str, optional
		Name of the node when referred in another statement

	Returns
	-------
	out: str
		Neo4j statement

	"""

	limit_clause = f'LIMIT {limit}' if limit else ''

	return f"""\
	MATCH ({dummy}:{label})
	RETURN {dummy}
	{limit_clause};
	"""


def match_relationship(
		relationship: str, limit: int = None, surrogate_relationship: str = 'rel', surrogate_path: str = 'path') -> str:
	"""
	Returns a statement that matches relationships.

	Parameters
	----------
	relationship : str
		The (type of the) relationship
	limit : int, optional
		The maximum number of matches
	surrogate_relationship : str, optional
		Name of the relationship when referred in another statement
	surrogate_path : str, optional
		Name of the path when referred in another statement

	Returns
	-------
	out: str
		Neo4j statement

	"""

	limit_clause = f'LIMIT {limit}' if limit else ''

	return f'''\
	MATCH {surrogate_path}=()-[{surrogate_relationship}:{relationship}]->()
	RETURN {surrogate_path}
	{limit_clause}'''


def remove_property(label: str, property: str, surrogate_node: str = 'node') -> str:
	"""
	Returns a statement that removes a property from a node.

	Parameters
	----------
	label : str
		The label of the node
	property : str
		The name of the property
	surrogate_node : str, optional
		Name of the node when referred in another statement

	Returns
	-------
	out: str
		Neo4j statement

	"""

	return f'''\
	MATCH ({surrogate_node}:{label})
	REMOVE {surrogate_node}.`{property}`
	'''


def remove_relationship(relationship_type: str, var: str = 'r') -> str:
	"""
	Returns a statement that removes a relationship.

	Parameters
	----------
	relationship_type : str
		The type of the relationship
	var : str, optional
		Name of the relationship when referred in another statement

	Returns
	-------
	out: str
		Neo4j statement

	"""

	return f'''\
	MATCH ()-[{var}:{relationship_type}]-> ()
	DELETE {var}
	'''


def reset_database() -> typing.List[str]:
	"""
	Returns a statement that resets the database, i.e., deletes everything.

	Returns
	-------
	out: str
		Neo4j statement

	"""

	return ['MATCH (n) DETACH DELETE n', 'CALL apoc.schema.assert({},{},true)']


def create_or_merge_node(label: str, properties: dict = {}, name: str = "A", merge: bool = False) -> str:
	"""
	Returns a statement that creates or merges a new node.

	Parameters
	----------
	label : str
		The label of the node
	properties : dictionary
		The properties of the node
	name : str, optional
		Name of the node when referred in another statement
	merge : boolean
		Whether a "merge" statement rather than a "create" one is to be returned

	Returns
	-------
	out: str
		Neo4j statement

	"""

	return f'{"MERGE" if merge else "CREATE"} ({name}:{label}{dic_to_properties(properties)})'


def create_or_merge_relationship(
		origin: str, destination: str, rel_type: str, properties: dict = {}, name: str = "rel",
		merge: bool = False) -> str:
	"""
	Returns a statement that creates or merges a new relationship.

	Parameters
	----------
	origin : str
		Name (identifier/variable) of the origin node
	destination : str
		Name (identifier/variable) of the destination node
	rel_type : str
		Relationship type
	properties : dictionary
		Properties to be added to the relationship
	name : str, optional
		Name of the relationship when referred to in another statement
	merge : boolean
		Whether a "merge" statement rather than a "create" one is to be returned

	Returns
	-------
	out: str
		Neo4j statement

	"""

	return (
		f'{"MERGE" if merge else "CREATE"} '
		f'({origin}) -[{name}:{rel_type}{dic_to_properties(properties)}]-> ({destination})'
	)


def merge_relationship_with_properties_clause(
		origin: str = 'origin', destination: str = 'destination', origin_label: str = '', destination_label: str = '',
		relationship_type: str = '', origin_properties: dict = {}, relationship_properties: dict = {},
		destination_properties: dict = {}, relationship: str = 'rel', arrowhead: str = '>'):
	"""
	Returns a clause that merges a relationship specifying properties for each entity.

	Parameters
	----------
	origin : str
		Name (identifier/variable) of the origin node
	destination : str
		Name (identifier/variable) of the destination node
	origin_label : str
		Label of the origin node
	destination_label : str
		Label of the destination node
	relationship_type : str
		Type of the relationship
	origin_properties : dictionary
		Properties of the origin node
	relationship_properties : dictionary
		Properties of the relationship
	destination_properties : dictionary
		Properties of the destination node
	relationship : str, optional
		Name of the relationship when referred to in another statement
	arrowhead : string
		Suffix controlling whether the relationship is (bi)directional or not

	Returns
	-------
	out: str
		Neo4j statement

	"""

	# assembly of something along the lines of, e.g.,
	# MERGE (a:character {`name`:Rick})
	origin = f'MERGE ({origin}{":" if origin_label else ""}{origin_label} {dic_to_properties(origin_properties)})'

	# -[rel:BECOMES {`how`:boom}]->
	relationship = (
		f'-[{relationship}{":" if relationship_type else ""}{relationship_type} '
		f'{dic_to_properties(relationship_properties)}]-{arrowhead}')

	# (b:fruit {`type`:pickle})
	destination = (
		f'({destination}{":" if destination_label else ""}'
		f'{destination_label} {dic_to_properties(destination_properties)})')

	# altogether: MERGE (a:character {`name`:Rick}) -[rel:BECOMES {`how`:boom}]->(b:fruit {`type`:pickle})
	return origin + relationship + destination


def make_relationship(
		origin_label: str = '', destination_label: str = '',
		relationship_type: str = '', origin_properties: dict = {}, relationship_properties: dict = {},
		destination_properties: dict = {}):
	"""
	Returns a statement to make a relationship while merging properties in the nodes involved.

	Parameters
	----------
	origin_label : str
		Label of the origin node
	destination_label : str
		Label of the destination node
	relationship_type : str
		Type of the relationship
	origin_properties : dictionary
		Properties of the origin node
	relationship_properties : dictionary
		Properties of the relationship
	destination_properties : dictionary
		Properties of the destination node

	Returns
	-------
	out: str
		Neo4j statement

	"""

	# assembly of something along the lines of, e.g.,
	# MERGE (origin:character {`name`:Rick})
	origin = create_or_merge_node(origin_label, origin_properties, name='origin', merge=True)

	# assembly of something along the lines of, e.g.,
	# MERGE (destination:character {`name`:Morty})
	destination = create_or_merge_node(destination_label, destination_properties, name='destination', merge=True)

	# assembly of something along the lines of, e.g.,
	# CREATE (origin) -[rel:KINSHIP{`type`:'grandpa'})]-> (destination)
	rel = create_or_merge_relationship('origin', 'destination', relationship_type, relationship_properties)

	# altogether:
	# MERGE (origin:character {`name`:Rick})
	# MERGE (destination:character {`name`:Morty})
	# CREATE (origin) -[rel:KINSHIP{`type`:'grandpa'})]-> (destination)
	return origin + '\n' + destination + '\n' + rel + ';'


def dic_to_properties(d: dict):
	"""
	Takes a dictionary and returns a string amenable to be used in cypher.

	Parameters
	----------
	d : dict
		A dictionary mapping properties to values.

	Returns
	-------
	out: str
		A piece of cypher statement specifying properties and values.

	"""

	# if the dictionary is not empty...
	if d:
		return '{' + ', '.join([f"`{k}`:'{v}'" for k, v in d.items()]) + '}'
	# if the dictionary is empty...
	else:
		# ...there is no need to add anything
		return ''


# -------------------------------------

class Statement:
	"""
	Base class for a Neo4j statement.

	"""

	pass


class DataLoadingStatement(Statement):
	"""
	Class to load data from csv files into Neo4j.

	"""

	def __init__(self, csv_files_path: pathlib.Path, csv_line: str = 'line', data_access: str = 'file://') -> None:
		"""
		Initializer.

		Parameters
		----------
		csv_files_path : pathlib's Path
			Path to the csv files
		csv_line : str, optional
			Name of a line from the file when referred to in another statement
		"""

		self.csv_files_path = csv_files_path
		self.csv_line = csv_line
		self.data_access = data_access

	def flatten_dic(self, d, sep=':', property_name_prefix=''):
		"""
		Takes a dictionary specifying SQL to neo4j conversions and returns a string amenable to be used in cypher.

		Parameters
		----------
		d : dict
			A dictionary mapping properties to values.
		sep : str
			Separator between the property and the value
		property_name_prefix : str
			A string to prepend to every property name.

		Returns
		-------
		out: str
			A piece of cypher statement specifying a mapping between properties and values.

		"""

		return ', '.join([f'{property_name_prefix}`{v}` {sep} {self.csv_line}.`{k}`' for k, v in d.items()])

	def load_clause(self, parameters, using_periodic_commit=True) -> str:
		"""
		Returns a clause to load data from a csv file.

		Parameters
		----------
		parameters : dictionary
			Settings
		using_periodic_commit : bool
			Whether to use the "periodic commit" operation mode

		Returns
		-------
		statement: str
			Neo4j statement


		"""

		# full path to the csv file combining `self.csv_files_path` with the file name in the parameters file
		csv_file = str((self.csv_files_path / parameters['intermediate csv file']).resolve())

		# breakpoint()

		# assembly of something along the lines of, e.g.,
		# LOAD CSV WITH HEADERS FROM "file:///researcher_project.csv" AS line
		statement = f"LOAD CSV WITH HEADERS FROM\n'{self.data_access}{csv_file}' AS {self.csv_line}"

		if using_periodic_commit:

			statement = 'USING PERIODIC COMMIT\n' + statement

		return statement

	def match_clause(self, labels, matching_columns, matching_properties, nodes=['origin', 'destination']) -> str:
		"""
		Returns a clause to match data read from the file with that in the database.

		Parameters
		----------
		labels : list
			Labels of the nodes to be matched
		matching_columns : list
			Columns in the csv file
		matching_properties : list
			Properties of the nodes to be matched
		nodes : list
			Names of the nodes to be matched when referred to in another statement

		Returns
		-------
		out: str
			Neo4j statement

		"""

		# assembly of something along the lines of, e.g.,
		# MATCH (origin:researcher {nif: line.NIF})
		# MATCH (destination:project {ref: line.REFERENCIA})
		return '\n'.join([
			f"MATCH ({node}:`{label}` {{`{prop}`: "f"{self.csv_line}.`{col}`}})"
			for label, col, prop, node in zip(labels, matching_columns, matching_properties, nodes)])


class LoadTable(DataLoadingStatement):
	"""
	Class to load a table from a csv file.

	"""

	def assemble(self, parameters: dict) -> typing.Union[str, typing.List[str]]:
		"""
		Returns a statement to load data from a csv file.

		Parameters
		----------
		parameters : dictionary
			Settings

		Returns
		-------
		out: str
			Neo4j statement

		"""

		# a (sub)dictionary of `parameters[SQL columns to neo4j properties]` in which SQL columns not providing a neo4j
		# property (=None) are excluded
		sql_to_neo4j = drop_null_valued_keys(parameters['SQL columns to neo4j properties'])

		# assembly of something along the lines of, e.g.,
		# ref: line.REFERENCIA, coord: line.PCOORDINADOS, centro:line.CENTRO
		properties_statement = self.flatten_dic(sql_to_neo4j)

		# assembly of something along the lines of, e.g.,
		# CREATE (p:Projecto {ref: line.REFERENCIA, coord: line.PCOORDINADOS, centro:line.CENTRO})
		create_statement = f"CREATE (n:{parameters['neo4j label']} {{{properties_statement}}})"

		# the *full* cypher statement is returned
		return f"{self.load_clause(parameters)}\n{create_statement};"


class UpdateTable(DataLoadingStatement):
	"""
	Class to update an existing table using data from a csv file.

	"""

	def assemble(self, parameters: dict) -> typing.Union[str, typing.List[str]]:
		"""
		Returns a statement to update an existing table using data from a csv file.

		Parameters
		----------
		parameters : dictionary
			Settings

		Returns
		-------
		out: str
			Neo4j statement

		"""

		# a (sub)dictionary of `parameters[SQL columns to neo4j properties]` in which SQL columns not providing a neo4j
		# property (=None) are excluded
		sql_to_neo4j = drop_null_valued_keys(parameters['SQL columns to neo4j properties'])

		# the column from the SQL table that is used for merging
		merging_column = parameters['merging column']

		# the latter is excluded from the properties to be added
		merging_property = sql_to_neo4j.pop(merging_column)

		# assembly of something along the lines of, e.g.,
		# name: line.person_name
		merging_property_clause = self.flatten_dic({merging_column: merging_property})

		merge_statement = f"MERGE (n:{parameters['neo4j label']} {{{merging_property_clause}}})"

		# assembly of something along the lines of, e.g.,
		# n.ref = line.REFERENCIA, n.coord = line.PCOORDINADOS, n.centro = line.CENTRO
		properties_clause = self.flatten_dic(sql_to_neo4j, sep='=', property_name_prefix='n.')

		return (
			f'{self.load_clause(parameters)}\n'
			f'{merge_statement}\n'
			# f'ON CREATE SET {properties_clause}\n'
			# f'ON MATCH SET {properties_clause};'
			f'SET {properties_clause};'
		)


class LoadJoinTable(DataLoadingStatement):
	"""
	Class to load data from a csv file representing a "join" table.

	"""

	def assemble(self, parameters: dict) -> typing.Union[str, typing.List[str]]:
		"""
		Returns a statement to load data from a csv file representing a "join" table.

		Parameters
		----------
		parameters : dictionary
			Settings

		Returns
		-------
		out: str
			Neo4j statement

		"""

		# the arguments required by the `match_clause` method
		# i) a list with the labels (one for every match clause)
		# ii) a list with the SQL matching columns (one for every match clause)
		# iii) a list with the neo4j matching properties (one for every match clause)
		labels_columns_properties = [
			[parameters[sec][arg] for sec in ['origin', 'destination']]
			for arg in ['neo4j label', 'SQL matching column', 'neo4j matching property']]

		match_statement = self.match_clause(*labels_columns_properties, nodes=['origin', 'destination'])

		# in order to account for duplicates in the SQL table
		arrowhead = '' if parameters['merge reciprocated relationships'] else '>'

		# assembly of something along the lines of, e.g.,
		# MERGE (r)-[rel:WORKED_ON]->(p)
		merge_statement = merge_relationship_with_properties_clause(
			'origin', 'destination', relationship_type=parameters['relationship type'], relationship='rel',
			arrowhead=arrowhead)

		# a (sub)dictionary of `parameters[SQL columns to neo4j properties]` in which SQL columns not providing a neo4j
		# property (=None) are excluded
		sql_to_neo4j = drop_null_valued_keys(parameters['SQL columns to neo4j properties'])

		# if the dictionary is not empty, i.e., there is some property to be set...
		if sql_to_neo4j:

			# assembly of something along the lines of, e.g.,
			# ON CREATE SET rel.role = row.ROL;
			on_create_statement = f"ON CREATE SET {self.flatten_dic(sql_to_neo4j, sep='=', property_name_prefix='rel.')}"

		# if the dictionary is empty...
		else:

			# ...an empty statement is below harmless
			on_create_statement = ""

		# full cypher statement is returned
		return f"{self.load_clause(parameters)}\n{match_statement}\n{merge_statement}\n{on_create_statement};"


class LoadAttributesTable(DataLoadingStatement):
	"""
	Class to load data from a csv file representing an "attributes" table.

	"""

	def assemble(self, parameters: dict) -> typing.Union[str, typing.List[str]]:
		"""
		Returns a statement to load data from a csv file representing an "attributes" table.

		Parameters
		----------
		parameters : dictionary
			Settings

		Returns
		-------
		out: str
			Neo4j statement

		"""

		match_statement = self.match_clause(
			labels=[parameters['neo4j label']], matching_columns=[parameters['SQL column']],
			matching_properties=[parameters['neo4j property']], nodes=['origin'])

		# a dictionary with the properties to be updated
		properties_to_update = drop_null_valued_keys(parameters['SQL columns to neo4j properties'])

		set_statement = f"SET {self.flatten_dic(properties_to_update, sep='=', property_name_prefix='origin.')}"

		return f"{self.load_clause(parameters)}\n{match_statement}\n{set_statement};"


class MakeRelationship(DataLoadingStatement):
	"""
	Class to load data from a csv file representing a "relationship" table.

	"""

	def assemble(self, parameters: dict) -> typing.Union[str, typing.List[str]]:
		"""
		Returns a statement to load data from a csv file representing a "relationship" table.

		Parameters
		----------
		parameters : dictionary
			Settings

		Returns
		-------
		out: str
			Neo4j statement

		"""

		labels_columns_properties = [
			[parameters[sec][arg] for sec in ['this table', 'related entity']]
			for arg in ['neo4j label', 'SQL matching column', 'neo4j matching property']]

		match_statement = self.match_clause(*labels_columns_properties, nodes=['origin', 'destination'])

		merge_statement = merge_relationship_with_properties_clause(
			'origin', 'destination', relationship_type=parameters['relationship type'], relationship='rel')

		# the *full* cypher statement is returned
		return f"{self.load_clause(parameters)}\n{match_statement}\n{merge_statement};"
