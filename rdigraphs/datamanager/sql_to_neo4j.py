#! /usr/bin/env python3
"""
It migrates a SQL database to Neo4j as specified in the given parameters file. Another "global" parameters file is
required for database connections (both SQL and Neo4j) credentials.
"""

import pathlib
import argparse
import sys
import operator

from neo4j import GraphDatabase
import yaml
import colorama

import cypher
import base_dm_sql
import callbacks

# color is reset at the end of every print
colorama.init(autoreset=True)

# color alias for the sake of brevity
color_info = colorama.Fore.LIGHTWHITE_EX
color_error = colorama.Fore.LIGHTRED_EX
color_caveat = colorama.Fore.LIGHTGREEN_EX
color_table = colorama.Fore.LIGHTBLUE_EX
color_relationship = colorama.Fore.LIGHTGREEN_EX
color_reset = colorama.Style.RESET_ALL

# ------------ command line arguments

parser = argparse.ArgumentParser(description='SQL to neo4j')

parser.add_argument(
	'-p', '--parameters_file', type=argparse.FileType('r'), default='parameters.yaml', help='migration parameters file')

# NOTE: the *global* parameters file is by default searched for in the parent directory and named "parameters.yaml"
parser.add_argument(
	'-g', '--global_parameters_file', type=argparse.FileType('r'),
	default=str(pathlib.Path.cwd().parent.parent / 'parameters.yaml'), help='global (project-level) parameters file')

parser.add_argument(
	'-d', '--dry-run', default=False, const=True, action='store_const', help='do not actually write anything to neo4j')

parser.add_argument(
	'-n', '--no-maximum-number-of-rows', default=False, const=True, action='store_const',
	help='ignore the specified `maximum number of rows` in the parameters files, i.e., migrate everything')

command_line_arguments = parser.parse_args(sys.argv[1:])

# ------------ parameters

# the *global* parameters file is read...
with open(command_line_arguments.global_parameters_file.name) as f:
	global_parameters = yaml.load(f)

# ...and also that with the *migration* parameters
with open(command_line_arguments.parameters_file.name) as f:
	parameters = yaml.load(f)

# connections setting for *neo4j* DB access
neo4j_db_server = global_parameters['connections']['neo4j']['server']
neo4j_db_user = global_parameters['connections']['neo4j']['user']
neo4j_db_password = global_parameters['connections']['neo4j']['password']

# ------------ neo4j

# if this is not a dry run (simulation)...
if not command_line_arguments.dry_run:

	# object to handle the connection to the neo4j DB
	neo4j_driver = GraphDatabase.driver(neo4j_db_server, auth=(neo4j_db_user, neo4j_db_password))

# a file to store every cypher statement sent to neo4j
cypher_log_filename = parameters['log']['cypher file']

# if none was given...
if not cypher_log_filename:

	# ...it will be the name of the parameters file with the *yaml* extension replaced with *cypher*
	cypher_log_filename = pathlib.Path(command_line_arguments.parameters_file.name).with_suffix(
		'.cypher').absolute()

# log file registering every cypher statement
cypher_log_file = open(cypher_log_filename, 'w')

# ------------

# `callbacks` module is initialized
callbacks.initialize(parameters['disambiguation'])

# for every database to be migrated...
for db_parameters in parameters['databases']:

	# the name of the current (SQL) database
	sql_db_name = db_parameters['input']['database name']

	print(f"{'-' * 30} {sql_db_name} {'-' * 30}")

	# connection settings for *SQL* DB access
	sql_db_server = global_parameters['connections']['SQL']['databases'][sql_db_name]['server']
	sql_db_user = global_parameters['connections']['SQL']['databases'][sql_db_name]['user']
	sql_db_password = global_parameters['connections']['SQL']['databases'][sql_db_name]['password']

	# directory where to write/read auxiliar csv files
	auxiliar_files_dir_name = db_parameters['output']["auxiliar csv files' directory"]

	# ------------ SQL

	# (Jero's) object for SQL DB access
	sql_db = base_dm_sql.BaseDMsql(
		db_name=sql_db_name, db_connector='mysql', path2db=None, db_server=sql_db_server,
		db_user=sql_db_user, db_password=sql_db_password)

	# ------------ output

	# if a directory for auxiliar files is passed...
	if auxiliar_files_dir_name:
		# ...an *object* for it is created
		auxiliar_files_dir = pathlib.Path(auxiliar_files_dir_name)

	# if no directory for auxiliar files is passed...
	else:
		# ...an *object* for the current working directory is created
		auxiliar_files_dir = pathlib.Path.cwd()

	# the directory is created if it doesn't exist
	auxiliar_files_dir.mkdir(exist_ok=True)

	# ------------ cypher statements

	cypher_load_table = cypher.LoadTable(auxiliar_files_dir)
	cypher_update_table = cypher.UpdateTable(auxiliar_files_dir)
	cypher_load_join_table = cypher.LoadJoinTable(auxiliar_files_dir)
	cypher_load_attributes_table = cypher.LoadAttributesTable(auxiliar_files_dir)
	cypher_make_relationship = cypher.MakeRelationship(auxiliar_files_dir)

	# ------------ index and constraints creation
	# NOTE: it is a good thing to create the indexes beforehand to avoid "schema await" after the tables' creation
	# (is it possible to do it "programmatically"?); indexing attributes involved in relationships avoids really
	# long times when creating the latter

	# for every table that must be processed...
	for table_parameters in db_parameters['tables']:

		# if a list of `unique properties` is given, AND is not empty (!=None)
		if ('unique properties' in table_parameters) and table_parameters['unique properties']:

			print(f"{color_info}enforcing UNIQUE constraints on table {color_table}{table_parameters['SQL table']}")

			for c in table_parameters['unique properties']:

				cypher_statement = cypher.create_unique_constraint(table_parameters['neo4j label'], c)

				# if this is not a dry run (simulation)...
				if not command_line_arguments.dry_run:

					with neo4j_driver.session() as session:

						# ...the cypher statement is run
						session.run(cypher_statement)

				# in any case, the statement is added to the log file
				cypher_log_file.write(cypher_statement + '\n// ----\n')

		# if an `indexes` list is given, AND is not empty (!=None)
		if ('indexes' in table_parameters) and table_parameters['indexes']:

			print(f"{color_info}building index(es) for table {color_table}{table_parameters['SQL table']}")

			cypher_statements = cypher.create_indexes(table_parameters['neo4j label'], table_parameters['indexes'])

			# if this is not a dry run (simulation)...
			if not command_line_arguments.dry_run:

				with neo4j_driver.session() as session:

					for s in cypher_statements:

						# a cypher statement to build an index is run
						session.run(s)

			# if this IS a dry run...
			else:

				# ...we still loop through all the statements...
				for s in cypher_statements:

					# ...to add them to the log file
					cypher_log_file.write(s + '\n// ----\n')

	# ------------ nodes creation

	# for every table that must be processed...
	for table_parameters in db_parameters['tables']:

		# intermediate output files
		csv_filename = table_parameters['intermediate csv file']

		# a `Path` object built from the corresponding file name
		csv_file = auxiliar_files_dir / pathlib.Path(csv_filename)

		# if the csv file has not been previously generated...
		# NOTE: if the file already exists, it is assumed that it has the "right" content (no checks)
		if not csv_file.exists():

			print(f"{color_caveat}writing csv file for table {color_table}{table_parameters['SQL table']}")

			# a `dict_keys` (as opposed to a `list`) object with the "relevant" columns
			sql_table_columns = table_parameters['SQL columns to neo4j properties'].keys()

			# if "no maximum number of rows" is requested...
			if command_line_arguments.no_maximum_number_of_rows:
				# ...the corresponding argument will be set to `None` regardless of the actual value (it's overridden)
				max_rows = None
			# otherwise...
			else:
				# ...the argument will be whatever it is given in the parameters file
				max_rows = table_parameters['maximum number of rows']

			# if a `callbacks` list has been specified and is not `None`...
			if ('callbacks' in table_parameters) and table_parameters['callbacks']:

				callbacks_list = []

				for c in table_parameters['callbacks']['functions']:

					try:

						callbacks_list.append(operator.attrgetter(c)(callbacks))

					except AttributeError:

						print(f"{color_error}can't find callback {color_reset}{c}")

						raise SystemExit

				# columns in the `DataFrame` that are created by the callback(s) are not actually in the SQL table (but
				# are added when `export_table_to_csv` calls the `callback` method) , and hence shouldn't be requested
				sql_table_columns = [
					c for c in sql_table_columns if c not in table_parameters['callbacks']['added columns']]

			else:

				# `callbacks` parameter is passed anyway, below
				callbacks_list = None

			# the requested columns are read and written to a csv file
			sql_db.export_table_to_csv(
				table_parameters['SQL table'], csv_file, block_size=table_parameters['block size'], max_rows=max_rows,
				select_options=','.join(sql_table_columns), callbacks=callbacks_list)

		print(f"{color_info}processing table {color_table}{table_parameters['SQL table']}")

		# if this is a join table (a couple of foreign keys plus extra attributes)...
		if table_parameters['type'] == 'join':

			cypher_statement = cypher_load_join_table.assemble(table_parameters)

		# if it is an "attributes" table (meant to add attributes to already existing rows -neo4j nodes-)
		elif table_parameters['type'] == 'attributes':

			cypher_statement = cypher_load_attributes_table.assemble(table_parameters)

		# if it is a good ol' classic boring SQL table...
		elif table_parameters['type'] == 'vanilla':

			cypher_statement = cypher_load_table.assemble(table_parameters)

		# if it is a (good ol') classic (boring) SQL table meant to update a previous one...
		elif table_parameters['type'] == 'update':

			cypher_statement = cypher_update_table.assemble(table_parameters)

		# otherwise...
		else:

			raise Exception(f'"{table_parameters["type"]}" is not a valid table type!!')

		# if this is not a dry run (simulation)...
		if not command_line_arguments.dry_run:

			# session is accessed...
			with neo4j_driver.session() as session:

				# ...to run the above statement
				session.run(cypher_statement)

		# the statement is added to the log file
		cypher_log_file.write(cypher_statement + '\n// ----\n')

	# ------------ relationships creation

	# if there is any relationship specified in the parameters file...
	if 'relationships' in db_parameters:

		# for every relationship that must be processed...
		for relationship_parameters in db_parameters['relationships']:

			print(f"{color_info}making relationship {color_relationship}{relationship_parameters['relationship type']}"
									f"{color_info} for table {color_table}{relationship_parameters['SQL table']}")

			cypher_statement = cypher_make_relationship.assemble(relationship_parameters)

			# if this is not a dry run (simulation)...
			if not command_line_arguments.dry_run:

				# session is accessed...
				with neo4j_driver.session() as session:

					# ...to run the above statement
					session.run(cypher_statement)

			# the statement is added to the log file
			cypher_log_file.write(cypher_statement + '\n// ----\n')

	# a comment in the log file signaling the end of the statements for this DB
	cypher_log_file.write(f'// {"=" * 80}\n')

# log file is closed
cypher_log_file.close()

# if this is not a dry run (simulation)...
if not command_line_arguments.dry_run:

	# in order to play it safe
	neo4j_driver.close()
