import pathlib
import typing

import pandas as pd

# these are initialized by calling function `initialize` with the appropriate parameters `dict` (you MUST do it)
authors_disambiguator = None
organizations_disambiguator = None


def initialize(parameters: dict) -> None:
	"""
	Initializes this module.

	Parameters
	----------
	parameters : dictionary
		Settings

	"""

	global authors_disambiguator, organizations_disambiguator

	# authors disambiguator is initialized...
	authors_disambiguator = AuthorsDisambiguator(**parameters['authors'])

	# ...and so is the one for organizations
	organizations_disambiguator = OrganizationsDisambiguator(**parameters['organizations'])


def projects_researchers(df: pd.DataFrame):
	"""
	Callback function for "researchers" table in "projects" database.

	Parameters
	----------
	df : Pandas dataframe
		Input data

	Returns
	-------
	df: Pandas dataframe
		Output data

	"""

	# names are lower-cased
	df['NOMBRE'] = df['NOMBRE'].str.lower()

	return df


def publications_authorship(df: pd.DataFrame):
	"""
	Callback function for "authorship" table in "publications" database.

	Parameters
	----------
	df : Pandas dataframe
		Input data

	Returns
	-------
	df: Pandas dataframe
		Output data

	"""

	df['fullname'] = df['initials'] + ' ' + df['surname']

	# names are lower-cased
	df['fullname'] = df['fullname'].str.lower()

	# some nuisance characters are removed
	df['affiliation'] = df['affiliation'].str.replace('\n', ' ', regex=False)
	df['affiliation'] = df['affiliation'].str.replace('\\', ' ', regex=False)

	return df


def patents_literature(df: pd.DataFrame):
	"""
	Callback function for relating patents literature with itself in "patents" database.

	Parameters
	----------
	df : Pandas dataframe
		Input data

	Returns
	-------
	df: Pandas dataframe
		Output data

	"""

	# the rows in which `cited_pat_publn_id` is 0 are filtered out
	return df[df['cited_pat_publn_id'] != 0]


def patents_non_literature(df: pd.DataFrame):
	"""
	Callback function for relating patents literature with non-patents literature in "patents" database.

	Parameters
	----------
	df : Pandas dataframe
		Input data

	Returns
	-------
	df: Pandas dataframe
		Output data

	"""

	# the rows in which `npl_publn_id` is 0 are filtered out
	return df[df['npl_publn_id'] != 0]


def patents_person(df: pd.DataFrame):
	"""
	Callback function for "person" table in "patents" database.

	Parameters
	----------
	df : Pandas dataframe
		Input data

	Returns
	-------
	df: Pandas dataframe
		Output data

	"""

	# names are lower-cased
	df['person_name'] = df['person_name'].str.lower()

	# rows in which the name is `na` are removed...
	res = df.dropna(subset=['person_name'])

	# ...and so are those in which the same field contains an empty string
	return res[res['person_name'] != '']


class Disambiguator:
	"""
	Class to process disambiguation data.
	"""

	def __init__(self, disambiguation_map: typing.List[str], new_id: str) -> None:
		"""
		Initializer.

		Parameters
		----------
		disambiguation_map : list
			Path to the disambiguation map
		new_id : str
			The name of the new field/column/neo4j property to be created for storing the final *disambiguated* id
		"""

		# a `pathlib` object
		self.file = pathlib.Path(*disambiguation_map)

		# the disambiguation map is read from the specified file
		self.map = pd.read_csv(self.file, sep='\s+')

		self.new_id = new_id

		# the *final* mapped-to id is obtained as the concatenation of the (output) "db" and "id" columns...
		self.map[self.new_id] = self.map['std_db'].str.cat(self.map['std_id'], sep='_')

		# ...and those are not needed anymore
		self.map = self.map.drop(['std_db', 'std_id'], axis=1)

		# prefix identifying the "patstats" entries in the disambiguation map
		self.patents_prefix = 'epo'

		# the subset of the dataframe corresponding to "patstats"
		self.patents_subset = self.map[self.map['orig_db'] == self.patents_prefix]

		# the column specifying the database is not needed anymore
		self.patents_subset = self.patents_subset.drop(['orig_db'], axis=1)

		# same for projects
		self.projects_prefix = 'pn'
		self.projects_subset = self.map[self.map['orig_db'] == self.projects_prefix]
		self.projects_subset = self.projects_subset.drop(['orig_db'], axis=1)

		# same for "Scopus"
		self.scopus_prefix = 'scps'
		self.scopus_subset = self.map[self.map['orig_db'] == self.scopus_prefix]
		self.scopus_subset = self.scopus_subset.drop(['orig_db'], axis=1)

	@staticmethod
	def handle_duplicates(df):
		"""
		Get rid of duplicated (disambiguated) ids.

		Parameters
		----------
		df : Pandas dataframe
			Data with duplicates

		Returns
		-------
		unmapped: Pandas dataframe
			Input data with the only row with the value "left_only" in column `source`, or the 1st row of the input
			dataframe if there are several.

		"""

		# a (possibly empty) `DataFrame` containing the unmapped value
		unmapped = df[df['source'] == 'left_only']

		# if all the elements in the group have been mapped...
		if unmapped.empty:

			# the first one (for example) is returned
			return df.head(1)

		else:

			return unmapped

	def merge(self, mapping: pd.DataFrame, df: pd.DataFrame, field: str, prefix: str):
		"""
		Replace within the passed `DataFrame` the values of `field` that are present in the disambiguation map,
		while at the same time ensuring that the new disambiguated values are unique.


		Parameters
		----------
		mapping : Pandas dataframe
			Disambiguation map
		df : Pandas dataframe
			Data to be disambiguated
		field : str
			Column to be used in `df`
		prefix : str
			Prefix to be added to a "new" identifier built from the already existing
			(not present in the disambiguation map)

		Returns
		-------
		merge: Pandas dataframe
			Input data with the disambiguation map applied.

		"""

		# lef-join: only the rows in the data (as opposed to those in the disambiguation map) will be present
		merge = df.merge(mapping, how='left', left_on=field, right_on='orig_id', indicator='source')

		# if the value in new column `new_id` is `NaN`, then replace it with that in `field` with `prefix` prepended
		merge[self.new_id] = merge[self.new_id].where(merge[self.new_id].notna(), prefix + '_' + merge[field])

		# (all) the indexes of the rows which have duplicate values in `new_id`
		i_duplicated = merge.duplicated(subset=self.new_id, keep=False)

		# a sub-`DataFrame` with the rows which do *not* contain any duplicates
		unique = merge[~i_duplicated]

		# the rows which contain duplicates are grouped and *deduplicated* by applying the method `handle_duplicates`
		# NOTE: the index of this `DataFrame` is a `MultiIndex`, but that's not a problem since index is not to be
		#  written to the csv file later on
		deduplicated = merge[i_duplicated].groupby(self.new_id, group_keys=True).apply(self.handle_duplicates)

		# `unique` and `deduplicated` rows are vertically stacked together
		merge = pd.concat((unique, deduplicated))

		# `orig_id` column (=`field` when there is a match, `NaN` otherwise) is not needed anymore
		merge = merge.drop(['orig_id', 'source'], axis=1)

		return merge

	def replace(self, mapping: pd.DataFrame, df: pd.DataFrame, field: str, prefix: str):
		"""
		Replace within the passed `DataFrame` the values of `field` that are present in the disambiguation map.


		Parameters
		----------
		mapping : Pandas dataframe
			Disambiguation map
		df : Pandas dataframe
			Data to be disambiguated
		field : str
			Column to be used in `df`
		prefix : str
			Prefix to be added to a "new" identifier built from the already existing
			(not present in the disambiguation map)

		Returns
		-------
		merge: Pandas dataframe
			Input data with the disambiguation map applied.

		"""

		# TODO: factorize the following code (which is exactly the same as above)

		# lef-join: only the rows in the data (as opposed to those in the disambiguation map) will be present
		merge = df.merge(mapping, how='left', left_on=field, right_on='orig_id', indicator='source')

		# if the value in new column `new_id` is `NaN`, then replace it with that in `field` with `prefix` prepended
		merge[self.new_id] = merge[self.new_id].where(merge[self.new_id].notna(), prefix + '_' + merge[field])

		# `orig_id` column (=`field` when there is a match, `NaN` otherwise) is not needed anymore
		merge = merge.drop(['orig_id', 'source'], axis=1)

		return merge


class AuthorsDisambiguator(Disambiguator):
	"""
	Class containing callbacks methods needed for disambiguating authors.
	"""

	# NOTE: no subclasses are used for the logic below to avoid duplicating the disambiguation mapping in memory

	def patstats_person(self, df: pd.DataFrame):
		"""
		Callback function for "person" table in "patstats" database.

		Parameters
		----------
		df : Pandas dataframe
			Input data

		Returns
		-------
		df: Pandas dataframe
			Output data

		"""

		# `orig_id` in the disambiguation map is a `str` but `person_id` in "PATSTATS" DB is an `int`
		df['person_id'] = df['person_id'].astype('str')

		return self.merge(self.patents_subset, df, 'person_id', self.patents_prefix)

	def patstats_person_application(self, df: pd.DataFrame):
		"""
		Callback function for "application" table in "patstats" database.

		Parameters
		----------
		df : Pandas dataframe
			Input data

		Returns
		-------
		df: Pandas dataframe
			Output data

		"""


		# `orig_id` in the disambiguation map is a `str` but `person_id` in "PATSTATS" DB is an `int`
		df['person_id'] = df['person_id'].astype('str')

		return self.replace(self.patents_subset, df, 'person_id', self.patents_prefix)

	def projects_researchers(self, df: pd.DataFrame):
		"""
		Callback function for "researchers" table in "projects" database.

		Parameters
		----------
		df : Pandas dataframe
			Input data

		Returns
		-------
		df: Pandas dataframe
			Output data

		"""

		return self.merge(self.projects_subset, df, 'NIF', self.projects_prefix)

	def projects_researcher_project(self, df: pd.DataFrame):
		"""
		Callback function for "researcher_project" table in "projects" database.

		Parameters
		----------
		df : Pandas dataframe
			Input data

		Returns
		-------
		df: Pandas dataframe
			Output data

		"""

		return self.replace(self.projects_subset, df, 'NIF', self.projects_prefix)

	def scopus_authorship(self, df: pd.DataFrame):
		"""
		Callback function for "authorship" table in "scopus" database.

		Parameters
		----------
		df : Pandas dataframe
			Input data

		Returns
		-------
		df: Pandas dataframe
			Output data

		"""

		# `orig_id` in the disambiguation map is a `str` but `author_id` in "Scopus" DB is an `int`
		df['author_id'] = df['author_id'].astype('str')

		return self.merge(self.scopus_subset, df, 'author_id', self.scopus_prefix)


# NOTE: this class inherits from `AuthorsDisambiguator` since the methods for "patstats" would be the same
class OrganizationsDisambiguator(AuthorsDisambiguator):
	"""
	Class containing callbacks methods needed for disambiguating organizations.
	"""

	def projects_organizations(self, df: pd.DataFrame):
		"""
		Callback function for "organizations" table in "projects" database.

		Parameters
		----------
		df : Pandas dataframe
			Input data

		Returns
		-------
		df: Pandas dataframe
			Output data

		"""

		return self.merge(self.projects_subset, df, 'CIF', self.projects_prefix)

	def projects_projects(self, df: pd.DataFrame):
		"""
		Callback function for "projects" table in "projects" database.

		Parameters
		----------
		df : Pandas dataframe
			Input data

		Returns
		-------
		df: Pandas dataframe
			Output data

		"""

		return self.replace(self.projects_subset, df, 'CIFEMPRESA', self.projects_prefix)

	def scopus_authorship(self, df: pd.DataFrame):
		"""
		Callback function for "authorship" table in "scopus" database.

		Parameters
		----------
		df : Pandas dataframe
			Input data

		Returns
		-------
		df: Pandas dataframe
			Output data

		"""

		# the id's are converted from `float` (they are `float`because of the missing values) to `str` when they are
		# not `na`
		df['affiliation_id'] = df['affiliation_id'].astype(str).str.rstrip('0.').where(df['affiliation_id'].notna(), 'NA')

		return self.replace(self.scopus_subset, df, 'affiliation_id', self.scopus_prefix)
