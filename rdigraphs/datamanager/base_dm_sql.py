"""
This class provides functionality for managing a generig sqlite or mysql
database:

* reading specific fields (with the possibility to filter by field values)
* storing calculated values in the dataset

Created on May 11 2018

@author: Jerónimo Arenas García

"""

from __future__ import print_function    # For python 2 copmatibility
import os
import pandas as pd
import MySQLdb     # conda install mysqlclient
import sqlite3
import numpy as np
import copy

import pathlib
import typing
import gzip

import tqdm


class BaseDMsql(object):
    """
    Data manager base class.
    """

    def __init__(self, db_name, db_connector, path2db=None,
                 db_server=None, db_user=None, db_password=None,
                 db_port=None, unix_socket=None, charset='utf8mb4'):
        """
        Initializes a DataManager object

        Parameters
        ----------
        db_name : str
            Name of the DB
        db_connector : str {'mysql', 'sqlite'}
            Connector
        path2db : str or None, optional (default=None)
            Path to the project folder (sqlite only)
        db_server : str or None, optional default=None)
            Server (mysql only)
        db_user : str or None, optional (default=None)
            User (mysql only)
        db_password : str or None, optional (default=None)
            Password (mysql only)
        db_port : str or None, optional (default=None)
            Port(mysql via TCP only) Necessary if not 3306
        unix_socket : str or None, optional (default=None)
            Socket for local connectivity. If available, connection is slightly
            faster than through TCP.
        charset : str, optional (default='utf8mb4')
            Coding to use by default in the connection
        """

        # Store paths to the main project folders and files
        self._path2db = copy.copy(path2db)
        self.dbname = db_name
        self.connector = db_connector
        self.server = db_server
        self.user = db_user
        self.password = db_password
        self.port = db_port
        self.unix_socket = unix_socket

        # Other class variables
        self.dbON = False    # Will switch to True when the db is connected.
        # Connector to database
        self._conn = None
        # Cursor of the database
        self._c = None

        # Try connection
        try:
            if self.connector == 'mysql':
                if self.unix_socket:
                    self._conn = MySQLdb.connect(
                        host=self.server, user=self.user, passwd=self.password,
                        db=self.dbname, unix_socket=unix_socket,
                        charset=charset)
                elif self.port:
                    self._conn = MySQLdb.connect(
                        host=self.server, user=self.user, passwd=self.password,
                        db=self.dbname, port=self.port, charset=charset)
                else:
                    self._conn = MySQLdb.connect(
                        host=self.server, user=self.user, passwd=self.password,
                        db=self.dbname, charset=charset)
                self._c = self._conn.cursor()
                print('MySQL database connection successful. Default db:',
                      self.dbname)
                self.dbON = True
                # self._conn.set_character_set('utf8')
            elif self.connector == 'sqlite3':
                # sqlite3
                # sqlite file will be in the root of the project, we read the
                # name from the config file and establish the connection
                db_fname = os.path.join(self._path2db,
                                        self.dbname + '.db')
                print("---- Connecting to {}".format(db_fname))
                self._conn = sqlite3.connect(db_fname)
                self._c = self._conn.cursor()
                self.dbON = True
            else:
                print("---- Unknown DB connector {}".format(self.connector))
        except:
            print("---- Error connecting to the database")

        return

    def __del__(self):
        """
        When destroying the object, it is necessary to commit changes
        in the database and close the connection
        """

        try:
            self._conn.commit()
            self._conn.close()
        except:
            print("---- Error closing database")

    def deleteDBtables(self, tables=None):
        """
        Delete existing database, and regenerate empty tables

        Parameters
        ----------
        tables : str or list or None, optional (default=None)
            If string, name of the table to reset.
            If list, list of tables to reset
            If None, all tables are deleted, and all tables
            (inlcuding those that might not exist previously)
        """

        # If tables is None, all tables are deleted an re-generated
        if tables is None:
            # Delete all existing tables
            for table in self.getTableNames():
                self._c.execute("DROP TABLE " + table)

        else:

            # It tables is not a list, make the appropriate list
            if type(tables) is str:
                tables = [tables]

            # Remove all selected tables (if exist in the database).
            for table in set(tables) & set(self.getTableNames()):
                self._c.execute("DROP TABLE " + table)

        self._conn.commit()

        return

    def addTableColumn(self, tablename, columnname, columntype):
        """
        Add a new column to the specified table.

        Parameters
        ----------
        tablename : str
            Table to which the column will be added
        columnname : str
            Name of new column
        columntype : 
            Type of new column.

        Notes
        -----
        For mysql, if type is TXT or VARCHAR, the character set if forzed to
        be utf8.
        """

        # Check if the table exists
        if tablename in self.getTableNames():

            # Check that the column does not already exist
            if columnname not in self.getColumnNames(tablename):

                #Allow columnames with spaces
                columnname = '`'+columnname+'`'

                # Fit characters to the allowed format if necessary
                fmt = ''
                if (self.connector == 'mysql' and
                    ('TEXT' in columntype or 'VARCHAR' in columntype) and
                    not ('CHARACTER SET' in columntype or
                         'utf8' in columntype)):

                    # We need to enforze utf8 for mysql
                    fmt = ' CHARACTER SET utf8'

                sqlcmd = ('ALTER TABLE ' + tablename + ' ADD COLUMN ' +
                          columnname + ' ' + columntype + fmt)
                self._c.execute(sqlcmd)

                # Commit changes
                self._conn.commit()

            else:
                print(("WARNING: Column {0} already exists in table {1}."
                       ).format(columnname, tablename))

        else:
            print('Error adding column to table. Please, select a valid ' +
                  'table name from the list')
            print(self.getTableNames())

    def dropTableColumn(self, tablename, columnname):
        """
        Remove column from the specified table

        Parameters
        ----------
        tablename : str
            Table containing the column to be removed
        columnname : str
            Name of column to be removed
        """

        # Check if the table exists
        if tablename in self.getTableNames():

            # Check that the column exists
            if columnname in self.getColumnNames(tablename):

                #Allow columnames with spaces
                columname = '`'+columnname+'`'

                # ALTER TABLE DROP COLUMN IS ONLY SUPPORTED IN MYSQL
                if self.connector == 'mysql':

                    sqlcmd = ('ALTER TABLE ' + tablename + ' DROP COLUMN ' +
                              columnname)
                    self._c.execute(sqlcmd)

                    # Commit changes
                    self._conn.commit()

                else:
                    print('Error deleting column. Column drop not yet supported for SQLITE')

            else:
                print('Error deleting column. The column does not exist')
                print(tablename, columnname)

        else:
            print('Error deleting column. Please, select a valid table name' +
                  ' from the list')
            print(self.getTableNames())

        return

    def readDBtable(self, tablename, limit=None, selectOptions=None,
                    filterOptions=None, orderOptions=None):
        """
        Read data from a table in the database can choose to read only some
        specific fields

        Parameters
        ----------
        tablename : str
            Table to be read from
        limit : int or None, optional (default=None)
            The maximum number of records to retrieve
        selectOptions : str or None, optional (default=None)
            string with fields that will be retrieved
            (e.g. 'REFERENCIA, Resumen')
        filterOptions : str or None, optional (default=None)
            Filtering options for the SQL query
            (e.g., 'WHERE UNESCO_cd=23')
        orderOptions: str or None, optional (default=None)
            Field that will be used for sorting the
            results of the query (e.g, 'Cconv')
        """

        try:

            # Check that table name is valid

            if tablename in self.getTableNames():

                sqlQuery = 'SELECT '
                if selectOptions:
                    sqlQuery = sqlQuery + selectOptions
                else:
                    sqlQuery = sqlQuery + '*'

                sqlQuery = sqlQuery + ' FROM ' + tablename + ' '

                if filterOptions:
                    sqlQuery = sqlQuery + ' WHERE ' + filterOptions

                if orderOptions:
                    sqlQuery = sqlQuery + ' ORDER BY ' + orderOptions

                if limit:
                    sqlQuery = sqlQuery + ' LIMIT ' + str(limit)

                # This is to update the connection to changes by other
                # processes.
                self._conn.commit()

                # Return the pandas dataframe. Note that numbers in text format
                # are not converted to
                return pd.read_sql(sqlQuery, con=self._conn,
                                   coerce_float=False)

            else:
                print('Error in query. Please, select a valid table name ' +
                      'from the list')
                print(self.getTableNames())

        except Exception as E:
            print(str(E))

    def getTableNames(self):
        """
        Provides acces to table names

        Returns
        -------
        tbnames : list
            Names of all tables in the database
        """

        # The specific command depends on whether we are using mysql or sqlite
        if self.connector == 'mysql':
            sqlcmd = ("SELECT table_name FROM INFORMATION_SCHEMA.TABLES " +
                      "WHERE table_schema='" + self.dbname + "'")
        else:
            sqlcmd = "SELECT name FROM sqlite_master WHERE type='table'"

        self._c.execute(sqlcmd)
        tbnames = [el[0] for el in self._c.fetchall()]

        return tbnames

    def getColumnNames(self, tablename):
        """
        Returns a list with the names of all columns in the indicated table

        Parameters
        ----------
        tablename : str
            Table to be read from

        Returns
        -------
        columnames : list
            Names of all columns in the selected table
        """

        # Check if tablename exists in database
        if tablename in self.getTableNames():
            # The specific command depends on whether we are using mysql or
            #  sqlite
            if self.connector == 'mysql':
                sqlcmd = "SHOW COLUMNS FROM " + tablename
                self._c.execute(sqlcmd)
                columnnames = [el[0] for el in self._c.fetchall()]
            else:
                sqlcmd = "PRAGMA table_info(" + tablename + ")"
                self._c.execute(sqlcmd)
                columnnames = [el[1] for el in self._c.fetchall()]

            return columnnames

        else:
            print('Error retrieving column names: Table does not exist on ' +
                  'database')
            return []

    def getTableInfo(self, tablename):
        """
        Get information about the given table (size and columns)

        Parameters
        ----------
        tablename : str
            Table to be read from        

        Returns
        -------
        cols : list
            Names of all columns in the table
        n_rows : int
            Number of rows in table
        """

        # Get columns
        cols = self.getColumnNames(tablename)

        # Get number of rows
        sqlcmd = "SELECT COUNT(*) FROM " + tablename
        self._c.execute(sqlcmd)
        n_rows = self._c.fetchall()[0][0]

        return cols, n_rows

    def insertInTable(self, tablename, columns, arguments):
        """
        Insert new records into table

        Parameters
        ----------
        tablename : str
            Name of table in which the data will be inserted
        columns : list
            Name of columns for which data are provided
        arguments : list of list of tuples
            A list of lists of tuples, each element associated to one new entry
            for the table
        """

        # Make sure columns is a list, and not a single string
        if not isinstance(columns, (list,)):
            columns = [columns]

        # To allow for column names that have spaces
        columns = list(map(lambda x: '`'+x+'`', columns))

        ncol = len(columns)

        if len(arguments[0]) == ncol:
            # Make sure the tablename is valid
            if tablename in self.getTableNames():
                # Make sure we have a list of tuples; necessary for mysql
                arguments = list(map(tuple, arguments))

                # # Update DB entries one by one.
                # for arg in arguments:
                #     # sd
                #     sqlcmd = ('INSERT INTO ' + tablename + '(' +
                #               ','.join(columns) + ') VALUES(' +
                #               ','.join('{}'.format(a) for a in arg) + ')'
                #               )

                #     try:
                #         self._c.execute(sqlcmd)
                #     except:
                #         import ipdb
                #         ipdb.set_trace()

                sqlcmd = ('INSERT INTO ' + tablename +
                          '(' + ','.join(columns) + ') VALUES (')
                if self.connector == 'mysql':
                    sqlcmd += '%s' + (ncol-1)*',%s' + ')'
                else:
                    sqlcmd += '?' + (ncol-1)*',?' + ')'

                self._c.executemany(sqlcmd, arguments)

                # Commit changes
                self._conn.commit()
        else:
            print('Error inserting data in table: number of columns mismatch')

        return

    def setField(self, tablename, keyfld, valueflds, values):
        """
        Update records of a DB table

        Parameters
        ----------
        tablename : str
            Table that will be modified
        keyfld : str
            Name of the column that will be used as key (e.g. 'REFERENCIA')
        valueflds : list
            Names of the columns that will be updated (e.g., 'Lemas')
        values : list of tuples
            A list of tuples in the format (keyfldvalue, valuefldvalue)
            (e.g., [('Ref1', 'gen celula'), ('Ref2', 'big_data, algorithm')])
        """

        # Auxiliary function to circularly shift a tuple one position to the
        # left
        def circ_left_shift(tup):
            ls = list(tup[1:]) + [tup[0]]
            return tuple(ls)

        # Make sure valueflds is a list, and not a single string
        if not isinstance(valueflds, (list,)):
            valueflds = [valueflds]

        # To allow for column names that have spaces
        valueflds = list(map(lambda x: '`'+x+'`', valueflds))

        ncol = len(valueflds)

        if len(values[0]) == (ncol+1):
            # Make sure the tablename is valid
            if tablename in self.getTableNames():

                # # Update DB entries one by one.
                # # WARNING: THIS VERSION MAY NOT WORK PROPERLY IF v
                # #          HAS A STRING CONTAINING "".
                # for v in values:
                #     sqlcmd = ('UPDATE ' + tablename + ' SET ' +
                #               ', '.join(['{0} ="{1}"'.format(f, v[i + 1])
                #                          for i, f in enumerate(valueflds)]) +
                #               ' WHERE {0}="{1}"'.format(keyfld, v[0]))
                #     self._c.execute(sqlcmd)

                # This is the old version: it might not have the problem of
                # the above version, but did not work properly with sqlite.
                # Make sure we have a list of tuples; necessary for mysql
                # Put key value last in the tuples
                values = list(map(circ_left_shift, values))

                sqlcmd = 'UPDATE ' + tablename + ' SET '
                if self.connector == 'mysql':
                    sqlcmd += ', '.join([el+'=%s' for el in valueflds])
                    sqlcmd += ' WHERE ' + keyfld + '=%s'
                else:
                    sqlcmd += ', '.join([el+'=?' for el in valueflds])
                    sqlcmd += ' WHERE ' + keyfld + '=?'

                self._c.executemany(sqlcmd, values)

                # Commit changes
                self._conn.commit()
        else:
            print('Error updating table values: number of columns mismatch')

        return

    def upsert(self, tablename, keyfld, df, robust=True):
        """
        Update records of a DB table with the values in the df
        This function implements the following additional functionality:
        * If there are columns in df that are not in the SQL table, columns will be added
        * New records will be created in the table if there are rows in the dataframe without
        an entry already in the table. For this, keyfld indicates which is the column that will be used as an index

        Parameters
        ----------
        tablename : str
            Table that will be modified
        keyfld : str
            Name of the column that will be used as key (e.g. 'REFERENCIA')
        df : dataframe
            Dataframe that we wish to save in table tablename
        robust : bool, optional (default=True)
            If False, verifications are skipped (for a faster execution)
        """

        # Check that table exists and keyfld exists both in the Table and the
        # Dataframe
        if robust:
            if tablename in self.getTableNames():
                if not ((keyfld in df.columns) and
                   (keyfld in self.getColumnNames(tablename))):
                    print("Upsert function failed: Key field does not exist",
                          "in the selected table and/or dataframe")
                    return
            else:
                print('Upsert function failed: Table does not exist')
                return

        # Reorder dataframe to make sure that the key field goes first
        flds = [keyfld] + [x for x in df.columns if x != keyfld]
        df = df[flds]

        if robust:
            # Create new columns if necessary
            for clname in df.columns:
                if clname not in self.getColumnNames(tablename):
                    if df[clname].dtypes == np.float64:
                        self.addTableColumn(tablename, clname, 'DOUBLE')
                    else:
                        if df[clname].dtypes == np.int64:
                            self.addTableColumn(tablename, clname, 'INTEGER')
                        else:
                            self.addTableColumn(tablename, clname, 'TEXT')

        # Check which values are already in the table, and split
        # the dataframe into records that need to be updated, and
        # records that need to be inserted
        keyintable = self.readDBtable(tablename, limit=None,
                                      selectOptions=keyfld)
        keyintable = keyintable[keyfld].tolist()
        values = [tuple(x) for x in df.values]
        values_insert = list(filter(lambda x: x[0] not in keyintable, values))
        values_update = list(filter(lambda x: x[0] in keyintable, values))

        if len(values_update):
            self.setField(tablename, keyfld, df.columns[1:].tolist(),
                          values_update)
        if len(values_insert):
            self.insertInTable(tablename, df.columns.tolist(), values_insert)

        return

    def exportTable(self, tablename, fileformat, path, filename, cols=None):
        """
        Export columns from a table to a file.

        Parameters
        ----------
        tablename : str
            Name of the table
        fileformat : str {'xlsx', 'pkl'}
            Type of output file
        path : str
            Route to the output folder
        filename : str
            Name of the output file
        cols : list or str
            Columns to save. It can be a list or a string of comma-separated
            columns.
            If None, all columns saved.
        """

        # Path to the output file
        fpath = os.path.join(path, filename)

        # Read data:
        if cols is list:
            options = ','.join(cols)
        else:
            options = cols

        df = self.readDBtable(tablename, selectOptions=options)

        # ######################
        # Export results to file
        if fileformat == 'pkl':
            df.to_pickle(fpath)

        else:
            df.to_excel(fpath)

        return

    def export_table_to_csv(
            self, table_name: str,
            output_file: typing.Union[str, pathlib.Path],
            block_size: typing.Optional[int] = None, 
            max_rows: typing.Optional[int] = None, gzipped: bool = True,
            callbacks: typing.Optional[typing.List[
                typing.Callable[[pd.DataFrame], pd.DataFrame]]] = None,
            select_options: typing.Optional[str] = None,
            filter_options: typing.Optional[str] = None,
            order_options: typing.Optional[str] = None):
        """
        Exports a table to csv.

        Parameters
        ----------
        table_name : str
            Name of the SQL table.
        output_file : str
            Name of the output csv file.
        block_size : int, optional
            Table is read blockwise (to avoid running out of memory) in batches
            of this size.
        max_rows : int, optional
            Maximum number of rows to be read. If `None`, the whole table is
            read.
        gzipped : bool, optional (default=T)
            Whether to write a gzipped csv file (as opposed to plain text).
        callbacks : list or None, optional (default=None)
            List of callable receiving a `DataFrame` and returning another one
            with the same structure
            A list of functions to be called on every block read before
            actually writing to disk.
        select_options : str or None, optional (default=None)
            "select" options to be passed `readDBtable`
        filter_options : str or None, optional (default=None)
            "filter" options to be passed `readDBtable`
        order_options : str or None, optional (default=None)
            "order" options to be passed `readDBtable`

        Returns
        -------

        """

        # `select_options` is "rewritten" enclosing every column name in quotes
        # (to avoid issues with spaces)
        select_options = ','.join(
            ['`' + s.strip() + '`' for s in select_options.split(',')])

        # the received `output_file` is turned into a Path object (if it's not
        # already one)
        output_file = pathlib.Path(output_file)

        # if the file already exists...
        if output_file.exists():

            # ...it is deleted (this is necessary to get the usual behaviour
            # "overwrite the file if it exists"  because the file *must* be
            # opened in "append" mode below)
            output_file.unlink()

        # FIXME: this call is slow for *some* (???) databases
        # the overall number of rows in the table
        n_rows_overall = self.getTableInfo(table_name)[1]

        # if a maximum number of rows has been passed...
        if max_rows:

            # ...it replaces the actual overall number of rows
            n_rows_overall = max_rows

        # if a block size has not been passed...
        if not block_size:

            # ...everything is read at once
            block_size = n_rows_overall

        # if a compressed file is requested....
        if gzipped:

            # the file is opened in *append* mode
            f = gzip.open(output_file, 'at')

        # otherwise, a regular text file is assumed
        else:
            # the file is opened in *append* mode
            f = open(output_file, 'a')

        # the first row requested to the SQL database
        offset = 0

        # for initialization purposes, we assume the requested number of rows
        # was read in the previous (non-existent) iteration
        n_rows_read = block_size

        # the loop is wrapped in an object that will show a progress bar
        with tqdm.tqdm(total=n_rows_overall) as progress_bar:

            # as long as the DB returns as many rows as requested...
            while (n_rows_read == block_size) and (offset < n_rows_overall):

                # if a maximum number of rows has been passed...
                if max_rows:

                    # ...and the *remaining number of rows* to read is smaller
                    # than the block size, the former becomes the block size
                    block_size = min(block_size, n_rows_overall - offset)

                # this part of the SQL query specifies which rows are to be
                # read from the DB
                limit_clause = f'{offset}, {block_size}'

                # running the SQL query is delegated to `readDBtable`
                res_df = self.readDBtable(
                    table_name, limit=limit_clause,
                    selectOptions=select_options, filterOptions=filter_options,
                    orderOptions=order_options)

                # if a `callbacks` *list* has been passed...
                if callbacks:

                    for c in callbacks:

                        # ...the read `DataFrame` is transformed by means of them
                        res_df = c(res_df)

                # the number of rows read is inferred from the size of the `DataFrame`
                n_rows_read = len(res_df)

                # if this is the first batch...
                if offset == 0:

                    # ...the dataframe is saved along *with the header*
                    res_df.to_csv(f, index=False)

                # otherwise...
                else:

                    # ...the dataframe is saved *excluding the header*
                    res_df.to_csv(f, index=False, header=False)

                # the offset (starting row) for the next iteration is advanced
                offset += n_rows_read

                # the progress bar is updated
                progress_bar.update(n_rows_read)

        # file is closed
        f.close()
