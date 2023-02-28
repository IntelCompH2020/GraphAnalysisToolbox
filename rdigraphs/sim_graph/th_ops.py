#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Python libraries
import numpy as np
import logging
import sys
import csv
import os
import random

from time import time
from scipy.sparse import csr_matrix, issparse
import numba

from typing import List, Tuple, Union

try:
    import cupy.sparse
    # import GPUtil

    memory_pool = cupy.cuda.MemoryPool()
    cupy.cuda.set_allocator(memory_pool.malloc)
    pinned_memory_pool = cupy.cuda.PinnedMemoryPool()
    cupy.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)

except ModuleNotFoundError:
    print("-- GPU options disabled. If a GPU is available, install cupy "
          "library to enable GPU options")

except Exception:
    print("-- Error during GPU setting with cupy.cuda. "
          "GPU options will be disabled")


def vprint(msg, verbose=True):
    """
    If verbose is True, prints a status message. A time stamp is returned
    anyway.

    Parameters
    ----------
    verbose : boolean
        True if the msg must be printed
    msg : str
        Message to be printed
    """
    if verbose:
        print(f".. .. .. {msg}")
    return time()


def size_of_sparse(a: csr_matrix) -> float:
    """
    Compute size of the sparse matrix, in Megabytes.

    Parameters
    ----------
    a : csr_matrix
        Sparse matrix

    Returns
    -------
    s : float
        Size
    """

    return (a.data.nbytes + a.indptr.nbytes + a.indices.nbytes) / 1024**2


# #############################################################################
# The following method is outside the ThOps class because it uses Numba, and it
# would not work inside the class.
# #############################################################################
@numba.jit
def edges_weights_from_sparseU(
        data: np.ndarray, indptr: np.ndarray, indices: np.ndarray, n_cols: int,
        i_block: int, j_block: int) -> Tuple[list, list]:
    """
    Computes edge weight and absolute coordinates from a subblock of the Upper
    triangular component of a sparse matrix

    Parameters
    ----------
    data : np.ndarray
        Data
    indptr : np.ndarray
        Indices from the sparse matrix
    indices : np.ndarray
        Second list of indices
    n_cols : int
        Number of columns
    i_block : int
        Absolute i coordinate of the relative 0 coordinate
    j_block: int
        Absolute j coordinate of the relative 0 coordinate

    Returns
    -------
    absolute_coordinates : list
        List of edges
    values : list
        List of weights
    """

    # The number of rows can be inferred from the rows' indexes
    n_rows = len(indptr) - 1

    # The operation is different depending on whether the block is diagonal
    # or not
    diagonal = i_block == j_block

    # relative_coordinates = [1.0]

    # A list to accumulate the (absolute) *coordinates* of the array in the
    # non-zero positions
    absolute_coordinates = []

    # a list to accumulate the *values* in the above coordinates
    values = []

    # if the passed sparse array is the product of a matrix with itself...
    if diagonal:

        # for every row...
        for i_row in range(n_rows):
            # for every non-zero index and column...
            for i_col, d in zip(indices[indptr[i_row]:indptr[i_row + 1]],
                                data[indptr[i_row]:indptr[i_row + 1]]):

                if i_row < i_col and d != 0:
                    # The coordinates are recorded...
                    # relative_coordinates.append((i_row, i_col))
                    # For some reason, an error arise in cuda in int() is
                    # not used below
                    absolute_coordinates.append(
                        (i_block + i_row, j_block + i_col))
                    # ...and so is the corresponding value of the array at
                    # that position
                    values.append(d)

    else:

        for i_row in range(n_rows):
            for i_col, d in zip(indices[indptr[i_row]:indptr[i_row + 1]],
                                data[indptr[i_row]:indptr[i_row + 1]]):

                if d != 0:
                    # the coordinates are recorded...
                    # relative_coordinates.append((i_row, i_col))
                    absolute_coordinates.append(
                        (i_block + i_row, j_block + i_col))

                    # ...and so is the corresponding value of the array at
                    # that position
                    values.append(d)

    return absolute_coordinates, values


@numba.jit
def edges_weights_from_sparseLDU(
        data: np.ndarray, indptr: np.ndarray, indices: np.ndarray, n_cols: int,
        i_block: int, j_block: int) -> Tuple[list, list]:
    """
    Computes edge weight and absolute coordinates from a subblock of a sparse
    matrix

    Parameters
    ----------
    data : np.ndarray
        Data
    indptr : np.ndarray
        Indices from the sparse matrix
    indices : np.ndarray
        Second list of indices
    n_cols : int
        Number of columns
    i_block : int
        Absolute i coordinate of the relative 0 coordinate
    j_block: int
        Absolute j coordinate of the relative 0 coordinate

    Returns
    -------
    absolute_coordinates : list
        List of edges
    values : list
        List of weights
    """

    # The number of rows can be inferred from the rows' indexes
    n_rows = len(indptr) - 1

    # A list to accumulate the absolute coordinates of the non-zero array
    # components
    absolute_coordinates = []

    # a list to accumulate the *values* in the above coordinates
    values = []

    for i_row in range(n_rows):
        for i_col, d in zip(indices[indptr[i_row]:indptr[i_row + 1]],
                            data[indptr[i_row]:indptr[i_row + 1]]):

            if d != 0:
                # the coordinates are recorded...
                # relative_coordinates.append((i_row, i_col))
                absolute_coordinates.append(
                    (i_block + i_row, j_block + i_col))
                # ...and so is the corresponding value of the array at
                # that position
                values.append(d)

    return absolute_coordinates, values


class ThOps(object):
    """
    Generic class for the efficient computation of thresholded operation with
    dense and sparse matrices.
    """

    def __init__(self, blocksize=25_000, useGPU=False, tmp_folder=None,
                 save_every=1e300):
        """
        Stores the main attributes of a datagraph object and loads the graph
        data as a list of node attributes from a database

        Parameters
        ----------
        X : scipy.sparse.csr or numpy.array
            Matrix of node attribute vectors
        blocksize : int, optional (default=25_000)
            Size (number of rows) of each block in blocwise processing.
        useGPU : bool, optional (default=False)
            If True, matrix operations are accelerated using GPU
        tmp_folder : str or None, optional (defautl = None)
            Name of the folder to save temporary files
        save_every : int, optional (default=1e300)
            Maximum size of the growing lists. The output lists are constructed
            incrementally. To avoid memory overload, growing lists are saved
            every time they reach this size limit. The full lists are thus
            incrementally saved in files.
            The default value is extremely large, which de facto implies no
            temporary saving.
        """

        # ################
        # Object variables

        # Size of blocks taken from the data matrix for blockwise processing
        # "int" is just to make sure that blocksize is an integer, to avoid an
        # execution error when used as an array index
        self.blocksize = int(blocksize)

        # If True, gpu will be used...
        self.useGPU = useGPU

        # Length of edge and value lists that will activate temporary saving
        self.save_every = save_every

        # #######################
        # Setting temporary files

        # Note that the names of these temporary files has a random suffix to
        # avoid colissions with other files being used for other processes
        random.seed(time())
        suff = random.randint(0, 1e10)
        if tmp_folder is None:
            # Default temporary folder
            self.tmp_folder = os.path.join('.', 'tmp')
        else:
            self.tmp_folder = tmp_folder

        # Path to the files storing (temporarily) edges and weights
        self.fpath_edges = os.path.join(self.tmp_folder, f'edges_{suff}.csv')
        self.fpath_values = os.path.join(self.tmp_folder, f'values_{suff}.csv')

        # This prevents some occasional but large debug messages from numba
        logging.getLogger('numba').setLevel(logging.WARNING)

        return

    def _move_to_file(self, edge_list, thp=[], mode='w'):
        """
        Write or append input lists in to files

        Parameters
        ----------
        edge_list : list of tuple
            List of edges
        thp : list or none, optional (default=None)
            List of weights. If None, not used.
        mode : str {'w', 'a'}, optional (default='w')
            write or append mode
        """

        # If mode == 'w', create output folder if it does not exist (if
        # mode == 'a', the output folder and files must already exist)
        if mode == 'w' and not os.path.isdir(self.tmp_folder):
            os.makedirs(self.tmp_folder)

        # Save edge_list
        with open(self.fpath_edges, mode) as f:
            writer = csv.writer(f)
            writer.writerows(edge_list)

        # Save weights
        if len(thp) > 0:
            with open(self.fpath_values, mode) as f:
                writer = csv.writer(f)
                # zip is used to save thp as a column
                writer.writerows(zip(thp))

        return

    def _move_from_file(self, get_values=True):
        """
        Write or append input lists in to files

        Returns
        -------
        edges : list of tuple
            List of edges
        values : list
            List of valuesw
        """

        # Read edges and delete temporary file
        with open(self.fpath_edges, 'r') as f:
            reader = csv.reader(f)
            edges = [(int(x[0]), int(x[1])) for x in map(tuple, reader)]
        os.remove(self.fpath_edges)

        # Get values, if requested
        if get_values:
            # Read values and delete temporary file
            with open(self.fpath_values, 'r') as f:
                reader = csv.reader(f)
                values = [float(x[0]) for x in reader]
            os.remove(self.fpath_values)
        else:
            values = None

        return edges, values

    def th_selfprod(self, th: float, X: csr_matrix, mode: str = 'distance',
                    verbose: bool = False) -> Union[
                        List[Tuple[int, int]], Tuple[List[Tuple[int, int]],
                                                     List[float]]]:
        """
        A blockwise implementation of the thresholded self-product of a matrix,
        X, which is defined as

            g(X @ X.T)

        where g(u) = u * (u>=th) (and it is computed element-wise when the
        argument is a matrix).

        This is a particular case of the thresholded product computed by
        self.th_product(), taking Y=X. However, there is a major difference:
        since g(X @ X.T) is a symmetric matrix, only the diagonal values and
        the upper triangular component is computed.

        Parameters
        ----------
        th : float
            Threshold
        X : numpy array
            Input matrix
        mode : {'distance', 'connectivity'}, optional (default='distance')
            If distance, a distance graph is computed. Edges and distances are
            returned.
            If connectivity, a binary connectivity graph is computed. Only
            edges are returned
        verbose : boolean
            If True, ongoing processing messages are shown.

        Returns
        -------
        edge_list : list of tuples
            List of edges
        thp : list
            List of thresholded products (corresponding to the elements in
            edge_list). Only for mode='distance'.
        """

        # # # number of bytes available in the first GPU
        # if self.useGPU:
        #     available_gpu_memory = int(
        #         GPUtil.getGPUs()[0].memoryFree * 1024**2)

        #     # seemingly, there is an overhead at the beginning
        #     available_gpu_memory -= 50 * 1024**2
        #     # max_n_elements = (available_gpu_memory //
        #     #                   np.dtype(np.float32).itemsize)

        # Number of rows of the input matrix
        N = X.shape[0]

        # Initialize outputs
        edge_list = []
        thp = []

        # Starting write mode (only for temporal savings
        save_mode = 'w'

        # Blockwise processing
        for i in range(0, N, self.blocksize):

            # print(f'{size_of_sparse(Z[i: i + blocksize, :])} megabytes')

            # the lhs (compressed sparse) array is arranged
            if self.useGPU:
                lhs_matrix = cupy.sparse.csr_matrix(
                    X[i: i + self.blocksize, :]).astype(np.float32)
            else:
                lhs_matrix = X[i: i + self.blocksize, :]

            for j in range(i, N, self.blocksize):

                t0 = vprint(f"i = {i}, j = {j}.  ", verbose)
                # Arrange rhs (compressed sparse) array

                if self.useGPU:
                    rhs_matrix = cupy.sparse.csr_matrix(
                        X[j: j + self.blocksize, :]).astype(np.float32)
                else:
                    rhs_matrix = X[j: j + self.blocksize, :]
                # Compute product
                S = lhs_matrix.dot(rhs_matrix.T)
                if self.useGPU:
                    S = S.get()
                t0 = vprint(f"Multiplying: {time()-t0:.4f} secs.", verbose)

                # Apply threshold
                if issparse(X):
                    S.data[S.data < th] = 0
                else:
                    S[S < th] = 0
                    if self.useGPU:
                        S = cupy.sparse.csr_matrix(S)
                    else:
                        S = csr_matrix(S)
                t0 = vprint(f"Thresholding: {time()-t0:.4f} secs.", verbose)

                # Extract coordinates and values from product
                uu = S.shape[1]
                abs_coord, coord_values = edges_weights_from_sparseU(
                    S.data, S.indptr, S.indices, uu, i, j)
                # WARNING: OLD CODE: THE THRESHOLD-DEPENDENT VERSION
                # abs_coord, coord_values = (
                #     edges_weights_from_sparseU_THRESHOLED(
                #         S.data, S.indptr, S.indices, S.shape[1], th, i, j))
                t0 = vprint(f"Listing: {time()-t0:.4f} secs", verbose)

                edge_list += abs_coord
                if mode == 'distance':
                    thp += coord_values
                t0 = vprint(f"Accumulating: {time()-t0:.4f} secs", verbose)
                vprint(f".. .. .. Edge list: {len(edge_list)} edges. "
                       f"Memory usage: {sys.getsizeof(edge_list)}", verbose)

                # Store lists if they reach the limit size
                if len(edge_list) > self.save_every:
                    vprint("Saving to temporary file", verbose)
                    # Save current lists
                    self._move_to_file(edge_list, thp, mode=save_mode)
                    edge_list, thp = [], []
                    # The next savings must be in append mode:
                    save_mode = 'a'

                del rhs_matrix, S, abs_coord, coord_values

        # This is not robust coding, but it works: we should not save the
        # current edge list if no saving has been done before. The way to
        # verify this is checking save_mode
        if save_mode == 'a':
            # Append last block to the saved list
            if len(edge_list) > 0:
                # Save current lists
                vprint("Saving to temporary file", verbose)
                self._move_to_file(edge_list, thp, mode=save_mode)

            # Load full list
            edge_list, thp = self._move_from_file(
                get_values=(mode == 'distance'))

        if mode == 'connectivity':
            return edge_list
        else:
            return edge_list, thp

    def th_prod(self, th: float, X: csr_matrix, Y: csr_matrix,
                mode: str = 'distance', verbose: bool = False) -> Union[
                    List[Tuple[int, int]],
                    Tuple[List[Tuple[int, int]], List[float]]]:
        """
        A blockwise implementation of the thresholded roduct of matrices X and
        Y, which is defined as

            g(X @ Y.T)

        where g(u) = u * (u>=th) (and it is computed elementwise when the
        argument is a matrix).

        Parameters
        ----------
        th : float
            Threshold
        X : numpy array
            Left matrix
        Y : numpy array
            Right matrix
        mode : {'distance', 'connectivity'}, optional (default='distance')
            If distance, a distance graph is computed. Edges and distances are
            returned.
            If connectivity, a binary connectivity graph is computed. Only
            edges are returned
        verbose : boolean
            If True, ongoing processing messages are shown.

        Returns
        -------
        edge_list : list of tuples
            List of edges
        thp : list
            List of thresholded products (corresponding to the elements in
            edge_list). Only for mode='distance'.
        """

        # # number of bytes available in the first GPU
        # if self.useGPU:
        #     available_gpu_memory = int(
        #         GPUtil.getGPUs()[0].memoryFree * 1024**2)

        #     # seemingly, there is an overhead at the beginning
        #     available_gpu_memory -= 50 * 1024**2
        #     # max_n_elements = (available_gpu_memory //
        #     #                  np.dtype(np.float32).itemsize)

        # in order to make sure it's an integer
        Nx, dimx = X.shape
        Ny, dimy = Y.shape

        # Initialize outputs
        edge_list = []
        thp = []

        # Starting write mode (only for temporal savings
        save_mode = 'w'

        # Blockwise processing
        for i in range(0, Nx, self.blocksize):

            # print(f'{size_of_sparse(Z[i: i + blocksize, :])} megabytes')

            # the lhs (compressed sparse) array is arranged
            if self.useGPU:
                lhs_matrix = cupy.sparse.csr_matrix(
                    X[i: i + self.blocksize, :]).astype(np.float32)
            else:
                lhs_matrix = X[i: i + self.blocksize, :]

            for j in range(0, Ny, self.blocksize):

                t0 = vprint(f"i = {i}, j = {j}.  ", verbose)

                # Arrange rhs (compressed sparse) array
                if self.useGPU:
                    rhs_matrix = cupy.sparse.csr_matrix(
                        X[j: j + self.blocksize, :]).astype(np.float32)
                else:
                    rhs_matrix = Y[j: j + self.blocksize, :]

                # Compute product
                S = lhs_matrix.dot(rhs_matrix.T)
                if self.useGPU:
                    S = S.get()
                t0 = vprint(f"Multiplying: {time()-t0:.4f} secs.", verbose)

                # Apply threshold
                if issparse(X) and issparse(Y):
                    S.data[S.data < th] = 0
                else:
                    S[S < th] = 0
                    S = csr_matrix(S)
                t0 = vprint(f"Thresholding: {time()-t0:.4f} secs.", verbose)

                # Extract coordinates and values from product
                abs_coord, coord_values = edges_weights_from_sparseLDU(
                    S.data, S.indptr, S.indices, S.shape[1], i, j)
                # WARNING: OLD CODE: THE THRESHOLD-DEPENDENT VERSION
                # abs_coord, coord_values = (
                #     edges_weights_from_sparseLDU_THRESHOLED(
                #         S.data, S.indptr, S.indices, S.shape[1], th, i, j))
                t0 = vprint(f"Listing: {time()-t0:.4f} secs", verbose)

                edge_list += abs_coord
                if mode == 'distance':
                    thp += coord_values
                t0 = vprint(f"Accumulating: {time()-t0:.4f} secs", verbose)
                vprint(f".. .. .. Edge list: {len(edge_list)} edges. "
                       f"Memory usage: {sys.getsizeof(edge_list)}", verbose)

                # Store lists if they reach the limit size
                if len(edge_list) > self.save_every:
                    vprint("Saving to temporary file", verbose)
                    # Save current lists
                    self._move_to_file(edge_list, thp, mode=save_mode)
                    edge_list, thp = [], []
                    # The next savings must be in append mode:
                    save_mode = 'a'

                del rhs_matrix, S, abs_coord, coord_values

        # This is not robust coing, but it works: we should not save the
        # current edge list if no saving has been done before. The way to
        # verify this is checking save_mode
        if save_mode == 'a':
            # Append last block to the saved list
            if len(edge_list) > 0:
                # Save current lists
                vprint("Saving to temporary file", verbose)
                self._move_to_file(edge_list, thp, mode=save_mode)

            # Load full list
            edge_list, thp = self._move_from_file(
                get_values=(mode == 'distance'))

        if mode == 'connectivity':
            return edge_list
        else:
            return edge_list, thp

    def cosine_sim_graph(
            self, X: csr_matrix, s_min: float, mode: str = 'distance',
            verbose: bool = False) -> Union[List[Tuple[int, int]], Tuple[
                List[Tuple[int, int]], List[float]]]:
        """
        Computes the truncated cosine similarity matrix between the rows of X

        If y and z are two rows of X, the cosine similarity is given
        by `y·z^T / (||y|·||z||)`

        Parameters
        ----------
        X : numpy array
            Input matrix
        s_min : float
            Threshold. All similarity values below smin will be set to zero
        mode : {'distance', 'connectivity'}, optional (default='distance')
            If 'distance', a similarity graph is computed
            If 'connectivity', a binary connectivity graph is computed
        verbose : boolean, optional (default=False)
            If True, ongoing processing messages are shown.

        Returns
        -------
        y : tuple of (list, list) or list
            If mode = 'connectivity': a list of edges
            If mode = 'distance': a tuple (list of edges, list of values)
        """

        # Normalize rows
        if issparse(X):
            normX = np.sum(X.power(2), axis=1)
        else:
            normX = np.sum(X**2, axis=1, keepdims=True)
        Z = X / np.sqrt(normX)

        # Compute thresholded product g(Z @ Z.T)
        return self.th_selfprod(s_min, Z, mode=mode, verbose=verbose)

    def cosine_sim_bigraph(
            self, X: csr_matrix, Y: csr_matrix, s_min: float,
            mode: str = 'distance', verbose: bool = False) -> Union[
                List[Tuple[int, int]],
                Tuple[List[Tuple[int, int]], List[float]]]:
        """
        Computes the truncated cosine similarity matrix between the
        rows of X and Y

        If x and y are two rows of X and Y, respectively, the cosine
        similarity is given by `y·z^T / (||y|·||z||)`

        Parameters
        ----------
        X : numpy array
            Input matrix
        Y : numpy array
            Input matrix
        s_min : float
            Threshold. All similarity values below smin will be set to zero
        mode : {'distance', 'connectivity'}, optional (default='distance')
            If 'distance', a similarity graph is computed
            If 'connectivity', a binary connectivity graph is computed
        verbose : boolean, optional (default=False)
            If True, ongoing processing messages are shown.

        Returns
        -------
        y : tuple of (list, list) or list
            If mode = 'connectivity': a list of edges
            If mode = 'distance': a tuple (list of edges, list of values)
        """

        # Normalize rows
        normX = np.sum(X**2, axis=1, keepdims=True)
        Zx = X / np.sqrt(normX)
        normY = np.sum(Y**2, axis=1, keepdims=True)
        Zy = Y / np.sqrt(normY)

        # Compute thresholded product g(Z @ Z.T)
        return self.th_prod(s_min, Zx, Zy, mode=mode, verbose=verbose)

    def ncosine_sim_graph(
            self, X: csr_matrix, s_min: float, mode: str = 'distance',
            verbose: bool = False) -> Union[List[Tuple[int, int]], Tuple[
                List[Tuple[int, int]], List[float]]]:
        """
        Computes the truncated and normalized cosine similarity matrix
        between the rows of X.

        It differs from :func:`cosine_sim_graph` in that the similarity
        values are rescaled from [-1, 1] to [0, 1]

        Parameters
        ----------
        X : numpy array
            Input matrix
        s_min : float
            Threshold. All similarity values below smin will be set to zero
        mode : {'distance', 'connectivity'}, optional (default='distance')
            If 'distance', a similarity graph is computed
            If 'connectivity', a binary connectivity graph is computed
        verbose : boolean, optional (default=False)
            If True, ongoing processing messages are shown.

        Returns
        -------
        y : tuple of (list, list) or list
            If mode = 'connectivity': a list of edges
            If mode = 'distance': a tuple (list of edges, list of values)
        """

        # A theshold s_min over the ncosine similarity is equivalent to a
        # threshold 2·s_min-1 over the cosine similarity
        s_min_cosine = 2 * s_min - 1

        # Compute cosine similarity rows
        edge_ids, weights = self.cosine_sim_graph(
            X, s_min_cosine, mode=mode, verbose=verbose)

        # Normalize
        weights = [(w + 1) / 2 for w in weights]

        return edge_ids, weights

    def ncosine_sim_bigraph(
            self, X: csr_matrix, Y: csr_matrix, s_min: float,
            mode: str = 'distance', verbose: bool = False) -> Union[
                List[Tuple[int, int]],
                Tuple[List[Tuple[int, int]], List[float]]]:
        """
        Computes the truncated and normalized cosine similarity matrix
        between the rows of X and Y

        It differs from :func:`cosine_sim_graph` in that the similarity
        values are rescaled from [-1, 1] to [0, 1]

        Parameters
        ----------
        X : numpy array
            Input matrix
        Y : numpy array
            Input matrix
        s_min : float
            Threshold. All similarity values below s_min will be set to zero
        mode : {'distance', 'connectivity'}, optional (default='distance')
            If 'distance', a similarity graph is computed
            If 'connectivity', a binary connectivity graph is computed
        verbose : boolean, optional (default=False)
            If True, ongoing processing messages are shown.

        Returns
        -------
        y : tuple of (list, list) or list
            If mode = 'connectivity': a list of edges
            If mode = 'distance': a tuple (list of edges, list of values)
        """

        # Compute cosine similarity rows
        edge_ids, weights = self.cosine_sim_graph(
            X, s_min, mode=mode, verbose=verbose)

        # Normalize
        weights = [(w + 1) / 2 for w in weights]

        return edge_ids, weights

    def bc_sim_graph(
            self, X: csr_matrix, s_min: float, mode: str = 'distance',
            verbose: bool = False) -> Union[List[Tuple[int, int]], Tuple[
                List[Tuple[int, int]], List[float]]]:
        """
        Computes the Bhattacharyya Coefficient (BC) between the rows of matrix
        X

        It asumes that the rows of X are stochastic vectors.

        If y and z are two rows of X, the BC is given by `srt(y)·sqrt(z)`,
        where the square roots are computed component-wise.

        Parameters
        ----------
        X : numpy array
            Input matrix.
        s_min : float
            Threshold. All similarity values below smin will be set to zero
        mode : {'distance', 'connectivity'}, optional (default='distance')
            If distance, a similarity graph is computed
            If connectivity, a binary connectivity graph is computed
        verbose : boolean, optional (default=False)
            If True, ongoing processing messages are shown.

        Returns
        -------
        y : tuple of (list, list) or list
            If mode = 'connectivity': a list of edges
            If mode = 'distance': a tuple (list of edges, list of values)
        """

        # Take the square root of matrix components in order to compute the He
        # distances as a function of a matrix product
        Z = np.sqrt(X)

        # Compute thresholded product g(Z @ Z.T)
        return self.th_selfprod(s_min, Z, mode=mode, verbose=verbose)

    def bc_sim_bigraph(
            self, X: csr_matrix, Y: csr_matrix, s_min: float,
            mode: str = 'distance', verbose: bool = False) -> Union[
                List[Tuple[int, int]],
                Tuple[List[Tuple[int, int]], List[float]]]:
        """
        Computes the truncated matrix of Bhattacharyya coefficients between the
        rows of X and Y

        If x and y are two rows of X and Y, respectively, the BC is given by
        `srt(x)·sqrt(y)`
        where the square roots are computed component-wise.


        Parameters
        ----------
        X : numpy array
            Input matrix
        Y : numpy array
            Input matrix
        s_min : float
            Threshold. All similarity values below smin will be set to zero
        mode : {'distance', 'connectivity'}, optional (default='distance')
            If 'distance', a similarity graph is computed
            If 'connectivity', a binary connectivity graph is computed
        verbose : boolean, optional (default=False)
            If True, ongoing processing messages are shown.

        Returns
        -------
        y : tuple of (list, list) or list
            If mode = 'connectivity': a list of edges
            If mode = 'distance': a tuple (list of edges, list of values)
        """

        # Normalize rows
        Zx = np.sqrt(X)
        Zy = np.sqrt(Y)

        # Compute thresholded product g(Z @ Z.T)
        return self.th_prod(s_min, Zx, Zy, mode=mode, verbose=verbose)

    def he_neighbors_graph(
            self, X: csr_matrix, R2: float, mode: str = 'distance',
            verbose: bool = False) -> Union[List[Tuple[int, int]], Tuple[
                List[Tuple[int, int]], List[float]]]:
        """
        Computes the truncated squared Hellinger distance matrix between the
        rows of X

        Parameters
        ----------
        X : numpy array
            Input matrix
        R2 : float
            Squared radius
        mode : {'distance', 'connectivity'}, optional (default='distance')
            If distance, a similarity graph is computed
            If connectivity, a binary connectivity graph is computed
        verbose : boolean, optional (default=False)
            If True, ongoing processing messages are shown.

        Returns
        -------
        edge_list : list of tuples
            List of edges
        he_dist2 : list
            Hellinger distances (only for mode = 'distance')
        """

        # Take the square root of matrix components in order to compute the He
        # distances as a function of a matrix product
        Z = np.sqrt(X)

        # Compute the threshold to be applied over the product Z @ Z.T
        s_min = 1 - R2 / 2

        if mode == 'connectivity':
            # Compute coordinates of the nonzero values of thresholded product
            # g(Z @ Z.T)
            edge_list = self.th_selfprod(s_min, Z, mode=mode, verbose=verbose)

            return edge_list

        else:
            # Compute thresholded product g(Z @ Z.T)
            edge_list, values = self.th_selfprod(
                s_min, Z, mode=mode, verbose=verbose)

            # Transform products into similarities (we override values to
            # reduce memory usage for large matrices)
            # This is a pythonic choice:
            # he_dist2 = [max(2. - 2. * p, 0.) for p in he_dist2]
            # This is much faster:
            values = np.clip(2. - 2. * np.array(values), a_min=0.,
                             a_max=None).tolist()

            return edge_list, values

    def he_neighbors_bigraph(
            self, X: csr_matrix, Y: csr_matrix, R2: float,
            mode: str = 'distance', verbose: bool = False) -> Union[
                List[Tuple[int, int]],
                Tuple[List[Tuple[int, int]], List[float]]]:
        """
        Computes the truncated squared Hellinger distance matrix between the
        rows of X and the rows of Y

        Parameters
        ----------
        X : numpy array
            Input left matrix
        Y : numpy array
            Input right matrix
        R2 : float
            Squared radius
        mode : {'distance', 'connectivity'}, optional (default='distance')
            If distance, a similarity graph is computed
            If connectivity, a binary connectivity graph is computed
        verbose : boolean, optional (default=False)
            If True, ongoing processing messages are shown.

        Returns
        -------
        edge_list : list of tuples
            List of edges
        he_dist2 : list
            Hellinger distances (only for mode = 'distance')
        """

        # Take the square root of matrix components in order to compute the He
        # distances as a function of a matrix product
        Zx = np.sqrt(X)
        Zy = np.sqrt(Y)

        # Compute the threshold to be applied over the product Z @ Z.T
        s_min = 1 - R2 / 2

        if mode == 'connectivity':
            # Compute coordinates of the nonzero values of thresholded product
            # g(Z @ Z.T)
            edge_list = self.th_prod(s_min, Zx, Zy, mode=mode, verbose=verbose)

            return edge_list

        else:
            # Compute thresholded product g(Z @ Z.T)
            edge_list, values = self.th_prod(
                s_min, Zx, Zy, mode=mode, verbose=verbose)

            # Transform products into similarities (we override he_dist2 to
            # reduce memory usage for large matrices)
            # This is a pythonic choice:
            # he_dist2 = [max(2. - 2. * p, 0.) for p in values]
            # This is much faster:
            values = np.clip(2. - 2. * np.array(values), a_min=0.,
                             a_max=None).tolist()

            return edge_list, values


@numba.jit
def edges_weights_from_sparseU_THRESHOLDED(
        data: np.ndarray, indptr: np.ndarray, indices: np.ndarray, n_cols: int,
        threshold: float, i_block: int, j_block: int) -> Tuple[list, list]:
    """
    Computes edge weight and absolute coordinates from a subblock of the Upper
    triangular component of a sparse matrix

    WARNING: This is an older version of self.edges_weigs_from_sparseU() that
    aimed to return the whole list of edges, i.e. those with zero and nonzero
    weight. HOWEVER, it was never tested and it has a bug on the "else" half.

    Since think that the case threshold <= 0 is no longer needed, I will not
    fix the bug, but I let the code alive just in case I re-discover its
    necessity.

    Parameters
    ----------
    data : np.ndarray
        Data
    indptr : np.ndarray
        Indices from the sparse matrix
    indices : np.ndarray
        Second list of indices
    n_cols : int
        Number of columns
    threshold : float
        Threshold value
    i_block : int
        Absolute i coordinate of the relative 0 coordinate
    j_block: int
        Absolute j coordinate of the relative 0 coordinate

    Returns
    -------
    absolute_coordinates : list
        List of edges
    values : list
        List of weights
    """

    # The number of rows can be inferred from the rows' indexes
    n_rows = len(indptr) - 1

    # The operation is different depending on whether the block is diagonal
    # or not
    diagonal = i_block == j_block

    # relative_coordinates = [1.0]

    # A list to accumulate the (absolute) *coordinates* of the array in the
    # non-zero positions
    absolute_coordinates = []

    # if there is a threshold...
    if threshold > 0:

        # a list to accumulate the *values* in the above coordinates
        values = []

        # if the passed sparse array is the product of a matrix with itself...
        if diagonal:

            # for every row...
            for i_row in range(n_rows):
                # for every non-zero index and column...
                for i_col, d in zip(indices[indptr[i_row]:indptr[i_row + 1]],
                                    data[indptr[i_row]:indptr[i_row + 1]]):

                    if i_row < i_col and d != 0:
                        # The coordinates are recorded...
                        # relative_coordinates.append((i_row, i_col))
                        # For some reason, an error arise in cuda in int() is
                        # not used below
                        absolute_coordinates.append(
                            (i_block + i_row, j_block + i_col))
                        # ...and so is the corresponding value of the array at
                        # that position
                        values.append(d)

        else:

            for i_row in range(n_rows):
                for i_col, d in zip(indices[indptr[i_row]:indptr[i_row + 1]],
                                    data[indptr[i_row]:indptr[i_row + 1]]):

                    if d != 0:
                        # the coordinates are recorded...
                        # relative_coordinates.append((i_row, i_col))
                        absolute_coordinates.append(
                            (i_block + i_row, j_block + i_col))

                        # ...and so is the corresponding value of the array at
                        # that position
                        values.append(d)

    else:

        # values that are not filled in should be zero
        values = [0.0 for _ in range(n_rows * n_cols)]

        if diagonal:

            # *all* the coordinates...
            for i_row in range(n_rows):
                for i_col in range(n_cols):

                    # ...of the upper triangular part of the matrix
                    if i_row < i_col:

                        # relative_coordinates.append((i_row, i_col))
                        absolute_coordinates.append(
                            (i_block + i_row, j_block + i_col))

            # for extracting the data
            for i_row in range(n_rows):
                for i_col, d in zip(indices[indptr[i_row]:indptr[i_row + 1]],
                                    data[indptr[i_row]:indptr[i_row + 1]]):

                    if i_row < i_col:
                        values[i_row * n_cols + i_col] = d

        else:

            for i_row in range(n_rows):
                for i_col in range(n_cols):

                    # relative_coordinates.append((i_row, i_col))
                    absolute_coordinates.append(
                        (i_block + i_row, j_block + i_col))

            # for extracting the data
            for i_row in range(n_rows):
                for i_col, d in zip(indices[indptr[i_row]:indptr[i_row + 1]],
                                    data[indptr[i_row]:indptr[i_row + 1]]):

                    values[i_row * n_cols + i_col] = d

    return absolute_coordinates, values


@numba.jit
def edges_weights_from_sparseLDU_THRESHOLDED(
        data: np.ndarray, indptr: np.ndarray, indices: np.ndarray, n_cols: int,
        threshold: float, i_block: int, j_block: int) -> Tuple[list, list]:
    """
    Computes edge weight and absolute coordinates from a subblock of a sparse
    matrix

    Parameters
    ----------
    data : np.ndarray
        Data
    indptr : np.ndarray
        Indices from the sparse matrix
    indices : np.ndarray
        Second list of indices
    n_cols : int
        Number of columns
    threshold : float
        Threshold value
    i_block : int
        Absolute i coordinate of the relative 0 coordinate
    j_block: int
        Absolute j coordinate of the relative 0 coordinate

    Returns
    -------
    absolute_coordinates : list
        List of edges
    values : list
        List of weights
    """

    # The number of rows can be inferred from the rows' indexes
    n_rows = len(indptr) - 1

    # A list to accumulate the absolute coordinates of the non-zero array
    # components
    absolute_coordinates = []

    # if there is a threshold...
    if threshold > 0:

        # a list to accumulate the *values* in the above coordinates
        values = []

        for i_row in range(n_rows):
            for i_col, d in zip(indices[indptr[i_row]:indptr[i_row + 1]],
                                data[indptr[i_row]:indptr[i_row + 1]]):

                if d != 0:
                    # the coordinates are recorded...
                    # relative_coordinates.append((i_row, i_col))
                    absolute_coordinates.append(
                        (i_block + i_row, j_block + i_col))
                    # ...and so is the corresponding value of the array at
                    # that position
                    values.append(d)

    # If the threshold is <=0, all matrix components should be returned.
    else:

        # values that are not filled in should be zero
        values = [0.0 for _ in range(n_rows * n_cols)]

        for i_row in range(n_rows):
            for i_col in range(n_cols):

                # relative_coordinates.append((i_row, i_col))
                absolute_coordinates.append(
                    (i_block + i_row, j_block + i_col))

        # for extracting the data
        for i_row in range(n_rows):
            for i_col, d in zip(indices[indptr[i_row]:indptr[i_row + 1]],
                                data[indptr[i_row]:indptr[i_row + 1]]):

                values[i_row * n_cols + i_col] = d

    return absolute_coordinates, values


class base_ThOps(object):
    """
    Generic class for the efficient computation of thresholded operation with
    dense and sparse matrices.

    This is a precursor class of ThOps. The methods in this class produce the
    exact same ouputs than those of ThOps. However, ThOps is much faster
    because methods have been accelerated with Numba. Thus, base_ThOps is
    deprecated and, in general, ThOps should be used.
    """

    def __init__(self, blocksize=25_000):
        """
        Stores the main attributes of a datagraph object and loads the graph
        data as a list of node attributes from a database

        Parameters
        ----------
        X : scipy.sparse.csr or numpy.array
            Matrix of node attribute vectors
        blocksize : int, optional (default=25_000)
            Size (number of rows) of each block in blocwise processing.
        """

        # ###############
        # Other variables

        # Size of blocks taken from the data matrix for blockwise processing
        # "int" is just to make sure that blocksize is an integer, to avoid an
        # execution error when used as an array index
        self.blocksize = int(blocksize)

        return

    def th_selfprod(self, th, X, mode='distance', verbose=True):
        """
        A blockwise implementation of the thresholded self-product of a matrix,
        X, which is defined as

            g(X @ X.T)

        where g(u) = u * (u>=th) (and it is computed element-wise when the
        argument is a matrix).

        This is a particular case of the thresholded product computed by
        self.th_product(), taking Y=X. However, there is a major difference:
        since g(X @ X.T) is a symmetric matrix, only the diagonal values and
        the upper triangular component is computed.

        Parameters
        ----------
        th : float
            Threshold
        X : numpy array
            Input matrix
        Y : numpy array, scipy.sparse.csr or None, optional (default=None)
            Input matrix
        mode : 'distance', 'connectivity', default = 'distance'
            If distance, a distance graph is computed
            If connectivity, a binary connectivity graph is computed
        verbose : boolean
            If True, ongoing process messages are shown.

        Returns
        -------
        edge_list : list of tuples
            List of edges
        thp : list
            List of thresholded products (corresponding to the elements in
            edge_list)
        """

        N = X.shape[0]

        # Initialize outputs
        edge_list = []
        if mode == 'distance':
            thp = []

        # Blockwise processing
        for i in range(0, N, self.blocksize):

            Xi = X[i: i + self.blocksize, :]

            for j in range(i, N, self.blocksize):

                t0 = vprint(f"i = {i}, j = {j}.  ", verbose)
                S = Xi @ X[j: j + self.blocksize, :].T
                t0 = vprint(
                    f"Multiplying: {time()-t0:.4f} secs.", verbose)

                # Apply threshold
                if issparse(X):
                    S.data[S.data < th] = 0
                else:
                    S[S < th] = 0
                    #     S = S * (S <= s_min)
                t0 = vprint(f"Thresholding: {time()-t0} secs.", verbose)

                # List of edges in block coordinates
                if i == j:
                    # For diagonal blocks, only edges with increasing
                    # coordinates are selected, to avoid duplicates, because
                    # the graph is undirected.
                    if th > 0:
                        # This is the usual case...
                        edges0 = [x for x in zip(*S.nonzero()) if x[0] < x[1]]
                    else:
                        # This is only for the unusual case that the fully
                        # connected graph is required
                        edges0 = [(m, n) for m in range(S.shape[0])
                                  for n in range(S.shape[1]) if m < n]
                else:
                    if th > 0:
                        # This is the usual case
                        edges0 = list(zip(*S.nonzero()))
                    else:
                        # Unusual case: fully connected graph
                        edges0 = [(m, n) for m in range(S.shape[0])
                                  for n in range(S.shape[1])]

                t0 = vprint(f"Listing {len(edges0)} edges in block "
                            f"coords: {time()-t0} secs.", verbose)

                if mode == 'distance':
                    thp += [S[x] for x in edges0]

                # list of edges in absolute coordinates
                edge_list += [(x[0] + i, x[1] + j) for x in edges0]
                vprint(f"Listing: {time()-t0} secs.", verbose)

        if mode == 'connectivity':
            return edge_list
        else:
            return edge_list, thp

    def th_prod(self, th, X, Y, mode='distance', verbose=True):
        """
        A blockwise implementation of the thresholded product of two matrices.

        For matrices X and Y and threshold th, the thresholded product is
        computed as

            g(X @ Y.T)

        where g(u) = u * (u>=th)

        Parameters
        ----------
        th : float
            Threshold
        X : numpy array or scipy.sparse.csr
            Input matrix
        Y : numpy array or scipy.sparse.csr
            Input matrix. It must the of the same type as X
        mode : 'distance', 'connectivity', default = 'distance'
            If distance, a distance graph is computed
            If connectivity, a binary connectivity graph is computed
        verbose : boolean
            If True, ongoing processing messages are shown.

        Returns
        -------
        edge_list : list of tuples
            List of edges
        thp : list
            List of thresholded products (corresponding to the tuples in
            edge_list)
        """

        Nx, dimx = X.shape
        Ny, dimy = Y.shape

        # Check data consistency
        if dimx != dimy:
            logging.error('-- -- Error in thresholded product: input '
                          'matrices must have the same number of columns')

        # Initialize outputs
        edge_list = []
        if mode == 'distance':
            thp = []

        # Blockwise processing
        for i in range(0, Nx, self.blocksize):

            Xi = X[i: i + self.blocksize, :]

            for j in range(0, Ny, self.blocksize):

                t0 = vprint(f"i = {i}, j = {j}.  ", verbose)
                S = Xi @ Y[j: j + self.blocksize, :].T
                t0 = vprint(f"Multiplying: {time()-t0} secs.", verbose)

                # Apply threshold
                if issparse(X) and issparse(Y):
                    S.data[S.data < th] = 0
                else:
                    S[S < th] = 0
                    #     S = S * (S <= s_min)
                t0 = vprint(f"Thresholding: {time()-t0} secs.", verbose)

                if th > 0:
                    # This is the usual case
                    edges0 = list(zip(*S.nonzero()))
                else:
                    # Unusual case: fully connected graph
                    edges0 = [(m, n) for m in range(S.shape[0])
                              for n in range(S.shape[1])]

                t0 = vprint(f"Listing {len(edges0)} edges in block "
                                  f"coords: {time()-t0} secs.", verbose)

                if mode == 'distance':
                    thp += [S[x] for x in edges0]

                # list of edges in absolute coordinates
                edge_list += [(x[0] + i, x[1] + j) for x in edges0]
                vprint(f"Listing: {time()-t0} secs.", verbose)

        if mode == 'connectivity':
            return edge_list
        else:
            return edge_list, thp

    def he_neighbors_graph(self, X, R0, mode='distance', verbose=True):
        """
        Computes the list of coordinates and vales of the truncated squared
        Hellinger distance matrix between the rows of X

        Parameters
        ----------
        X : numpy array
            Input matrix
        R0 : float
            Radius
        mode : 'distance', 'connectivity', default = 'distance'
            If distance, a distance graph is computed
            If connectivity, a binary connectivity graph is computed. No
            distance values are returned
        verbose : boolean
            If True, ongoing processing messages are shown.

        Returns
        -------
        edge_list : list of tuples
            List of edges
        he_dist2 : list
            Hellinger distances (only for mode = 'distance')
        """

        # Take the square root of matrix components in order to compute the He
        # distances as a function of a matrix product
        Z = np.sqrt(X)

        # Compute the threshold to be applied over the product Z @ Z.T
        s_min = 1 - R0**2 / 2

        if mode == 'connectivity':
            # Compute coordinates of the nonzero values of thresholded product
            # g(Z @ Z.T)
            edge_list = self.th_selfprod(s_min, Z, mode=mode, verbose=verbose)

            return edge_list

        else:
            # Compute thresholded product g(Z @ Z.T)
            edge_list, he_dist2 = self.th_selfprod(
                s_min, Z, mode=mode, verbose=verbose)
            # Transform products into similarities (we override he_dist2 to
            # reduce memory usage for large matrices)
            # This is a pythonic choice:
            # he_dist2 = [max(2. - 2. * p, 0.) for p in he_dist2]
            # This is much faster:
            he_dist2 = np.clip(2. - 2. * np.array(he_dist2), a_min=0.,
                               a_max=None).tolist()

            return edge_list, he_dist2

    def he_neighbors_bigraph(self, X, Y, R0, mode='distance', verbose=True):
        """
        Computes the truncated squared Hellinger distance matrix between the
        rows of X and the rows of Y

        Parameters
        ----------
        X : numpy array
            Input left matrix
        Y : numpy array
            Input right matrix
        R0 : float
            Radius
        mode : 'distance', 'connectivity', default = 'distance'
            If distance, a distance graph is computed
            If connectivity, a binary connectivity graph is computed
        verbose : boolean
            If True, ongoing processing messages are shown.

        Returns
        -------
        edge_list : list of tuples
            List of edges
        he_dist2 : list
            Hellinger distances (only for mode = 'distance')
        """

        # Take the square root of matrix components in order to compute the He
        # distances as a function of a matrix product
        Zx = np.sqrt(X)
        Zy = np.sqrt(Y)

        # Compute the threshold to be applied over the product Z @ Z.T
        s_min = 1 - R0**2 / 2

        if mode == 'connectivity':
            # Compute coordinates of the nonzero values of thresholded product
            # g(Z @ Z.T)
            edge_list = self.th_prod(s_min, Zx, Zy, mode=mode, verbose=verbose)

            return edge_list

        else:
            # Compute thresholded product g(Z @ Z.T)
            edge_list, he_dist2 = self.th_prod(
                s_min, Zx, Zy, mode=mode, verbose=verbose)

            # Transform products into similarities (we override he_dist2 to
            # reduce memory usage for large matrices)
            # This is a pythonic choice:
            # he_dist2 = [max(2. - 2. * p, 0.) for p in he_dist2]
            # This is much faster:
            he_dist2 = np.clip(2. - 2. * np.array(he_dist2), a_min=0.,
                               a_max=None).tolist()

            return edge_list, he_dist2




