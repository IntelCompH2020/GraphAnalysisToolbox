#! /usr/bin/env python3

import os

from scipy.sparse import csr_matrix
import time
import logging
import numpy as np
import matplotlib.pyplot as plt

# Debugger
from .sim_graph import SimGraph


def get_data(T, nmax):
    # Takes at most nmax rows from T at random.
    if nmax >= T.shape[0]:
        ind = range(n_nodes)
    else:
        np.random.seed(3)    # For comparative purposes (provisional)
        ind = np.random.choice(range(n_nodes), nmax, replace=False)
    Tg = T[ind]
    return Tg


# #############################################################################
# CONFIGURABLE PARAMETERS
# #############################################################################

# Type of data source.
source_file = True
sparse_matrix = True

# Create datagraph with the full matrix
nmax = 1_500        # Nodes for the quick test (smaller than n_nodes)
n_nodes = 1_000_000   # Sample size (use >= 1000)

if source_file:

    # Path to the npz file containing the topic model for the whole corpus
    path_2_T = '../../../source_data/topic_models/FECYT_25/modelo_sparse.npz'

else:

    # Size of the artificial dataset
    dim = 25          # Data dimension
    # Data generator
    alpha = 0.001 * np.ones(dim)  # Parameters of the Dirichlet distribution
    th = 0.01          # Threshold to sparsify the data matrix

# Parameters for the similarity graphs
sim = 'He2'      # Similarity measure
R = 0.02         # Radius for all neighbor graphs (the higher the Radius, 
                 # the denser the similarity graph)
g = 1            # Power factor for the similarity graph computation
rescale = False  # If True, similarities are rescaled to span interval [0, 1] 
outpath = './test/'   # Path to save the graphical results
blocksize = 40_000    # Size of each block (no. of rows) for the block-by-
                      # block computation of the similarity matrix

logging.basicConfig(level='INFO')

# #############################################################################
# DATA GENERATION
# #############################################################################

# Load data matrix
if source_file:

    # ################
    # Read matrix data
    data = np.load(path_2_T, mmap_mode=None, allow_pickle=True,
                   fix_imports=True, encoding='ASCII')
    if sparse_matrix:
        print(f"-- Dense matrix.")
        T = csr_matrix(
            (data['thetas_data'], data['thetas_indices'],
             data['thetas_indptr']), shape=data['thetas_shape'])
    else:
        print(f"-- Sparse matrix.")
        T = data_full['thetas']

    n_nodes_T, dim = T.shape
    print(f"-- Data matrix with dimension {n_nodes_T} X {dim}:")
    if n_nodes > n_nodes_T:
        print(f"-- WARNING: The selected number of nodes, {n_nodes} is " +
              f"higher than the size of the dataset, {n_nodes_T}. " +
              f"We take {n_nodes_T}")
        n_nodes = n_nodes_T

else:

    print('-- DATA GENERATION')
    T = np.random.dirichlet(alpha, size=n_nodes)

    if sparse_matrix:
        B = T >= th
        sp_factor = (1 - np.sum(B) / (dim * n_nodes)) * 100
        print(f"-- Generating sparse thresholded matrix with {sp_factor} % of " +
              "zeros")
        T = csr_matrix(T * B)
    print('\n')


# # #############################################################################
# # TEST ALL SIMILARITY MODELS
# # #############################################################################
#
# # Create datagraph with the full matrix
# print("-----------------------------------")
# print("-- TESTING GRAPH MODEL CONSTRUCTION")
# # Create SimGraph object
# Tg = get_data(T, nmax)
#
# # This is just to test if all topic vectors are probabilistic, as expected.
# # (not critical, so the execution continues anyway)
# nw = np.sum(np.sum(Tg, axis=1) < 0.5)
# print(f'Topic matrix with {nw} non-probabilistic vectors')
#
# # Compute similariries.
# print(f'\n-- -- He similarity:')
# sg = SimGraph(Tg)
# sg.computeGraph(R=R, sim=sim, g=g, rescale=False)
# print("-- TEST OK\n.")

# #############################################################################
# TEST COMPUTING TIMES
# #############################################################################

# Select similarities and data sizes
nmax_all = np.logspace(2, np.log10(n_nodes), num=8, endpoint=True,
                       base=10.0)
nmax_all = [int(n) for n in nmax_all]

# nmax_all = [50_000]
# nmax_all = [25_000]
# nmax_all = [35_000]
nmax_all = [100_000]

print("-------------------------")
print("-- TESTING COMPUTING TIME")

# Dictionary of empty lists
time_record = []
for nmax in nmax_all:

    print(f'-- nmax = {nmax}')

    # Create SimGraph object
    Tg = get_data(T, nmax)
    sg = SimGraph(Tg)

    # Compute similariries.
    print(f'\n-- -- He similarities:')
    t0 = time.time()
    sg.computeGraph(R=R, sim=sim, g=g, rescale=False, blocksize=blocksize,
                    quick_comp=False)
    tm = time.time() - t0
    time_record.append(tm)
    print(f'-- -- Graph computed in {tm} seconds')

    sg_fast = SimGraph(Tg)
    # Compute similariries.
    print(f'\n-- -- He similarities (fast version):')
    t0 = time.time()
    sg.computeGraph(R=R, sim=sim, g=g, rescale=False, blocksize=blocksize,
                    quick_comp=True)
    tm = time.time() - t0
    print(f'-- -- Graph computed in {tm} seconds')

plt.figure()
plt.loglog(nmax_all, time_record, '.-', label='He2')
plt.xlabel('Sample size')
plt.ylabel('Time')
plt.legend()
plt.show(block=False)

fpath = os.path.join(outpath, 'TimeAnalysis_full.png')
plt.savefig(fpath)
print("-- TEST OK\n.")
