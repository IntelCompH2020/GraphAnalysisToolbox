
import os
import shutil

from scipy.sparse import csr_matrix, issparse
import colored
import time
import logging
import numpy as np
import matplotlib.pyplot as plt

# Debugger
from sim_graph import SimGraph

import ipdb


def get_data(T, nmax):
    # Takes at most nmax rows from T at random.
    if nmax >= T.shape[0]:
        ind = range(n_nodes)
        nmax = n_nodes
    else:
        np.random.seed(3)    # For comparative purposes (provisional)
        ind = np.random.choice(range(n_nodes), nmax, replace=False)
    Tg = T[ind]
    return Tg


# #######################
# CONFIGURABLE PARAMETERS

# Type of data source.
source_file = False

# Create datagraph with the full matrix
nmax = 1_500      # Nodes for the quick test (smaller than n_nodes)
n_nodes = 2_000   # Sample size (use >= 1000)

if not source_file:
    # Size of the artificial dataset
    dim = 25          # Data dimension
    # Data generator
    alpha = 0.001 * np.ones(dim)  # Parameters of the Dirichlet distribution
    th = 0.01          # Threshold to sparsify the data matrix

# Similarity graphs
th_gauss = 0.01    # Threshold for the Gauss similarity graph
R = 0.2            # Radius for all neighbor graphs
g = 1              # Power factor for the similarity computation
outpath = './test/'      # Path to save the graphical results

logging.basicConfig(level='INFO')

# ###############
# DATA GENERATION

# Load data matrix
if source_file:
    print("-- Data loaded from files:")
    # Read data from numpy file
    # path_2_full = '../../../source_data/topics/pr_topics_fecyt_25.npz'
    # path_2_sparse = '../../../source_data/topics/pr_topics_fecyt_25_sparse.npz'
    path_2_full = '../../../source_data/FECYT_25/modelo.npz'
    path_2_sparse = '../../../source_data/FECYT_25/modelo_sparse.npz'
    print(f"   -- {path_2_full}")
    print(f"   -- {path_2_sparse}")

    # Read full matrix
    data_full = np.load(path_2_full, mmap_mode=None, allow_pickle=True,
                        fix_imports=True, encoding='ASCII')
    T = data_full['thetas']
    n_nodes_full, dim_full = T.shape
    print(f"-- Full data matrix with dimension {n_nodes_full} X {dim_full}:")

    # Read sparse matrix
    data_sparse = np.load(path_2_sparse, mmap_mode=None, allow_pickle=True,
                          fix_imports=True, encoding='ASCII')
    sparseT = csr_matrix(
        (data_sparse['thetas_data'], data_sparse['thetas_indices'],
         data_sparse['thetas_indptr']), shape=data_sparse['thetas_shape'])
    n_nodes_sp, dim_sp = sparseT.shape
    print(f"-- Sparse data matrix with dimension {n_nodes_sp} X {dim_sp}:")
    if n_nodes_sp != n_nodes_full or dim_sp != dim_full:
        print("--WARNING: The full and sparse data matrices have different" +
              " dimensions.")

    n = min(n_nodes_full, n_nodes_sp)
    if n_nodes > n:
        print(f"--WARNING: The selected number of nodes, {n_nodes} is " +
              f"higher than the size of the dataset, {n}. We take {n}")
        n_nodes = n
else:
    print('-- DATA GENERATION')
    T = np.random.dirichlet(alpha, size=n_nodes)
    B = T >= th
    sp_factor = (1 - np.sum(B) / (dim*n_nodes)) * 100
    print(f"-- Generating sparse thresholded matrix with {sp_factor} % of " +
          "zeros")
    sparseT = csr_matrix(T * B)
    print('\n')


# ########%#################
# TEST ALL SIMILARITY MODELS

# # Create datagraph with the full matrix
# print("--------------------------------------------")
# print("-- TESTING ALL GRAPH MODELS WITH FULL MATRIX")
# # Create SimGraph object
# Tg = get_data(T, nmax)
# sg = SimGraph(Tg)
# # Select similarities.
# sims = ['l1', 'He', 'He2', 'Gauss', 'l1->JS', 'He->JS', 'He2->JS', 'JS']
# # Compute similariries.
# for sim in sims:
#     print(f'\n-- -- {sim} similarities:')
#     sg.computeGraph(R, sim, g, th_gauss)
# print("-- TEST OK\n.")

# print("-----------------------------------------------")
# print("-- TESTING ALL GRAPH MODELS WITH SPARSE MATRIX")
# # Create SimGraph object
# Tg = get_data(sparseT, nmax)
# sg = SimGraph(Tg)
# # Select similariries.
# sims = ['l1', 'He', 'He2', 'Gauss', 'l1->JS', 'He->JS', 'He2->JS']
# # Compute similariries.
# for sim in sims:
#     print(f'\n-- -- {sim} similarities:')
#     sg.computeGraph(R, sim, g, th_gauss)
# print("-- TEST OK\n.")


# #####################
# TEST SIMILARITY PLOTS

# Create datagraph with the full matrix
print("--------------------------------------------")
print("-- TESTING SIMILARITY PLOTS")
# Create SimGraph object
Tg = get_data(T, nmax)
sg = SimGraph(Tg)

nw = np.sum(np.sum(Tg, axis=1) < 0.5)
print(f'Topic matrix with {nw} non-probabilistic vectors')

# Select similariries.
sims = ['l1->JS', 'He->JS', 'He2->JS']
Rtst = 0.1
# Compute similariries.
for sim in sims:
    print(f'\n-- -- {sim} similarities:')
    sg.computeGraph(R=Rtst, sim=sim, g=g, th_gauss=th_gauss)
    sg.show_JS_bounds(Rtst, sim, g, outpath, verbose=True)
print("-- TEST OK\n.")

# # ####################
# # TEST COMPUTING TIMES

# # Select similarities and data sizes
# sims = ['He2->JS']   # , 'l1->JS', 'He->JS', ]
# nmax_all = np.logspace(2, np.log10(n_nodes), num=8, endpoint=True,
#                        base=10.0)
# nmax_all = [int(n) for n in nmax_all]

# print("---------------------------------------------------------")
# print("-- TESTING COMPUTING TIME OF JS VERSIONS WITH FULL MATRIX")

# # Dictionary of empty lists
# time_record = {s: [] for s in sims}
# for nmax in nmax_all:

#     print(f'-- nmax = {nmax}')

#     # Create SimGraph object
#     Tg = get_data(T, nmax)
#     sg = SimGraph(Tg, outpath)

#     # Compute similariries.
#     for sim in sims:
#         print(f'\n-- -- {sim} similarities:')
#         t0 = time.time()
#         sg.computeGraph(R, sim, g, th_gauss)
#         tm = time.time() - t0
#         time_record[sim].append(tm)
#         print(f'-- -- Graph computed in {tm} seconds')

# plt.figure()
# for sim in sims:
#     plt.loglog(nmax_all, time_record[sim], '.-', label=sim)
# plt.xlabel('Sample size')
# plt.ylabel('Time')
# plt.legend()
# plt.show(block=False)

# fpath = os.path.join(outpath, 'TimeAnalysis_full.png')
# plt.savefig(fpath)
# print("-- TEST OK\n.")


# print("-----------------------------------------------------------")
# print("-- TESTING COMPUTING TIME OF JS VERSIONS WITH SPARSE MATRIX")

# # Dictionary of empty lists
# time_record = {s: [] for s in sims}
# for nmax in nmax_all:

#     print(f'-- nmax = {nmax}')

#     # Create SimGraph object
#     Tg = get_data(sparseT, nmax)
#     sg = SimGraph(Tg, outpath)

#     # Compute similariries.
#     for sim in sims:
#         print(f'\n-- -- {sim} similarities:')
#         t0 = time.time()
#         sg.computeGraph(R, sim, g, th_gauss)
#         tm = time.time() - t0
#         time_record[sim].append(tm)
#         print(f'-- -- Graph computed in {tm} seconds')

# plt.figure()
# for sim in sims:
#     plt.loglog(nmax_all, time_record[sim], '.-', label=sim)
# plt.xlabel('Sample size')
# plt.ylabel('Time')
# plt.legend()
# plt.show(block=False)

# fpath = os.path.join(outpath, 'TimeAnalysis_sparse.png')
# plt.savefig(fpath)
# print("-- TEST OK\n.")

ipdb.set_trace()
