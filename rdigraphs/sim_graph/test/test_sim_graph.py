"""
Tester for the ThOps class. Run from the parent folder as
     python -m test.test_sim_graph
"""

from scipy.sparse import csr_matrix
import logging
import numpy as np
import time
import pathlib
import matplotlib.pyplot as plt

from rdigraphs.sim_graph.sim_graph import SimGraph


def get_data(X, nmax):
    """
    Takes at most nmax rows from X at random.
    """
    if nmax >= X.shape[0]:
        ind = range(n_nodes)
        nmax = n_nodes
    else:
        # np.random.seed(3)    # For comparative purposes (provisional)
        ind = np.random.choice(range(n_nodes), nmax, replace=False)
    Xg = X[ind]
    return Xg


logging.basicConfig(level='INFO')

# #######################
# CONFIGURABLE PARAMETERS

# Create datagraph with the full matrix
nmax = 1000     # 1_500     # Nodes for the quick test (smaller than n_nodes)
n_nodes = 1000   # 2_000    # Sample size (use >= 1000)
n_edges = int(0.02 * n_nodes * (n_nodes - 1) / 2)

# Size of the artificial dataset
dim = 25       # Data dimension
# Data generator
alpha = 0.001 * np.ones(dim)  # Parameters of the Dirichlet distribution
alpha = 0.1 * np.ones(dim)  # Parameters of the Dirichlet distribution
th = 0.01      # Threshold to sparsify the data matrix

# Similarity graphs
g = 1              # Power factor for the similarity computation
outpath = pathlib.Path('./rdigraphs/test/')    # Path to save graphical results

# ###############
# DATA GENERATION

# Load data matrix
print('-- DATA GENERATION')
X = np.random.dirichlet(alpha, size=n_nodes)
B = X >= th
sp_factor = (1 - np.sum(B) / (dim * n_nodes)) * 100
print(f"-- Generating sparse thresholded matrix with {sp_factor} % of zeros")
Xs = csr_matrix(X * B)

# ##########################
# TEST ALL SIMILARITY MODELS

# Create datagraph with the full matrix
print("--------------------------------------------")
print("-- TESTING ALL GRAPH MODELS WITH DENSE MATRIX")
# Create SimGraph object
Xg = get_data(X, nmax)
sg = SimGraph(Xg)
print(f"-- Target number of edges: {n_edges}")

# Select similarities.
sims = ['BC', 'ncosine', 'JS', 'l1', 'He', 'He2', 'Gauss', 'l1->JS', 'He->JS',
        'He2->JS']
# sims = ['ncosine']

# Compute similariries.
for sim in sims:
    print(f'\n-- -- {sim} similarities:')
    sg.sim_graph(n_edges=n_edges, sim=sim, g=g, verbose=False)
exit()

# Create datagraph with the full matrix
print("----------------------------------------------")
print("-- TESTING ALL GRAPH MODELS WITH SPARSE MATRIX")
# Create SimGraph object
Xg = get_data(Xs, nmax)
sg = SimGraph(Xg)
print(f"-- Target number of edges: {n_edges}")

# Compute similariries.
for sim in sims:
    print(f'\n-- -- {sim} similarities:')
    sg.sim_graph(n_edges=n_edges, sim=sim, g=g, verbose=False)

print("-- TEST OK\n.")

# #####################
# TEST SIMILARITY PLOTS

# Create datagraph with the full matrix
print("--------------------------------------------")
print("-- TESTING SIMILARITY PLOTS")
# Create SimGraph object
Xg = get_data(X, nmax)
sg = SimGraph(Xg)

nw = np.sum(np.sum(Xg, axis=1) < 0.5)
print(f'Topic matrix with {nw} non-probabilistic vectors')

# Select similariries.
sims = ['l1->JS', 'He->JS', 'He2->JS']
s_min = 0.8
# Compute similariries.
for sim in sims:
    print(f'\n-- -- {sim} similarities:')
    sg.sim_graph(s_min=s_min, sim=sim, g=g)
    sg.show_JS_bounds(s_min, sim, g, out_path=outpath,
                      verbose=True)
print("-- TEST OK\n.")

# ####################
# TEST COMPUTING TIMES

# Select similarities and data sizes
sims = ['He2->JS']   # , 'l1->JS', 'He->JS', ]
nmax_all = np.logspace(2, np.log10(n_nodes), num=8, endpoint=True,
                       base=10.0)
nmax_all = [int(n) for n in nmax_all]

print("---------------------------------------------------------")
print("-- TESTING COMPUTING TIME OF JS VERSIONS WITH FULL MATRIX")

# Dictionary of empty lists
time_record = {s: [] for s in sims}
for nmax in nmax_all:

    print(f'-- nmax = {nmax}')

    # Create SimGraph object
    Xg = get_data(X, nmax)
    sg = SimGraph(Xg)

    # Compute similariries.
    for sim in sims:
        print(f'\n-- -- {sim} similarities:')
        t0 = time.time()
        sg.sim_graph(s_min, sim=sim, g=g)
        tm = time.time() - t0
        time_record[sim].append(tm)
        print(f'-- -- Graph computed in {tm} seconds')

plt.figure()
for sim in sims:
    plt.loglog(nmax_all, time_record[sim], '.-', label=sim)
plt.xlabel('Sample size')
plt.ylabel('Time')
plt.legend()
plt.show(block=False)

fpath = outpath / 'TimeAnalysis_full.png'
plt.savefig(fpath)
print("-- TEST OK\n.")


print("-----------------------------------------------------------")
print("-- TESTING COMPUTING TIME OF JS VERSIONS WITH SPARSE MATRIX")

# Dictionary of empty lists
time_record = {s: [] for s in sims}
for nmax in nmax_all:

    print(f'-- nmax = {nmax}')

    # Create SimGraph object
    Xg = get_data(Xs, nmax)
    sg = SimGraph(Xg)

    # Compute similariries.
    for sim in sims:
        print(f'\n-- -- {sim} similarities:')
        t0 = time.time()
        sg.sim_graph(s_min, sim=sim, g=g)
        tm = time.time() - t0
        time_record[sim].append(tm)
        print(f'-- -- Graph computed in {tm} seconds')

plt.figure()
for sim in sims:
    plt.loglog(nmax_all, time_record[sim], '.-', label=sim)
plt.xlabel('Sample size')
plt.ylabel('Time')
plt.legend()
plt.show(block=False)

fpath = outpath / 'TimeAnalysis_sparse.png'
plt.savefig(fpath)
print("-- TEST OK\n.")

