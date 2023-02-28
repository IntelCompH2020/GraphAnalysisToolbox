"""
Tester for the ThOps class. Run from the parent folder as
     python -m test.test_th_ops

"""

import numpy as np
from scipy.sparse import csr_matrix

# Debugger
from th_ops import ThOps


def compare_graphs(e1, e2, w1, w2):
    """
    Checks if the graphs with edges and weights (e1, w1) and (e2, w2) are equal

    Edges may be sorted in different orders in each list.

    The comparison is not strict, but enough to verify equality with high
    probability.
    """

    if len(e1) != len(e2):
        print('-- TEST FAILED: edge lists have different sizes')

    elif len([1 for x, y in zip(sorted(e1), sorted(e2)) if x != y]) != 0:
        print('-- TEST FAILED: edges are different')
        exit()

    else:

        e = np.array(sorted(w1)) - np.array(sorted(w2))
        sse = np.mean(e**2)

        if sse > 1e-10:
            print('-- TEST FAILED: weights are different')
        else:
            print('ok')
        return


def compare_graph_bigraph(e1, e2, w1, w2):
    """
    Checks if the graph given by edges (e1, w1) is equal to the graph given
    by (e2, w2)

    The second graph has been computed as a bigraph, so it includes self
    edges (i, i) and symmetric edges (i, j)-(j,i).  They will be removed.
    """
    inds = [i for i, x in enumerate(e2) if x[0] < x[1]]
    e2 = [e2[i] for i in inds]
    w2 = [w2[i] for i in inds]

    compare_graphs(e1, e2, w1, w2)
    return


def generate_matrix(n_nodes=100, dim=10, spf=0.8):

    # Load data matrix
    alpha = 0.1 * np.ones(dim)  # Parameters of the Dirichlet distribution

    X = np.random.dirichlet(alpha, size=n_nodes)
    v = sorted(X.flatten())
    th_t = v[int(spf * n_nodes)]
    B = X >= th_t
    # Forze X to have many zeroes...
    X = X * B
    # Check no all-zero rows
    sumX = np.sum(X, axis=1, keepdims=True)
    if np.any(sumX == 0):
        exit("Some rows are all zeros. Increase the sparsity factor spf")
    # Normalize to get a row-stochastic matrix
    X = X / sumX

    return X


# #######################
# Configurable parameters

# # Size of the input matrix
n_nodes = 500   # No. of rows
dim = 20          # No. of columns
n_nodes = 10   # No. of rows
dim = 2          # No. of columns
spf = 0.8                   # Sparsity factor

# Thresholded product
s_min = 0.5          # Threshold for the thresholded product
tmp_folder = './tmp/'      # Path to save the graphical results
blocksize = 100
save_every = 200

# ########################
# Test thresholded product
sg = ThOps(blocksize=blocksize, useGPU=False, tmp_folder='./tmp/',
           save_every=save_every)

# Generate data matrices
X = generate_matrix(n_nodes=n_nodes, dim=dim, spf=spf)
Xs = csr_matrix(X)

# Compute gold standard
S = X @ X.T
S = np.triu(S * (S >= s_min), k=1)
edges = list(zip(*np.nonzero(S)))
weights = [S[e] for e in edges]

# ###########################################
# Test 01: Testing th_prod vs golden standard

print('Test 01: self product with non-sparse matrix... ', end="")
edges1, weights1 = sg.th_selfprod(s_min, X, mode='distance', verbose=False)
compare_graphs(edges, edges1, weights, weights1)

print('Test 02: self product with sparse matrix... ', end="")
edges2, weights2 = sg.th_selfprod(s_min, Xs, mode='distance', verbose=False)
compare_graphs(edges, edges2, weights, weights2)


# #####################################
# Test 02: Test if products are correct

print('Test 03: product with non-sparse matrix... ', end="")
edges3, weights3 = sg.th_prod(s_min, X, X, mode='distance', verbose=False)
compare_graph_bigraph(edges, edges3, weights, weights3)

print('Test 04: product with parse matrix... ', end="")
edges4, weights4 = sg.th_prod(s_min, Xs, Xs, mode='distance', verbose=False)
compare_graphs(edges3, edges4, weights3, weights4)


# #####################################
# Test 03: Test cosine neighbor graphs

print('Test 05: cosine_neighbors_graph... ', end="")
edges5, weights5 = sg.cosine_sim_graph(
    X, s_min, mode='distance', verbose=False)
print('ok')
print('Test 06: cosine_neighbors_bigraph... ', end="")
edges6, weights6 = sg.cosine_sim_bigraph(
    X, X, s_min, mode='distance', verbose=False)
compare_graph_bigraph(edges5, edges6, weights5, weights6)

# #####################################
# Test 04: Test bc neighbor graphs

print('Test 07: bc_neighbors_graph... ', end="")
e_bc, w_bc = sg.bc_sim_graph(
    X, s_min, mode='distance', verbose=False)
print('ok')
print('Test 08: bc_neighbors_bigraph... ', end="")
e_bc2, w_bc2 = sg.bc_sim_bigraph(
    X, X, s_min, mode='distance', verbose=False)
compare_graph_bigraph(e_bc, e_bc2, w_bc, w_bc2)

# #####################################
# Test 05: Test he neighbor graphs

radius = np.sqrt(2 - 2 * s_min)
print('Test 09: he_neighbors_graph... ', end="")
e_he, d_he = sg.he_neighbors_graph(X, radius, mode='distance', verbose=False)
w_he = [1 - d / 2 for d in d_he]
compare_graph_bigraph(e_bc, e_he, w_bc, w_he)

print('Test 10: he_neighbors_bigraph... ', end="")
e_he2, d_he2 = sg.he_neighbors_bigraph(
    X, X, radius, mode='distance', verbose=False)
compare_graph_bigraph(e_he, e_he2, d_he, d_he2)

