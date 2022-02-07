import numpy as np
from scipy.sparse import csr_matrix

# Debugger
from th_ops import ThOps


# #######################
# Configurable parameters

# Size of the input matrix
n_nodes = 1_000   # No. of rows
dim = 20          # No. of columns

# Parameers for the input matrix generation
alpha = 0.1 * np.ones(dim)  # Parameters of the Dirichlet distribution
spf = 0.8                   # Sparsity factor

# Thresholded product
th_xx = 0.5          # Threshold for the thresholded product
tmp_folder = './tmp/'      # Path to save the graphical results
blocksize = 200
save_every = 300

# ###############
# Data generation

# Load data matrix
T = np.random.dirichlet(alpha, size=n_nodes)
v = sorted(T.flatten())
th_t = v[int(spf * n_nodes)]
B = T >= th_t
# Renormalize rows
sumT = np.sum(T, axis=1, keepdims=True)
if np.any(sumT == 0):
    exit("Some rows are all zeros. Increase the sparsity factor spf")
T = T / sumT
X = csr_matrix(T * B)

# ########################
# Test thresholded product
sg = ThOps(blocksize=blocksize, useGPU=False, tmp_folder=tmp_folder,
           save_every=save_every)

# Test
print('Testing self product...')
edges, weights = sg.th_selfprod(th_xx, X, mode='distance', verbose=True)
print('Testing cross product...')
edges, weights = sg.th_prod(th_xx, X, X, mode='distance', verbose=True)

print("-- TEST OK\n.")

