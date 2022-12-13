#!/bin/bash

# only required if "anaconda" is not in the path
source $HOME/anaconda3/etc/profile.d/conda.sh

NAME=grafos

# carta: mysqlclient
# conda-forge: python-louvain python-igraph
conda create --yes -n $NAME python=3 pip pyyaml colorama pandas matplotlib seaborn scipy scikit-learn numba cython tqdm ipdb networkx mysqlclient sphinx python-louvain python-igraph cupy leidenalg -c defaults -c conda-forge -c carta

conda activate $NAME

pip install neo4j-driver

# had to replace "ld" (the linker) in $HOME/anaconda3/envs/$NAME/compiler_compat with a symlink to the system's "ld" (which ld)
pip install fa2
