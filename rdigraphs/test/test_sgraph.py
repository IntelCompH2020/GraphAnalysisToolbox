import numpy as np

from rdigraphs.supergraph.supergraph import SuperGraph


def generate_edges(n_nodes, epn):
    """
    Generates a random collection of edges from a number (n_nodes) of nodes.

    Parameters
    ----------
    n_nodes : int
        Number of nodes
    epn : int
        Number of edges per node
    """

    source_nodes, target_nodes, weights = [], [], []
    for i in range(n_nodes):
        top = (i % 2000 + 1) * 2000
        for j in range(epn):
            if i + j + 1 < min(n_nodes, top):
                source_nodes.append(i)
                target_nodes.append(i + j + 1)
                weights.append(np.random.rand())

    return source_nodes, target_nodes, weights


# Configurable parameters
n_nodes = 10000
epn = 10  # edges per node
path2supergraph = '../mysupergraph'

# Secondary parameters
n_edges = epn * n_nodes

# Create an empty supergraph object
mySG = SuperGraph(snode=None, path=path2supergraph, label="sg",
                  keep_active=False)

# Create main graph
# Create snode with the nodes only.
nodes = list(range(n_nodes))
mg = 'maingraph'
mySG.makeSuperNode(mg, nodes=nodes, edge_class='undirected')
# Add edges
source_nodes, target_nodes, weights = generate_edges(n_nodes, epn)
mySG.snodes[mg].set_edges(source_nodes, target_nodes, weights)

# Apply community detection algorithm to the main graph
comm_label = 'community'
mySG.detectCommunities(mg, alg='leiden', ncmax=20,
                       comm_label=comm_label)

# Generate graph partition into community subgraphs
n_comm = max(mySG.snodes[mg].df_nodes[comm_label])
xlabel = 'maingraph'
for i in range(n_comm):
    mySG.sub_snode_by_value(xlabel, comm_label, i, ylabel=f'G_{i}')

mySG.save_supergraph()
