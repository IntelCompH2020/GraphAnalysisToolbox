#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main program for the RDIgraph analyzer

Created on June 18 2018

@author: Jesús Cid Sueiro

# For profiling: kernprof -lv mainRDIgraph.py ...

"""

import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local imports
from rdigraphs.sgtaskmanager import SgTaskManager

# #######################
# Configurable parameters
# #######################

# Set tu True if you want to regenerate the graph data
reset_graph = True

# Select project:
n = 10
path2project = pathlib.Path('..') / 'projects' / f'cordisAI_{n}topics'
path2source = pathlib.Path('..') / 'datasets'
G = 'Cordis_Kwds3_AI_topics'
params = {'select_all': True, 'n_topics': f'topics{n}'}

# Graph generation
sim = 'BC'
n_epn = 10 if n == 26 else 13
# Community detection
# algorithm = 'louvain'
algorithm = 'leiden'

# Graph of communities
order = 1
# Threshold to sparsify the community graph
th = 0.016 if n == 26 else 0.011

# #################
# Open task manager
# #################

paths2data = {'topicmodels': path2source / 'topic_models',
              'agents': path2source / 'agents',
              'ACL_models': path2source / 'ACL_models'}
tm = SgTaskManager(path2project, paths2data, path2source)

paths2data = {'graphs': path2project / 'graphs',
              'bigraphs': path2project / 'bigraphs',
              'topicmodels': path2source / 'topic_models',
              'agents': path2source / 'agents'}

# ######################
# Load or create project
# ######################

if not reset_graph and path2project.is_dir():
    tm.load()
    tm.show_SuperGraph()
else:
    tm.create()
    tm.setup()

# ############
# Create graph
# ############

# Import nodes and attributes
graphs = tm.get_graphs_with_features()
if G not in graphs:
    tm.import_snode_from_table(
        G, n0=0, sampling_factor=1, params=params)

# Path to the graph
path2G = paths2data['graphs'] / G

# Compute similarity graph
tm.infer_sim_graph(path2G, sim, n0=0, n_epn=n_epn)

# Community detection
tm.detectCommunities('cc', path2G, comm_label=None)
tm.detectCommunities(algorithm, path2G, comm_label=None, seed=0)

# ##########
# Plot graph
# ##########
attribute = algorithm
tm.graph_layout(G, attribute)
att_2_rgb = tm.display_graph(G, attribute)

tm.SG.activate_snode(G)
coms = tm.SG.snodes[G].df_nodes.leiden.tolist()
Tmean = tm.SG.snodes[G].T.mean(axis=0)
T2mean = (tm.SG.snodes[G].T**2).mean(axis=0)
Tstd = np.sqrt(T2mean - Tmean**2)
flabels = tm.SG.snodes[G].metadata['feature_labels']

# ##################
# Transductive graph
# ##################

# Bipartite graph
tm.inferBGfromA(path2G, attribute, t_label=None, e_label=None)

# Transductive graph
GC = tm.SG.get_sedges()[0]
path2GC = paths2data['bigraphs'] / GC
tm.transduce(path2GC, order)
C = f"{algorithm}_{G}"

# Community detection
path2C = paths2data['graphs'] / C
tm.filter_edges(path2C, th)
tm.detectCommunities(algorithm, path2C, comm_label=None)
# Label nodes
tm.SG.label_nodes_from_features(C, att='tag', thp=2.5)

# Plot transductive graph
tm.graph_layout(C, attribute)
tm.display_graph(C, attribute)

# ##############################
# Export graphs to parquet files
# ##############################
tm.export_2_parquet(path2G)
tm.export_2_parquet(path2C)

# #######################
# Analyze EU projects
# #######################

# Get community labels from the snodes of the community graph
tm.SG.activate_snode(C)
comm_names = tm.SG.snodes[C].df_nodes.tag.tolist()
# Some name changes to reduce the length of the labels
comm_names = [name.replace(' and ', ' & ') for name in comm_names]
comm_names = [name.replace('Natural Language Processing', 'NLP')
              for name in comm_names]
comm_names = [name.replace('Environmental', 'Environm.')
              for name in comm_names]
comm_names = [name.replace('Reinforcement', 'Reinf.') for name in comm_names]
comm_names = [name.replace('Assessment', 'Assessm.') for name in comm_names]

# Compute a dictionary with the frequency of each community
comm_dict = dict(zip(tm.SG.snodes[C].df_nodes.Id.tolist(), comm_names))
eu_labels = [comm_dict[str(i)] for i in coms]

freq = {name: 0 for name in comm_names}
for label in eu_labels:
    freq[label] += 1
print(freq)

# Sort dictionary by decreasing values
freqs = list(freq.values())
colors = att_2_rgb.values()
sorted_tuples = sorted(zip(freqs, comm_names, colors), reverse=True)
sorted_freqs, sorted_comm_names, sorted_colors = zip(*sorted_tuples)

# Plot a pie chart with the frequency of each community
plt.figure(figsize=(10, 6))
plt.pie(sorted_freqs, labels=sorted_comm_names, autopct='%1.1f%%',
        colors=sorted_colors)
plt.title('EU projects by community')
plt.show(block=False)

# Save figure in the following path
tm.SG.activate_snode(G)
path = tm.SG.snodes[G].path2graph / 'eu_coms.png'
plt.savefig(path)

# #######################
# Analyze spanish projects
# #######################

# Detect Spanish projects in graph G.
isES = (tm.SG.snodes[G].df_nodes['coordinatorCountry'] == 'ES').tolist()
tm.SG.add_snode_attributes(G, 'is_spanish', isES)

# Display the graph highlighting Spanish projects
path = tm.SG.snodes[G].path2graph / 'is_spanish.png'
tm.display_graph(G, 'is_spanish', path=path)

# Get communities of Spanish projects
tm.SG.activate_snode(G)
coms = tm.SG.snodes[G].df_nodes.leiden.tolist()
isES = tm.SG.snodes[G].df_nodes.is_spanish.tolist()
comsES = [coms[i] for i in range(len(coms)) if isES[i]]

# Get community labels from the snodes of the community graph
tm.SG.activate_snode(C)

# Compute a dictionary with the frequency of each community
# in es_labels
es_labels = [comm_dict[str(i)] for i in comsES]
freq = {name: 0 for name in comm_names}
for label in es_labels:
    freq[label] += 1
print(freq)

# Sort dictionary by decreasing values
freqs = list(freq.values())
sorted_tuples = sorted(zip(freqs, comm_names, colors), reverse=True)
sorted_freqs, sorted_comm_names, sorted_colors = zip(*sorted_tuples)

# Plot a pie chart with the frequency of each community
plt.figure(figsize=(10, 6))
plt.pie(sorted_freqs, labels=sorted_comm_names, autopct='%1.1f%%',
        colors=sorted_colors)
plt.title('Spanish projects by community')
plt.show(block=False)

# Save figure in the following path
path = tm.SG.snodes[G].path2graph / 'es_coms.png'
plt.savefig(path)

# ##############################
# Evolution of communities
# ##############################

# Select columns of starting dates and communities
tm.SG.activate_snode(G)
# df_cy = tm.SG.snodes[G].df_nodes.loc[:, ['leiden', 'startDate', 'totalCost']]
df_cy = tm.SG.snodes[G].df_nodes.loc[:, ['leiden', 'startDate',
                                         'ecMaxContribution']]

# Keep rows without nan values only
ind = [i for i, x in enumerate(df_cy['startDate']) if isinstance(x, str)]
df_cy = df_cy.iloc[ind]

# Column 'startDate' contains dates as strings. Take first 4 characters (year)
comms = df_cy.leiden.tolist()
years = df_cy.startDate.tolist()
years = [x[:4] for x in years]
# budgets = df_cy.totalCost.tolist()
budgets = df_cy.ecMaxContribution.tolist()

# Maping years to indices
period = sorted(list(set(years)))
year_2_ind = dict(zip(period, list(range(len(period)))))

# Compute array communities times years
num_comms = len(set(comms))
num_years = len(period)
X = np.zeros((num_comms, num_years))
Y = np.zeros((num_comms, num_years))
for i in range(len(comms)):
    X[comms[i], year_2_ind[years[i]]] += 1
    bi = float(budgets[i].replace(',', '.')) / 1_000_000
    Y[comms[i], year_2_ind[years[i]]] += bi

# Save X and Y into xlsx files
dfX = pd.DataFrame(X, columns=period, index=comm_names)
dfY = pd.DataFrame(Y, columns=period, index=comm_names)
dfX.to_excel(tm.SG.snodes[G].path2graph / 'evolution.xlsx')
dfY.to_excel(tm.SG.snodes[G].path2graph / 'funding_evolution.xlsx')

# Make a linear plot of the evolution of communities
plt.figure(figsize=(10, 6))
plt.stackplot(period, X, labels=comm_names, alpha=0.8, colors=colors)
plt.xlabel('Years')
plt.ylabel('Number of projects')
plt.title('Evolution of communities')
# place the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# Make more space on the right side to fit the legend
plt.subplots_adjust(right=0.55)
# Rotate xticks
plt.xticks(rotation=90)
# Save figure in a file named 'evolution.png'
path = tm.SG.snodes[G].path2graph / 'evolution.png'
plt.savefig(path)
plt.show(block=False)

# Make a linear plot of the evolution of communities
plt.figure(figsize=(10, 6))
plt.stackplot(period, Y, labels=comm_names, alpha=0.8, colors=colors)
plt.xlabel('Years')
plt.ylabel('Budget (mill. €)')
plt.title('Evolution of communities')

# place the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# Make more space on the right side to fit the legend
plt.subplots_adjust(right=0.55)
# Rotate xticks
plt.xticks(rotation=90)
# Save figure in a file named 'evolution.png'
path = tm.SG.snodes[G].path2graph / 'funding_evolution.png'
plt.savefig(path)
plt.show(block=False)

# ############################
# Centrality measures
# ############################

local_metrics = ['centrality', 'degree', 'betweenness', 'closeness',
                 'pageRank', 'cluster_coef']
# katz not included because it raises an error.

# Compute local metrics
for metric in local_metrics:
    print(f"-- -- Metric: {metric}")
    tm.SG.local_snode_analysis(G, parameter=metric)

tm.SG.save_supergraph()





