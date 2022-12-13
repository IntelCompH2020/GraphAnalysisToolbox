#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Python libraries
import numpy as np
import matplotlib.pyplot as plt
import logging
import ipdb

"""
This script contains several methods to logging.info(and visualize components
and statistics from a datagraph.
"""


def printStats(df_nodes, label):
    """
    logging.info(some statistics about the size and content of the graph data

    Parameters
    ----------
    df_nodes : dataframe
        Dataframe
    label : str
        Label
    """

    n_items = len(df_nodes.index)
    attributes = set(list(df_nodes)) - set(['Id'])

    logging.info("=================================")
    logging.info("Database: " + label)
    logging.info("-- Some preliminary stats")
    logging.info("-- -- Total no. of items: {0}".format(n_items))
    for n, tok in enumerate(attributes):
        # Some stats about the number of fields containing data.
        is_data = np.array([0 if d == '' or d is None else 1
                            for d in df_nodes[tok]])
        n_data = np.count_nonzero(is_data)
        if type(df_nodes[tok][0]) is np.float64:
            n_nan = np.count_nonzero(
                [np.isnan(d) for d in df_nodes[tok]])
            n_data -= n_nan
        logging.info("-- -- Non empty values of field {0}: {1}".format(
                     tok, n_data))
    logging.info('')


def plot_cluster_analysis(scores, fpath):
    """
    Plots the values of one or more scores for clustering evaluation, as a
    function of the number of clusters.

    Parameters
    ----------
    scores : dict
        Scores
    fpath : str
        Path to save the figure
    """

    # Extract data from the list of tuples in scores
    k_set = scores['k_set']

    titles = {'sil': 'Silhouette score',
              'chs': 'Calinski-Harabaz score',
              'mod': 'Modularity'}

    for score_name in set(scores) - set(['k_set']):

        s = scores[score_name]

        plt.figure()
        plt.scatter(k_set, s, c='b')
        kopt = k_set[np.argmax(s)]
        sopt = np.max(s)
        plt.stem([kopt], [sopt], 'r-o')

        plt.axis('tight')
        plt.xlabel('Number of clusters')
        plt.title(titles[score_name])
        plt.show(block=False)

        plt.savefig(fpath + score_name + '.png')


def plotSortedFeatures(X, fpath=None):
    """
    Plots the average of all sorted topics vectors

    Parameters
    ----------
    X : array
        Input matrix
    fpath : str or None, optional (default=None)
        Path to save the figure
    """

    dim = X.shape[1]

    Xsorted = - np.sort(-X, axis=1)
    x_mean = np.mean(Xsorted, axis=0)
    x_std = np.std(Xsorted, axis=0)

    ind = np.arange(dim)  # the x locations for the groups
    width = 0.8       # the width of the bars
    fig, ax = plt.subplots()
    ax.bar(ind, x_mean, width, color='b', yerr=x_std)
    ax.set_title('Average of the sorted topic weights')
    ax.set_xlabel('Topic rank')
    ax.set_ylabel('Topic weight')
    ax.set_ylim([0, ax.get_ylim()[1]])

    logging.info("-- -- The mean of the sorted topic distribution is " +
                 "{0}".format(x_mean.dot(ind)))

    plt.show(block=False)

    if fpath is not None:
        fig.savefig(fpath)


def plotAverageTopic(X, fpath=None):
    """
    Plots the sorted average of all sorted topics vectors

    Parameters
    ----------
    X : array
        Input matrix
    fpath : str or None, optional (default=None)
        Path to save the figure
    """

    dim = X.shape[1]

    x_mean = np.mean(X, axis=0)
    x_std = np.std(X, axis=0)

    ind = np.arange(dim)  # the x locations for the groups
    width = 0.8       # the width of the bars
    fig, ax = plt.subplots()
    ax.bar(ind, x_mean, width, color='b', yerr=x_std)
    ax.set_xlabel('Topic index')
    ax.set_ylabel('Average Topic weight')
    ax.set_ylim([0, ax.get_ylim()[1]])

    plt.show(block=False)

    logging.info(zip(ind, x_mean))

    if fpath is not None:
        fig.savefig(fpath)


def plotMainTopic(X, fpath=None):
    """
    Plots the main topic

    Parameters
    ----------
    X : array
        Input matrix
    fpath : str or None, optional (default=None)
        Path to save the figure
    """

    dim = X.shape[1]
    ind = np.arange(dim)  # the x locations for the groups

    main_topics = np.argmax(X, axis=1)

    # Count nodes with a dominant topic
    n_nodes = X.shape[0]
    num_x = []
    for i in ind:
        num_x.append(float(np.count_nonzero(main_topics == i)) /
                     n_nodes * 100)

    # Count nodes with a highly dominant topic
    Xdom = X >= 0.5
    dom_x = np.sum(Xdom, axis=0).astype(float) / n_nodes * 100

    logging.info(("-- -- There is as strong dominant topic (with weight > " +
                  "0.5 in {0} % of nodes").format(np.sum(dom_x)))

    width = 0.8       # the width of the bars
    fig, ax = plt.subplots()
    ax.bar(ind, num_x, width, color='b', label="Dominant topic")
    ax.bar(ind, dom_x, width, color='g', label="Strongly dominant topic")
    ax.legend()
    ax.set_xlabel('Topic index')
    ax.set_ylabel("% of projects dominated by topic")

    plt.show(block=False)

    if fpath is not None:
        fig.savefig(fpath)


def plotCXmatrix(M, fpath=None):
    """
    Plots matrix

    Parameters
    ----------
    M : array
        Input matrix
    fpath : str or None, optional (default=None)
        Path to save the figure
    """

    fig, ax = plt.subplots()   # figsize=(9, 4))
    ax.imshow(M, interpolation='nearest', cmap="gray")
    plt.xticks(range(M.shape[1]), range(M.shape[1]))
    plt.yticks(range(M.shape[0]), range(M.shape[0]))
    ax.axis('equal')
    plt.show(block=False)

    if fpath is not None:
        fig.savefig(fpath)


def plotClusterWeights(labels, fpath=None, n=None):
    """
    Barplot of the nuber of items per cluster.

    Plots the main topic

    Parameters
    ----------
    labels : list
        Labels
    fpath : str or None, optional (default=None)
        Path to save the figure
    n : int or None, optional (default=None)
        If none, show all clusters. If integer, it only shows the highest n
        clusters
    """

    nc_max = int(np.max(labels)) + 1
    if n is None:
        nc = nc_max
    else:
        nc = min(n, nc_max)

    ind = range(nc)
    weights = []

    for c in ind:
        weights.append(labels.count(c))

    width = 0.8       # the width of the bars
    fig, ax = plt.subplots()
    ax.bar(ind, weights, width, color='b')
    ax.legend()
    ax.set_xlabel('Cluster index')
    ax.set_ylabel("Cluster size")
    plt.show(block=False)

    if fpath is not None:
        fig.savefig(fpath)


def printClusters(M):
    """
    Prints clusters


    Parameters
    ----------
    M : array
        Input matrix
    """
    nc = M.shape[0]

    Tdict = {}

    # logging.info("Grupos minimos de topicos que dominan al resto"

    # for c in range(nc):

    #     # Sort weights of cluster c
    #     msort = - np.sort(- M[c])
    #     isort = np.argsort(- M[c])

    #     m_up = 0
    #     m_rest = np.sum(msort)

    #     Tdict[c] = []
    #     for k, m in enumerate(msort):

    #         if k * m_rest >= m_up:
    #             Tdict[c].append(isort[k])
    #             m_up += m
    #             m_rest -= m
    #         else:
    #             break

    #     logging.info(u"Clúster {0}: tópicos ".format(c), Tdict[c]

    logging.info("Topicos que conjuntamente superan el 50 % del peso total")

    for c in range(nc):

        # Sort weights of cluster c
        msort = - np.sort(- M[c])
        isort = np.argsort(- M[c])

        m_up = 0
        m_rest = np.sum(msort)

        Tdict[c] = []
        for k, m in enumerate(msort):

            if m_rest >= 0.5:
                Tdict[c].append(isort[k])
                m_up += m
                m_rest -= m
            else:
                break

        logging.info(u"Cluster {0}: tópicos ".format(c), Tdict[c])


def rankEdges(df_edges, df_nodes, fields, n):
    """"
    Show the n edges with highest weight.

    Refer the nodes using the specified field in df_nodes

    Parameters
    ----------
        df_edges : dataframe
            Edges
        df_nodes : dataframe
            Nodes
        fields : str
            Column containing the weights
        n : int
            Number of edges
    """

    # Get indices of the n highest nodes.
    ind = np.argsort(df_edges['Weight'].tolist())[-1:-n-1:-1]

    logging.info("-- -- Most Similar Edges")

    for i in ind:

        source = df_edges.loc[i]['Source']
        sname = df_nodes[df_nodes[fields[0]] == source][fields[1]].tolist()[0]
        target = df_edges.loc[i]['Target']
        tname = df_nodes[df_nodes[fields[0]] == target][fields[1]].tolist()[0]
        w = df_edges.loc[i]['Weight']
        logging.info(sname + ', ' + tname + ': ' + str(w))

