#!/usr/bin/env python
# -*- coding: utf-8 -*-

# PlotLoss plots partial dives over the 3-class probability triangle.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import ipdb

# A constant for the smallest positive real.
EPS = np.nextafter(0, 1)


def compute_div(q, eta, div_name):
    '''
    Computes a divergence measure  D(eta, q), where eta is the true posterior
    and q is an estimate.

    Parameters
    ----------
    eta : array, shape (dimension, 1)
        True posterior
    q : array, shape (dimension, 1)
        Estimate of the posterior
    div_name : str {'L2', 'L1', 'KL', 'JS', 'dH'}
        Divergence meeasure. Available options are 'L2'  (square div), 'L1' 
        (L1 distance), 'KL' (Kullback-Leibler), 'JS' (Jensen-Shannon, 
        divergence), 'dH' (absolute difference between entropies, this is not
        a true divergence)

    Returns
    -------
    D : float
        Divergence value D(eta, q)
    '''

    if div_name == 'L2':   # ## Square div
        D = np.sum((eta - q)**2)
    elif div_name == 'L1':
        D = np.sum(np.abs(eta - q))
    elif div_name == 'KL':     # ## Cross entropy
        D = - np.dot(eta.T, np.log((q + EPS) / eta))
    elif div_name == 'JS':
        m = (q + eta) / 2
        d_eta = np.dot(eta.T, np.log(eta / m))
        d_q = np.dot(q.T, np.log((q + EPS) / m))
        D = (d_eta + d_q) / 2
    elif div_name == 'dH':
        # Absolute difference between entropies. This is not a true divergence
        h_q = - np.dot(q.T, np.log(q + EPS))
        h_eta = - np.dot(eta.T, np.log(eta + EPS))
        D = np.abs(h_q - h_eta)
    elif div_name == 'He':
        # Hellinger distance
        D = np.sum((np.sqrt(eta) - np.sqrt(q))**2)
    return D


def compute_simplex(div_name, eta, N=300):
    """
    Computes a divergence meeasure D(q, eta) between a given eta and all values
    of q in a grid over the triangular simplex of 3-class probabilities.

    Parameters
    ----------
    div_name : str {'L2', 'L1', 'KL', 'JS', 'dH'}
        Divergence meeasure. Available options are 'L2'  (square div), 'L1' 
        (L1 distance), 'KL' (Kullback-Leibler), 'JS' (Jensen-Shannon, 
        divergence), 'dH' (absolute difference between entropies, this is not
        a true divergence)
    eta : array, shape (dimension, 1)
        True posterior
    N : int
        Size of the grid

    Returns
    -------
    meandiv : array (N, N)
        Divergence values
    destMin : float
        Coordinate o the minimizer
    pestMin : float
        Second component of the minimizer
    """

    # ## Points (q0,q1,q2) to evaluate in the probability triangle
    p = np.linspace(0, 1, N)    # Values of q2;
    delta = np.linspace(-1, 1, N)   # Values of q0-q1;

    MinLossij = 1e10
    meandiv = np.ma.masked_array(np.zeros((N, N)))

    for i in range(N):
        for j in range(N):

            # ## Compute class probabilities corresponding to delta[i], p[j]
            q2 = p[j]
            q1 = (1 - q2 + delta[i])/2
            q0 = (1 - q2 - delta[i])/2
            q = np.array([q0, q1, q2])

            # Since q may lie out of the probability triengle, we must check
            # if it is a probability vector:
            if np.all(q >= 0):

                # The point is in the probability triange. Evaluate div
                meandiv[i, j] = compute_div(q, eta, div_name)

                # Locate the position of the minimum div
                if meandiv[i, j] < MinLossij:
                    MinLossij = meandiv[i, j]
                    destMin = delta[i]
                    pestMin = p[j]

            else:

                # The point is out of the probability simplex.
                # WARNING: surprisingly enough, the command below does not work
                #          if I use meandiv[i][j] instead of meandiv[i, j]
                meandiv[i, j] = np.ma.masked

    return meandiv, destMin, pestMin


def main():
    """
    Main
    """

    # ## Evaluate div in the probability triangle

    #########################
    # Configurable parameters

    # Parameters
    # eta = np.array([0.35, 0.2, 0.45])       # Location of the minimum
    # eta = np.array([0.3, 0.01, 0.69])       # Location of the minimum
    eta = np.array([0.3, 1e-10, 0.7 - 1e-10])       # Location of the minimum
    # 'L2', 'KL', 'L1', 'JS', 'dH' or 'He'
    div_names = ['KL', 'L1', 'JS', 'He']

    # Options
    n_div = len(div_names)
    options = {'L2': 'L2',
               'KL': 'Kullback-Leibler',
               'L1': 'L1',
               'JS': 'Jensen-Shannon',
               'dH': 'Difference between entropies',
               'He': 'Hellinger'}

    tags = [options[d] for d in div_names]

    # Parameters for the plots
    fs = 12   # Font size

    # Other common parameters
    # ## Location of the  minimum
    pMin = eta[2]
    dMin = eta[1] - eta[0]

    # ## Points (q0, q1, q2) to evaluate in the probability triangle
    N = 300
    p = np.linspace(0, 1, N)        # Values of q2;
    delta = np.linspace(-1, 1, N)   # Values of q0-q1;

    # This is a scale factor to make sure that the plotted triangle is
    # equilateral
    alpha = 1 / np.sqrt(3)

    # Saturated values. These values have been tested experimentally:
    vmax = {'L2': 1.6,
            'KL': 1.6,
            'JS': 1.6,
            'L1': 1.6,
            'dH': 1.6,
            'He': 1.6}

    fig = plt.figure(figsize=(4 * n_div, 2.6))

    for i, div_name in enumerate(div_names):

        print(div_name)

        # ###################
        # ## Loss computation

        # Compute div values over the probability simplex
        meandiv, destMin, pestMin = compute_simplex(div_name, eta, N)

        # ## Paint loss surface
        meandiv[meandiv > vmax[div_name]] = vmax[div_name]

        mlMax = np.max(np.max(meandiv))
        mlMin = np.min(np.min(meandiv))
        print(mlMin)
        print(mlMax)
        scaledmeandiv = 0.0 + 1.0 * (meandiv - mlMin) / (mlMax - mlMin)

        # Contour plot
        # from matplotlib import colors,
        ax2 = fig.add_subplot(1, n_div, i + 1)
        xx, yy = np.meshgrid(p, delta)
        levs = np.linspace(0, vmax[div_name], 10)
        mask = scaledmeandiv == np.inf

        scaledmeandiv[mask] = np.ma.masked

        ax2.contourf(alpha * yy, xx, scaledmeandiv, levs, cmap=cm.Blues)

        # Plot true posterior
        ax2.scatter(alpha * dMin, pMin, color='g')
        ax2.text(alpha * dMin + 0.025, pMin, '$\eta$', size=fs)

        # Plot minimum (estimated posterior)
        ax2.scatter(alpha * destMin, pestMin, color='k')
        ax2.text(alpha * destMin + 0.025, pestMin, '$\eta^*$', size=fs)
        ax2.plot([-alpha, 0, alpha, -alpha], [0, 1, 0, 0], '-')
        ax2.axis('off')

        # ## Write labels
        ax2.text(-0.74, -0.1, '$(1,0,0)$', size=fs)
        ax2.text(0.49, -0.1, '$(0,1,0)$', size=fs)
        ax2.text(0.05, 0.96, '$(0,0,1)$', size=fs)
        ax2.axis('equal')

        ax2.set_title(tags[i], size=fs)

    plt.show(block=False)

    plt.savefig('example3.svg')


if __name__ == "__main__":
    main()

