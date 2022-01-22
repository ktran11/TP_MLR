#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 12:01:18 2022

@author: Agathe
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import  KMeans
from sklearn.preprocessing import StandardScaler


def arbre(X, y, s):
    """
    Calcul et tracé de l'arbre

    Parameters
    ----------
    X : DONNÉES.
    y : LABELS.
    s : SEUIL.

    Returns
    -------
    None.

    """
    X = StandardScaler().fit_transform(X)
    M = linkage(X, method = 'ward', metric = 'euclidean')
    plt.figure()
    plt.title('CAH. Visualisation des classes au seuil de ' + str(s))
    d = dendrogram(M,labels = list(y), 
                   orientation = 'right', 
                   color_threshold = s)
    #print(np.round(M[:,2],2))


def groupes(X, y, s):
    """
    Récupération des groupes

    Parameters
    ----------
    X : DONNÉES.
    y : LABELS.
    s : SEUIL.

    Returns
    -------
    None.

    """
    X = StandardScaler().fit_transform(X)
    M = linkage(X, method = 'ward', metric = 'euclidean')
    groupes = fcluster(M, t = s, criterion = 'distance')
    for k in range(1, np.max(groupes) + 1):
        print('Classe ' + str(k).ljust(3, ' ') + ': ', end = '')
        print(*y[np.where(groupes==k)])

def intraclasse(X, y):
    """
    Décroissance des variances intraclasse

    Parameters
    ----------
    X : DONNÉES.
    y : LABELS.

    Returns
    -------
    None.

    """
    X = StandardScaler().fit_transform(X)
    M = linkage(X,method = 'ward', metric = 'euclidean')
    VI = np.cumsum(M[:,2] ** 2) / 2
    plt.figure()
    plt.plot(np.arange(len(VI)) + 1, np.flip(VI, axis = 0))
    plt.xlabel("Nombre de classes")
    plt.ylabel("Variance intraclasse")
    if 1==0: # Autre figure possible
        plt.figure()
        plt.plot(np.arange(len(VI / max(VI))) + 1, 
                 np.flip(VI / max(VI), axis = 0))
        plt.xlabel("Nombre de classes")
        plt.ylabel("Variance intraclasse/variance totale")
    


def Kmeans(X, y, s):
    """
    Comparaison avec les Kmeans

    Parameters
    ----------
    X : DONNÉES.
    y : LABELS.
    s : seuil

    Returns
    -------
    k_means.inertia_, VI[-nclus].

    """
    X = StandardScaler().fit_transform(X)
    M = linkage(X, method = 'ward', metric = 'euclidean')
    groupes = fcluster(M, t = s, criterion = 'distance')
    nclus = np.max(groupes)
    k_means = KMeans(init = 'k-means++', n_clusters = nclus, n_init = 10)
    k_means.fit(X)
    for k in range(nclus):
        print('Classe ' + str(k + 1).ljust(3,' ')+': ', end = '')
        print(*y[np.where(k_means.labels_==k)])
    

def comparaisons_inerties(X, y, s):
    """
    Comparaison des inerties Kmeans et CAH

    Parameters
    ----------
    X : DONNÉES.
    y : LABELS.

    Returns
    -------
    None.

    """
    
    X = StandardScaler().fit_transform(X)
    M = linkage(X, method = 'ward', metric = 'euclidean')
    groupes = fcluster(M, t = s, criterion = 'distance')
    nclus = np.max(groupes)
    VI = np.cumsum(M[:,2] ** 2) / 2
    k_means = KMeans(init = 'k-means++', n_clusters = nclus, n_init = 10)
    k_means.fit(X)
    print("Inertie Kmeans", nclus, "centres: ", k_means.inertia_)
    print("Inertie CAH", nclus, "classes: ", VI[-nclus])
