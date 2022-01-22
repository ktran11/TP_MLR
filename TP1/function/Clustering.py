#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 12:01:18 2022

@author: Agathe & Kevin
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import  KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Classification Ascendante hiérarchique --------------------------------------------


def Visualisation_CAH(X, y, method = 'ward', metric = 'euclidian', seuil = 10, stand = False):
    """
    
    Donne une visualisation de la classsification ascendante hiérarchique.

    Parameters
    ----------
    X : ndarray
        Donnée.
    y : ndarray
        Labels.
    method : string, optional
        Méthode pour la classification. The default is 'ward'.
    metric : string, optional
        Métrique utilisé pour la classification. The default is 'euclidian'.
    seuil : int
        Permet de déterminer le nombre de classes.
    stand : booleen, optional
        Standardise si True. The default is False.

    Returns
    -------
    None.

    """
    Xcopy = X.copy()
    if (stand):
        Xcopy = StandardScaler().fit_transform(Xcopy)
    M = linkage(Xcopy, method = 'ward', metric = 'euclidean')
    
    # Arbre de la CAH ------------------------------------------------
    plt.figure()
    plt.title('CAH. Visualisation des classes au seuil de ' + str(seuil))
    d = dendrogram(M,labels = list(y), 
                   orientation = 'right', 
                   color_threshold = seuil)
    #print(np.round(M[:,2],2))

def BarPlotMat(M):
    # Fait un barplot pour chaque colonne de M.
    # La couleur correspond à l'indice, la hauteur à la valeur
    I = M.shape[0]
    J = M.shape[1]
    ind = np.arange(J)
    haut = 0 * M[0,:]
    for i in range(I):
        plt.bar(ind,M[i,:],bottom=haut,color=plt.cm.inferno(i/(I-1)))
        haut += M[i,:]

def Kmeans_func(X, y,  n_init = 10, stand = False, val = False, n = 10):
    """
    Algorithme des K moyennes, taux d'erreurs et matrice de confusion

    Parameters
    ----------
    X : ndarray
        Données.
    y : ndarray
        Labels.
    n_init : int, optional
        Nombre d'initiation de l'algo K-means. The default is 10.
    stand : booleen, optional
        Standardisation si True. The default is False.
    val : booleen, optional
        Validation croisée si True. The default is False.
    n : int, optional
        nombre d'itération. The default is 10.
    Returns
    -------
    None.

    """

    Xcopy = X.copy()
    ycopy = y.copy()
    y_label = list(set(ycopy))
    nclus = len(y_label)
      
    for i in range(nclus):
        ycopy[ycopy == y_label[i]] = i
    
    ycopy = ycopy.astype(int)

    if (stand):
        Xcopy = StandardScaler().fit_transform(Xcopy)  
        
    
    k_means = KMeans(init = 'k-means++', n_clusters = nclus, n_init = n_init)
    k_means.fit(X)
    yhat = k_means.predict(X)
    # Matrice de confusion
    # Rq: La matrice fournie par confusion_matrix est carrée:
    #    On retire les lignes de zeros de conf_mat, dues au fait qu'il peut y avoir plus de classes que d'etiquettes
    #    Et de meme avec les colonnes s'il y a moins de classes
    conf_mat =  confusion_matrix(ycopy,yhat)
    conf_mat=conf_mat[np.sum(conf_mat,axis=1)>0,:]
    conf_mat=conf_mat[:,np.sum(conf_mat,axis=0)>0]
    print("Matrice de confusion:")
    print("   Une ligne = une maladie\n   Une colonne = un cluster\n")
    print(conf_mat)
    plt.figure(figsize = [8,5])
    BarPlotMat(conf_mat)
    plt.xlabel('Classe')
    plt.ylabel('Répartition des étiquettes')
    plt.title('Répartition dans chaque classe')
    plt.legend(y_label)


