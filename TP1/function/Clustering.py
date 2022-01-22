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
    
    if False: # Autre figure possible
        plt.figure()
        plt.plot(np.arange(len(VI/max(VI)))+1,np.flip(VI/max(VI),axis=0))
        plt.xlabel("Nombre de classes")
        plt.ylabel("Variance intraclasse/variance totale")


def Kmeans(X, y,  n_init = 10, stand = False, val = False, n = 10):
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
    nclus = len(set(y))
    Xcopy = X.copy()
    ycopy = y.copy()
    if (stand):
        Xcopy = StandardScaler().fit_transform(Xcopy)    
    
    k_means = KMeans(init = 'k-means++', n_clusters = nclus, n_init = n_init)
    if (val):
        err = 0
        for i in range (n):
            # on prend 1/10 de l'ensemble pour réduire le nombre d'erreurs
            ntest = np.floor(len(y)//10).astype(int)

            # mélange l'ensemble des données et la labels
            per = np.random.permutation(len(y))
            lt,la = per[:ntest], per[ntest:]

            # définition de l'ensemble d'apprentissage et de test
            Xa,Xt = Xcopy[la,:],Xcopy[lt,:] 
            ya,yt = ycopy[la],ycopy[lt] 

            k_means.fit(Xa)
            yhat = k_means.predict(Xt)      

            err += sum(yt != yhat)
        errm = err /n /len(yt)
        print("Taux d'erreur : ",round(errm,3))
    else: 
        k_means.fit(X)
        yhat = k_means.predict(X) 
        errl = sum(y != yhat) / len(y)
        print("Taux d'erreur: ", round(errl , 3))    

        # Matrice de confusion
        # Rq: La matrice fournie par confusion_matrix est carrée:
        #    On retire les lignes de zeros de conf_mat, dues au fait qu'il peut y avoir plus de classes que d'etiquettes
        #    Et de meme avec les colonnes s'il y a moins de classes
        conf_mat =  confusion_matrix(y, yhat)
        conf_mat=conf_mat[np.sum(conf_mat,axis=1)>0,:]
        conf_mat=conf_mat[:,np.sum(conf_mat,axis=0)>0]
        print("Matrice de confusion:")
        print("   Une ligne = un digit\n   Une colonne = un cluster\n")
        print(conf_mat) 


    
def comparaisons_inerties(X, y, s):
    """
    Comparaison des inerties Kmeans et CAH

    Parameters
    ----------
    X : DONNÉES.
    y : LABELS.
    stand : BOOLEEN, optional
        Standardise si True. The default is False
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
