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
        
def calc_errr_Kmeans(M):
    """
    Calcul le taux d'erreur à partir d'une matrice de confusion M
   
    Parameters
    ----------
    M : ndarray
        Matrice de confusion.

    Returns
    -------
    float
        taux d'erreur.

    """
    res = 0.
    for i in M:
        res += np.sum(i) - i.max()
    return round(res/M.sum(),3)

def Kmeans_func(X, y,  n_init = 10, val = False, n = 10):
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
    k_means = KMeans(init = 'k-means++', n_clusters = nclus, n_init = n_init)

    if (val):
        err_array = []
        err = 0
        for i in range(n):
            
            # on prend 1/10 de l'ensemble pour réduire le nombre d'erreurs
            ntest=np.floor(len(y)//2).astype(int)
            
            # mélange l'ensemble des données et la labels
            per=np.random.permutation(len(y))
            lt,la=per[:ntest], per[ntest:]
            
            # définition de l'ensemble d'apprentissage et de test
            Xa,Xt=Xcopy[la,:],Xcopy[lt,:] 
            ya,yt=ycopy[la],ycopy[lt] 
            
            k_means.fit(Xa,ya)
            yhat = k_means.predict(Xt)
            conf_mat =  confusion_matrix(yt,yhat)

            errx = calc_errr_Kmeans(conf_mat) 
            err_array += [errx]
            err += errx
        err = err /n 
            
        conf_mat =  confusion_matrix(yt,yhat)
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
        return err, err_array

    else:
    
        k_means.fit(Xcopy)
        yhat = k_means.predict(Xcopy)
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
        return calc_errr_Kmeans(conf_mat)




def Scatter_plot_k_means(X, y, index1 = 0, index2 = 1):
    """
    Crée un nuage de points sur les composantes principales données par l'analyse discriminante

    Parameters
    ----------
    X : DONNÉES.
    y : LABELS.
    index1: INT, optional
        axe des abscisses = composante principale index1. The default is 0.
    index2: INT, optional
        axe des ordonnées = composante principale index2. The default is 1.
 
    Returns
    -------
    None.

    """
    Xcopy = X.copy()
    ycopy = y.copy()
    y_label = list(set(ycopy))
    nclus = len(y_label)
    k_means = KMeans(init = 'k-means++', n_clusters = nclus, n_init = 10)
    C = k_means.fit_transform(Xcopy, ycopy)
    plt.figure()
    vlab = np.unique(y)
    for i, vl in enumerate(vlab):
        l = (y == vl)
        plt.scatter(C[l, index1], C[l, index2], s = 4, label = vl)
        plt.legend()
