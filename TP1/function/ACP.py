#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 11:20:02 2022

@author: Kevin
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def ACP(X,  stand = False):
    """
    Analyse par composantes principales
    
    Parameters
    ----------
    X : ndarray
        jeux de données.
    stand : boolean, optional
        standardise les données si True. The default is False.

    Returns
    -------
    C : ndarray
        composantes principales.
    A : ndarray
        axes principales.
    (U,D,VT) : ndarray
        résultats de la svd.
    """
    
    Xcopy = X.copy()
    nk = 0
    sk = 1
    if (stand):
        Std = StandardScaler()
        Xcopy = Std.fit_transform(Xcopy)
        nk = Std.mean_
        sk = Std.scale_
    
    (U,D,VT) = np.linalg.svd(Xcopy, full_matrices = False)
    V = VT.T
    C = U.dot(np.diag(D)) #composantes principales
    A = V.dot(np.diag(D)) #axes principaux
    return C, A, (U,D,VT), nk, sk

def Inertie(D):
    """
    Barplot de l'inertie en %

    Parameters
    ----------
    D : ndarray
        Matrice diagonale de la svd.

    Returns
    -------
    None.

    """
    
    plt.figure()
    plt.bar(np.arange(np.shape(D)[0])+1, 100*D**2/sum(D**2))
    plt.title('Inerties en %')

def nbr_comp_inert(D,seuil = 95):
    """
    Nombre de composantes à prendre pour retrouver seuil % de l'information

    Parameters
    ----------
    D : ndarray
        Vecteur diagonal de SVD.
    seuil : int, optional
        Pourcentage. The default is 95.

    Returns
    -------
    res : int
        Pourcentage de l'information retrouvé.
    k : int
        Nombre de composantes pour retrouver ce pourcentage.

    """
    L =  100*D**2/sum(D**2)
    res = 0
    k = 0
    while (res < seuil):
        res += L[k]
        k += 1
    return res,k
    

def cercle_correlation(A, varbs, index1 = 0, index2 = 1):
    """
    Cree un cercle de correlation si on a des variables avec des noms

    Parameters
    ----------
    A : ndarray
        Axes principales de l'ACP.
    varbs : ndarray
        noms des variables.
    index1 : int, optional
        Axe principale index1. The default is 0.
    index2 : int, optional
        Axe principale index2. The default is 1.
    Returns
    -------
    None.

    """
    A1 = A[index1]
    A2 = A[index2]
    plt.figure()
    plt.title('Cercle des correlations')
    Z = np.linspace(-np.pi, np.pi, 256,endpoint=True)
    C,S = np.cos(Z), np.sin(Z)
    plt.plot(C,S,c='black',lw=.7)
    plt.axvline(c='black',ls='dashed',lw=1)
    plt.axhline(c='black',ls='dashed',lw=1)
    for i, txt in enumerate(varbs):
        plt.arrow(0,0,A1[i],A2[i], length_includes_head=True,
        head_width=0.025, head_length=.05)
        plt.annotate(txt, (A1[i]+.01,A2[i]+.01),fontsize=12)
    plt.xlabel('C1')
    plt.ylabel('C2')

