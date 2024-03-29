#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 11:00:28 2022

@author: Agathe
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import plot_confusion_matrix
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def Taux_erreur_analyse_discriminante(X, y, quadratic = False, validation = False, n = 10):
    """
    Taux d'erreur de l'analyse discriminante

    Parameters
    ----------
    X : DONNÉES.
    y : LABELS.
    quadratic = BOOLÉEN, optional
        True si analyse discriminant quadratique. The default is False.
    validation : BOOLÉEN, optional
        True si analyse avec validation croisée. The default is False.
    n : INT, optional
        Si validation est True alors le nombre d'itération est n.
    Returns
    -------
    None

    """
    if (quadratic):
        da = QuadraticDiscriminantAnalysis()
    else:
        da = LinearDiscriminantAnalysis()
    Xcopy = X.copy()
    ycopy = y.copy()
    if (validation):
        err_array = []
        err = 0
        for i in range (n):
            # on prend 1/10 de l'ensemble pour réduire le nombre d'erreurs
            ntest=np.floor(len(y)//10).astype(int)
            
            # mélange l'ensemble des données et la labels
            per=np.random.permutation(len(y))
            lt,la=per[:ntest], per[ntest:]
            
            # définition de l'ensemble d'apprentissage et de test
            Xa,Xt=Xcopy[la,:],Xcopy[lt,:] 
            ya,yt=ycopy[la],ycopy[lt] 
            
            da.fit(Xa,ya)
            yhat = da.predict(Xt)      
            errx = sum(yt != yhat)/len(yt)
            err_array += [errx]
            err += errx
        errm = err /n 
        return errm, err_array
    else: 
        da.fit(X,y)
        yhat = da.predict(X) 
        errl = sum(y != yhat) / len(y)
        return errl
    
    
def Matrice_confusion_analyse_discriminante(X, y, quadratic = False, validation = False, title = 'matrice de confusion'):
    """
    Matrice de confusion pour l'analyse discriminante 

    Parameters
    ----------
    X : DONNÉES.
    y : LABELS.
    quadratic = BOOLÉEN, optional
        True si analyse discriminant quadratique. The default is False.
    validation : BOOLÉEN, optional
        True si analyse avec validation croisée. The default is False.
    title : str, optional
        titre de la matrice

    Returns
    -------
    None.

    """
    Xcopy = X.copy()
    ycopy = y.copy()
    if (quadratic):
        da = QuadraticDiscriminantAnalysis()
    else:
        da = LinearDiscriminantAnalysis()
    if (validation):
        ntest=np.floor(len(y)//10).astype(int)
        per=np.random.permutation(len(y))
        lt,la=per[:ntest], per[ntest:]
            
        # définition de l'ensemble d'apprentissage et de test
        Xa,Xt=Xcopy[la,:],Xcopy[lt,:] 
        ya,yt=ycopy[la],ycopy[lt] 
        da.fit(Xa,ya)
    else:    
        da.fit(X,y)
    plt.rcParams.update({'figure.figsize': (3,3),'font.size': 10})
    plot_confusion_matrix(da, X, y, cmap='YlOrBr', colorbar=False)  
    plt.rcdefaults() 
    plt.title(title)


def Scatter_plot_analyse_discriminant(X, y, index1 = 0, index2 = 1):
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
    Xcopy, ycopy = X.copy(), y.copy()
    lda = LinearDiscriminantAnalysis()
    C = lda.fit_transform(Xcopy, ycopy)
    plt.figure()
    vlab = np.unique(y)
    for i, vl in enumerate(vlab):
        l = (y == vl)
        plt.scatter(C[l, index1], C[l, index2], s = 4, label = vl)
        plt.legend()
