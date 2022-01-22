#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 11:00:28 2022

@author: Agathe
"""
import numpy as np
import matplotlib.pyplot as plt
# Analyse discriminante linéaire ----------------------------------------------
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import plot_confusion_matrix

def lineaire(X,y,validation = False):
    lda = LinearDiscriminantAnalysis()
    Xcopy = X.copy()
    ycopy = y.copy()
    if (validation):
        X = Xcopy
        y = ycopy
        err = 0
        for i in range (10):
            ntest=np.floor(len(y)//10).astype(int)#on prend 1/10 de l'ensemble pour réduire le nombre d'erreurs
            per=np.random.permutation(len(y))#permutation des nombres de 1 à 400
            lt,la=per[:ntest], per[ntest:]#lt = indices des premiers éléments entre 1 et 400 et la des derniers
            Xa,Xt=X[la,:],X[lt,:] 
            ya,yt=y[la],y[lt] 
            lda.fit(Xa,ya)
            yhat = lda.predict(Xt)      
            errl=sum(yt!=yhat)/len(yt)
            err += errl
        errm = err / 10
        return("Taux d'erreur : ",round(errm,3))
    lda.fit(X,y) # estimation
    yhat = lda.predict(X) #prédiction
    errl=sum(y!=yhat)/len(y) #somme des erreurs divisé par la longueur
    return("Taux d'erreur: ",round(errl,3)) #on imprime le taux d'erreur (on se trompe environ 1 fois sur 4) 

def confusion_lineaire(X,y):
    lda = LinearDiscriminantAnalysis()
    lda.fit(X,y)
    plt.rcParams.update({'figure.figsize': (3,3),'font.size': 16})#changement des tailles des caractères
    plot_confusion_matrix(lda, X, y,cmap='YlOrBr',colorbar=False)  #cmap = colormap, colorbar pour avoir l'échelle
    plt.rcdefaults() 

def classes_lineaire(X,y):
    lda = LinearDiscriminantAnalysis()
    C = lda.fit_transform(X, y)# On a 2 classes donc 1 seul axe factoriel
    plt.figure()
    vlab = np.unique(y)
    for i, vl in enumerate(vlab):
        l = (y == vl)
        plt.scatter(C[l, 0], C[l, 1], s = 4, label = vl)
        plt.legend()
    

# Analyse discriminante quadratique ----------------------------------------------
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
def quadratique(X,y,validation = False):
    qda = QuadraticDiscriminantAnalysis()
    Xcopy = X.copy()
    ycopy = y.copy()
    if (validation):
        X = Xcopy
        y = ycopy
        err = 0
        for i in range (10):
            ntest=np.floor(len(y)//10).astype(int)#on prend 1/10 de l'ensemble pour réduire le nombre d'erreurs
            per=np.random.permutation(len(y))#permutation des nombres de 1 à 400
            lt,la=per[:ntest], per[ntest:]#lt = indices des premiers éléments entre 1 et 400 et la des derniers
            Xa,Xt=X[la,:],X[lt,:] 
            ya,yt=y[la],y[lt] 
            qda.fit(Xa,ya)
            yhat = qda.predict(Xt)      
            errl=sum(yt!=yhat)/len(yt)
            err += errl
        errm = err / 10
        return("Taux d'erreur : ",round(errm,3))
    qda.fit(X,y) # estimation
    yhat = qda.predict(X) #prédiction
    errl=sum(y!=yhat)/len(y) #somme des erreurs divisé par la longueur
    return("Taux d'erreur: ",round(errl,3)) #on imprime le taux d'erreur (on se trompe environ 1 fois sur 4) 

def confusion_quadratique(X,y):
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X,y)
    plt.rcParams.update({'figure.figsize': (3,3),'font.size': 16})#changement des tailles des caractères
    plot_confusion_matrix(qda, X, y,cmap='YlOrBr',colorbar=False)  #cmap = colormap, colorbar pour avoir l'échelle
    plt.rcdefaults() 

