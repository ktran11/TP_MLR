#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 2 7:51:01 2022

@author: Agathe & Kevin
"""

import ACP
import Clustering as cl
import AnalyseDiscriminante as ad
import matplotlib.pyplot as plt
import numpy as np 

labels=None
indiv=[]
varbs=None

etudiant = 18022495 # numéro étudiant d'Agathe
np.random.seed(etudiant)

# On lit la premiere ligne pour obtenir le nombre de colonnes
X = np.loadtxt("data.csv",max_rows=1,delimiter=",",dtype=str)
nvars=len(X)-1

# Lecture des donnees correpondant aux nvars premieres variables
# et a la moitie des individus.

print("Sont lues les donnees correpondant aux nvars=",nvars,
      "premieres variables et a moitie des individus")
X = np.loadtxt("data.csv",skiprows=1,delimiter=",",usecols=np.arange(nvars)+1)

y =np.loadtxt("labels.csv",delimiter=",",skiprows=1,dtype=str)

y = y[:,-1]

# Elimination des variables constantes
l=np.std(X,axis=0)>1.e-8
print("Nombre de lignes et colonnes, apres elimination des variables constantes: ",
      X.shape)

C,A,svd = ACP.ACP_func(X) # sans standardisation
U, D, VT = svd
ACP.Inertie(D)

C,A,svd = ACP.ACP_func(X,True) # avec standardisation
U, D, VT = svd
ACP.Inertie(D)
r,k = ACP.nbr_comp_inert(D,90)

print("nombre de composantes à prendre pour retrouver 90% de l'information : ",
      k)


### Scatter plot ACP
vlab = list(set(y))
plt.figure()
for i,vl in enumerate(vlab):
   l=y==vl
   plt.scatter(C[l,1], C[l,2], s = 4, label = vl)
   plt.legend()



X = C[:,:k]

### Clustering
n_init = 10
errk = cl.Kmeans_func(X,y,n_init = n_init)
errkv,bark = cl.Kmeans_func(X, y, n_init = n_init,val=True, n = 200)

# Nuage de points, (indice 1 et 2 plus parlant)
sc12 = cl.Scatter_plot_k_means(X,y,index1=1,index2=2)
sc12


### Analyse Discriminante
n = 100
errl = ad.Taux_erreur_analyse_discriminante(X,y)
errq = ad.Taux_erreur_analyse_discriminante(X,y,quadratic=True)
errqv, barq = ad.Taux_erreur_analyse_discriminante(X,y,quadratic=True,validation=True,n=n)
errlv, barl = ad.Taux_erreur_analyse_discriminante(X,y,validation=True,n=n)

# Matrice de confusion
matl = ad.Matrice_confusion_analyse_discriminante(X, y, title = 'AD linéaire')
matq = ad.Matrice_confusion_analyse_discriminante(X, y, quadratic = True, 
                                                  title = 'AD quadratique')
matlv = ad.Matrice_confusion_analyse_discriminante(X, y, validation = True, 
                                                   title = 'AD linéaire avec validation')
matqv = ad.Matrice_confusion_analyse_discriminante(X, y, quadratic = True, 
                                                   validation = True, 
                                                   title = 'AD quadratique avec validation')
matl
matq
matlv
matqv

#nuage de points

sc12 = ad.Scatter_plot_analyse_discriminant(X, y, index1 = 1, index2 = 2)

sc12


# Barplot
plt.figure()

barl = np.array(barl)*100
barq = np.array(barq)*100
bark = np.array(bark)*100

bar = [bark,barl,barq]

plt.boxplot(bar, labels = ('K-Moyennes','AD Linéaire','AD Quadratique'),positions = np.array([0.3,1.5,2.7]))
plt.title('Boite à moustache des différentes erreurs pour chaque classification')
plt.ylabel('Taux d\'erreur en %')
plt.xlabel('Classification')

# Taux d'erreur
print("taux d'erreur k-moyennes: ", round(errk,3))
print("taux d'erreur analyse discriminante linéaire: ", round(errl,3))
print("taux d'erreur analyse discriminante quadratique: ", round(errq,3))
print("taux d'erreur k-moyennes avec validation: ", round(errkv,3))
print("taux d'erreur analyse discriminante quadratique avec validation: ", round(errqv,3))
print("taux d'erreur analyse discriminante linéaire avec validation: ", round(errlv,3))
