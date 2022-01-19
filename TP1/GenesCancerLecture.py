# -*- coding: utf-8 -*-

# Adresse des deux fichiers de donnees
# https://perso.univ-rennes1.fr/valerie.monbet/MachineLearning/TCGA-PANCAN-HiSeq-801x20531/data.csv
# https://perso.univ-rennes1.fr/valerie.monbet/MachineLearning/TCGA-PANCAN-HiSeq-801x20531/labels.csv

import numpy as np

etudiant = 100 # nombre à  remplacer par votre numéro d'etudiant
np.random.seed(etudiant)

# On lit la premiere ligne pour obtenir le nombre de colonnes
X = np.loadtxt("./data/data.csv",max_rows=1,delimiter=",",dtype=str)
nvars=len(X)-1

# Lecture des donnees correpondant aux nvars premieres variables
# et a la moitie des individus.
print("Vous pouvez choisir le nombre de variables nvars.")
nvars=200
print("Sont lues les donnees correpondant aux nvars=",nvars,
      "premieres variables et a moitie des individus",
      "(tiree aleatoirement sur la base de votre numero d'etudiant)")
X = np.loadtxt("./data/data.csv",skiprows=1,delimiter=",",usecols=np.arange(nvars)+1)
nech=X.shape[0]//2
y =np.loadtxt("./data/labels.csv",delimiter=",",skiprows=1,dtype=str)
per=np.random.permutation(X.shape[0])[:nech]
X,y = X[per,:], y[per,1]
print("Nombre de lignes, nombre de colonnes : ",X.shape)

# Elimination des variables constantes
l=np.std(X,axis=0)>1.e-8
X=X[:,l]
print("Nombre de lignes et colonnes, apres elimination des variables constantes: ",X.shape)


