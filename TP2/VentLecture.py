#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 17:44:34 2020

@author: delyon

"""

import numpy as np

etudiant = 18022495
 # nombre à remplacer par votre numéro d'etudiant
np.random.seed(etudiant)

# Lecture des donnees
nomvar=np.loadtxt("data/Vent.csv",delimiter=',',dtype='str',max_rows=1)[1:]
X=np.loadtxt("data/Vent.csv",delimiter=',',skiprows=1,usecols=1+np.arange(len(nomvar)))
y=np.loadtxt("data/Vagues.csv",delimiter=',',skiprows=1,usecols=1)
n=len(y)

# On extrait un sous-ensemble d'individus
select=np.random.choice(n,size=n//8, replace=False)
X=X[select,:]
y=y[select]
n=n=len(y)