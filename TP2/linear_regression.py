#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 10:44:10 2022

@author: Agathe
"""
import numpy as np


etudiant = 18022495
np.random.seed(etudiant)

# Lecture des donnees
nomvar=np.loadtxt("data/Vent.csv",delimiter=',',dtype='str',max_rows=1)[1:]
X=np.loadtxt("data/Vent.csv",delimiter=',',skiprows=1,usecols=1+np.arange(len(nomvar)))
y=np.loadtxt("data/Vagues.csv",delimiter=',',skiprows=1,usecols=1)
n=len(y)


# On extrait un sous-ensemble d'individus
select=np.random.choice(n,size=3000, replace=False)
X=X[select,:]
y=y[select]
n=len(y)

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


#Calcul manuel avec la formule  classique
X1=np.append(np.ones((len(y),1)),X,1)
Xt=X1.T
Ri=np.linalg.inv(Xt.dot(X1))
bet=Ri.dot((Xt).dot(y))#beta = (XTX)-1.XT.y
yhat=X1.dot(bet)#yhat = X1.beta
yhat

plt.close()
plt.plot(y,yhat,'r+')
plt.xlabel('Reponse')
plt.ylabel("Prediction")
plt.plot(y,y,color='black')
plt.title('Droite de regression')
plt.show()

mod = LinearRegression()
mod.fit(X=X,y=y) #fait le calcul
yhat = mod.predict(X)
sighat = np.sqrt(sum((yhat-y)**2)/len(y))
plt.close()
plt.plot(y,mod.predict(X),'r+')#donne une idée de la variabilité des erreurs
plt.xlabel('Réponse')
plt.ylabel("Prédiction")
plt.plot(y,y, color = 'black')
plt.show()
X1=np.append(np.ones((len(y),1)),X,1)
Xt=X1.T
Ri=np.linalg.inv(Xt.dot(X1))
np.sqrt(np.diag(Ri))*sighat

#### Calcul de l'erreur par validation croisée


# 5-Fold
err=0
it=20
ntest=np.around(len(y)/5).astype(int)
for i in range(it):
  per = np.random.permutation(X.shape[0])
  lt, la = per[:ntest], per[ntest:]
  Xa, Xt = X[la,:], X[lt,:]
  ya, yt = y[la], y[lt]
  reg = LinearRegression()
  reg.fit(Xa,ya)
  yh=reg.predict(Xt)
  err+=np.mean((yh-yt)**2)
err = np.sqrt(err/it)
print('Erreur par CV 1/5 = ',round(err, 2))


