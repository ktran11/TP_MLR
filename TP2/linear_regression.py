#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 10:44:10 2022

@author: Agathe
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


etudiant = 18022495
np.random.seed(etudiant)

# Lecture des donnees
nomvar=np.loadtxt("data/Vent.csv",delimiter=',',dtype='str',max_rows=1)[1:]
X=np.loadtxt("data/Vent.csv",delimiter=',',skiprows=1,usecols=1+np.arange(len(nomvar)))
y=np.loadtxt("data/Vagues.csv",delimiter=',',skiprows=1,usecols=1)
n=len(y)




select=np.random.choice(n,size = 1000, replace=False)
Xselect = X[select,:]
yselect = y[select]

mod = LinearRegression()
mod.fit(X = Xselect ,y = yselect) #fait le calcul
yhat = mod.predict(Xselect)

plt.close()
plt.figure()
plt.plot(yselect, mod.predict(Xselect),'r+')#donne une idée de la variabilité des erreurs
plt.xlabel('Réponse')
plt.ylabel("Prédiction")

plt.plot(yselect, yselect, color = 'black')
plt.title('Droite de regression du modèle X avec 1000 individus')
   
# K-Fold
K = 5
it_Kfolds = 50
err_fold = 0.0
ntest = np.around(len(yselect)/K).astype(int)
for i in range(it_Kfolds):
    per = np.random.permutation(Xselect.shape[0])
    lt, la = per[:ntest], per[ntest:]
    Xa, Xt = Xselect[la,:], Xselect[lt,:]
    ya, yt = yselect[la], yselect[lt]
    reg = LinearRegression()
    reg.fit(Xa,ya)
    yh=reg.predict(Xt)
    err_fold += np.mean((yh-yt)**2)
    err_fold = np.sqrt(err_fold/it_Kfolds)

plt.savefig(fr'./image_rapport/reg_line_mod_X_1000_{int(round(err_fold,2)*100)}.png')
print(err_fold)

nind = 4000
# On extrait un sous-ensemble d'individus
select=np.random.choice(len(y),size=nind, replace=False)
Xselect = X[select,:]
yselect = y[select]
n = len(yselect)


lineaire_vent = Xselect
carre_vent = np.multiply(Xselect, Xselect)
cube_vent = np.multiply(carre_vent, Xselect)
quatre_vent = np.multiply(carre_vent, carre_vent)

model =  [lineaire_vent, carre_vent, cube_vent, quatre_vent]
nom_mod = ['X','$X^2$','$X^3$','$X^4$']
err_fold = np.zeros((len(model)))

it_Kfolds = 50
K = 5

for index, X_mod in enumerate(model): 
    mod = LinearRegression()
    mod.fit(X = X_mod ,y = yselect) #fait le calcul
    yhat = mod.predict(X_mod)
    sighat = np.sqrt(sum((yhat-yselect)**2)/len(yselect))
    plt.close()
    plt.figure()
    plt.plot(yselect, mod.predict(X_mod),'r+')#donne une idée de la variabilité des erreurs
    plt.xlabel('Réponse')
    plt.ylabel("Prédiction")
    plt.plot(yselect, yselect, color = 'black')
    plt.title(fr'Droite de regression du modèle {nom_mod[index]} avec {nind} individus')
    
    # K-Fold

    ntest = np.around(len(yselect)/K).astype(int)
    for i in range(it_Kfolds):
      per = np.random.permutation(X_mod.shape[0])
      lt, la = per[:ntest], per[ntest:]
      Xa, Xt = X_mod[la,:], X_mod[lt,:]
      ya, yt = yselect[la], yselect[lt]
      reg = LinearRegression()
      reg.fit(Xa,ya)
      yh=reg.predict(Xt)
      err_fold[index] +=np.mean((yh-yt)**2)
    err_fold[index] = np.sqrt(err_fold[index]/it_Kfolds)
    plt.savefig(fr'./image_rapport/reg_line_mod_{nom_mod[index]}_{nind}_{int(round(err_fold[index],2)*100)}.png')

    
for index, err in enumerate(err_fold):
    print(f'Erreur par {K}-folds avec le modèle {nom_mod[index]}: ',round(err, 2))
    

