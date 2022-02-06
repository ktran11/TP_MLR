#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 17:44:34 2020

@author: Agathe Blanvillain & Kevin Tran

"""

import numpy as np
import matplotlib.pyplot as plt
# Adresse des deux fichiers de donnees
# https://perso.univ-rennes1.fr/bernard.delyon/tp/WindSpeed.csv
# https://perso.univ-rennes1.fr/bernard.delyon/tp/Hs.csv

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
n=len(y)

X = np.multiply(X,X)
from sklearn.linear_model import Ridge
alphas_Ridge =  10.**np.arange(-1,1,0.1) 

norbet=np.zeros(len(alphas_Ridge))
rmse=np.zeros(len(alphas_Ridge))
for j,a in enumerate(alphas_Ridge):
    reg = Ridge(alpha=a,normalize=True)
    reg.fit(X,y)
    norbet[j]=np.sqrt(np.sum(reg.coef_**2))
    rmse[j]=np.sqrt(np.mean((reg.predict(X)-y)**2))
    yfin = reg.predict(X)
    
plt.subplots()
plt.semilogx()
plt.xlabel(r'$\alpha$')
plt.legend()
plt.title('Ridge. Norme de de beta (rond) et RMSE (croix)')
plt.plot(alphas_Ridge,norbet,'r')
plt.plot(alphas_Ridge,norbet,'ro',label=r"$|\beta|$")
plt.legend(loc='center left')
plt.twinx()
plt.plot(alphas_Ridge,rmse,'b')
plt.plot(alphas_Ridge,rmse,'bx',label="RMSE")
plt.legend(loc='center right')
plt.show()

B = 100
ntest= n//2
mse_datasplitting = np.zeros(len(alphas_Ridge))
for b in range(B):
  per = np.random.permutation(X.shape[0])
  lt, la = per[:ntest], per[ntest:]
  Xa,ya =X[la,:], y[la]
  Xt,yt =X[lt,:], y[lt]
  for j,a in enumerate(alphas_Ridge):
    reg = Ridge(alpha=a,normalize=True)
    reg.fit(Xa,ya)
    yh = reg.predict(Xt)
    mse_datasplitting[j]+=np.mean((yh-yt)**2)
mse_datasplitting=mse_datasplitting/B
rmse_datasplitting=np.sqrt(mse_datasplitting)
plt.figure()
plt.suptitle(r'Ridge. RMSE par validation croisée pour divers $\alpha$.')
plt.semilogx()
plt.plot(alphas_Ridge,rmse_datasplitting,alphas_Ridge,rmse_datasplitting,'ro')
plt.show()
print('Minimum Ridge CV RMSE=',round(min(rmse_datasplitting),3))
rmseridge_datasplitting=rmse_datasplitting
print(r'Ridge. $\alpha$ qui minimise RMSE en VA:', alphas_Ridge[mse_datasplitting.argmin()])




# K folds
mse_Kfolds = np.zeros(len(alphas_Ridge))

K = 5
it= 100
ntest=np.around(len(y)/K).astype(int)

for i in range(it):
    per = np.random.permutation(X.shape[0])
    lt, la = per[:ntest], per[ntest:]
    Xa, Xt = X[la,:], X[lt,:]
    ya, yt = y[la], y[lt]
    for j,a in enumerate(alphas_Ridge):
      reg = Ridge(alpha=a,normalize=True)
      reg.fit(Xa,ya)
      yh = reg.predict(Xt)
      mse_Kfolds[j] += np.mean((yh-yt)**2)
mse_Kfolds = mse_Kfolds/it
rmse_Kfolds = np.sqrt(mse_Kfolds)
plt.figure()
plt.suptitle(r'Ridge. RMSE par Leave one out pour divers $\alpha$.')
plt.semilogx()
plt.plot(alphas_Ridge,rmse_Kfolds,alphas_Ridge,rmse_Kfolds,'ro')
plt.show()
print('Minimum Ridge LLO RMSE=',min(rmse))
rmseridge_Kfolds=rmse_Kfolds

print(r'$\alpha$ qui minimise RMSE:', alphas_Ridge[mse_Kfolds.argmin()])


plt.figure()
plt.suptitle(r'Ridge. RMSE par validation croisée pour divers $\alpha$.')
plt.semilogx()
plt.plot(alphas_Ridge,rmse_Kfolds)
plt.plot(alphas_Ridge,rmse_Kfolds,'r.',label='5-folds')
plt.plot(alphas_Ridge[rmse_Kfolds.argmin()],min(rmse_Kfolds),'ko', label = 'min 5 folds')
plt.plot(alphas_Ridge,rmse_datasplitting)
plt.plot(alphas_Ridge,rmse_datasplitting,'bx',label = 'datasplitting')
plt.plot(alphas_Ridge[rmseridge_datasplitting.argmin()],min(rmse_datasplitting),'kD', label = 'min datasplitting')
plt.xlabel(r'$\alpha$')
plt.ylabel('RMSE')
plt.legend()
plt.show()




from sklearn.linear_model import LassoLars


alphas_Lasso =  10.**np.arange(-4,-1,0.3) 
    
B = 100
ntest=int(n/2)
mse_lasso_dattasplitting = np.zeros(len(alphas_Lasso))
for b in range(B):
  per = np.random.permutation(X.shape[0])
  lt, la = per[:ntest], per[ntest:]
  Xa,ya =X[la,:], y[la]
  Xt,yt =X[lt,:], y[lt]
  for j,a in enumerate(alphas_Lasso):
    reg = LassoLars(alpha=a,normalize=True)
    reg.fit(Xa,ya)
    yh = reg.predict(Xt)
    mse_lasso_dattasplitting[j] += np.mean((yh-yt)**2)
mse_lasso_dattasplitting = mse_lasso_dattasplitting/B
rmse_lasso_dattasplitting = np.sqrt(mse_lasso_dattasplitting)
plt.figure()
plt.suptitle(r'Lasso. RMSE par validation croisée pour divers $\alpha$.')
plt.semilogx()
plt.plot(alphas_Lasso,rmse_lasso_dattasplitting,alphas_Lasso,rmse_lasso_dattasplitting,'ro')
plt.show()
print('Minimum Ridge CV RMSE=',min(rmse_lasso_dattasplitting))
print(r'Lasso. $\alpha$ qui minimise RMSE en VA:', alphas_Lasso[mse_lasso_dattasplitting.argmin()])


# K folds
mse_lasso_Kfolds = np.zeros(len(alphas_Lasso))

K = 5
it= 100
ntest=np.around(len(y)/K).astype(int)

for i in range(it):
    per = np.random.permutation(X.shape[0])
    lt, la = per[:ntest], per[ntest:]
    Xa, Xt = X[la,:], X[lt,:]
    ya, yt = y[la], y[lt]
    for j,a in enumerate(alphas_Lasso):
      reg = LassoLars(alpha=a,normalize=True)
      reg.fit(Xa,ya)
      yh = reg.predict(Xt)
      mse_lasso_Kfolds[j] += np.mean((yh-yt)**2)
mse_lasso_Kfolds = mse_lasso_Kfolds/it
rmse_lasso_Kfolds = np.sqrt(mse_lasso_Kfolds)
plt.figure()
plt.suptitle(r'Ridge. RMSE par 5-folds pour divers $\alpha$.')
plt.semilogx()
plt.plot(alphas_Lasso,rmse_lasso_Kfolds,alphas_Lasso,rmse_lasso_Kfolds,'ro')
plt.show()
print('Minimum Ridge 5-folds RMSE=',min(rmse_lasso_Kfolds))

print(r'$\alpha$ qui minimise RMSE:', alphas_Lasso[mse_lasso_Kfolds.argmin()])



plt.figure()
plt.suptitle(r'Lasso. RMSE par validation croisée pour divers $\alpha$.')
plt.semilogx()
plt.plot(alphas_Lasso,rmse_lasso_Kfolds)
plt.plot(alphas_Lasso,rmse_lasso_Kfolds,'r.',label='5-folds')
plt.plot(alphas_Lasso[rmse_lasso_Kfolds.argmin()],min(rmse_lasso_Kfolds),'ko', label = 'min 5 folds')
plt.plot(alphas_Lasso,rmse_lasso_dattasplitting)
plt.plot(alphas_Lasso,rmse_lasso_dattasplitting,'bx',label = 'datasplitting')
plt.plot(alphas_Lasso[rmse_lasso_dattasplitting.argmin()],min(rmse_lasso_dattasplitting),'kD', label = 'min datasplitting')
plt.xlabel(r'$\alpha$')
plt.ylabel('RMSE')
plt.legend()
plt.show()





nco=alphas_Lasso*0.
for j,a in enumerate(alphas_Lasso):
  reg = LassoLars(alpha=a,normalize=True)
  reg.fit(X,y)
  nco[j]=sum(abs(reg.coef_)>1.e-16)
  yh = reg.predict(X)
  mse[j] =np.mean((yh-y)**2)
plt.subplots()
plt.semilogx()
plt.title('Lasso. Nb de coeff non nuls (rond) et RMSE (croix)')
plt.plot(alphas_Lasso,nco,'r')
plt.plot(alphas_Lasso,nco,'ro',label=r"Nb$\ne$0")
plt.legend(loc='center left')
plt.twinx()
plt.plot(alphas_Lasso,rmse,'b')
plt.plot(alphas_Lasso,rmse,'bx',label="RMSE")
plt.legend(loc='center right')



