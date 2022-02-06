#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 14:02:31 2022

@author: Agathe Blanvillain & Kevin Tran

"""

import numpy as np
import matplotlib.pyplot as plt
import warnings 
# Adresse des deux fichiers de donnees
# https://perso.univ-rennes1.fr/bernard.delyon/tp/WindSpeed.csv
# https://perso.univ-rennes1.fr/bernard.delyon/tp/Hs.csv

warnings.filterwarnings('ignore', category=DeprecationWarning)
etudiant = 18022495  # nombre à remplacer par votre numéro d'etudiant
np.random.seed(etudiant)

# Lecture des donnees
nomvar=np.loadtxt("data/Vent.csv",delimiter=',',dtype='str',max_rows=1)[1:]
X=np.loadtxt("data/Vent.csv",delimiter=',',skiprows=1,usecols=1+np.arange(len(nomvar)))
y=np.loadtxt("data/Vagues.csv",delimiter=',',skiprows=1,usecols=1)
n=len(y)

############################ Regression linéaire ############################
from sklearn.linear_model import LinearRegression

######### Pour 1000 individus #########
select=np.random.choice(n,size = 1000, replace=False)
Xselect = X[select,:]
yselect = y[select]

mod = LinearRegression()
mod.fit(X = Xselect ,y = yselect) 
yhat = mod.predict(Xselect)
plt.close()
plt.figure()
plt.plot(yselect, mod.predict(Xselect),'r+') #donne une idée de la variabilité des erreurs
plt.xlabel('Réponse')
plt.ylabel("Prédiction")
plt.plot(yselect, yselect, color = 'black')
plt.title('Droite de regression du modèle X avec 1000 individus')

######### K-Fold #########
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
print('Erreur par 5-folds avec le modèle X pour 1000 individus :', err_fold)


######### Pour 4000 individus #########
nind = 4000
# On extrait un sous-ensemble d'individus #
select=np.random.choice(len(y), size=nind, replace=False)
Xselect = X[select,:]
yselect = y[select]


lineaire_vent = Xselect
carre_vent = np.multiply(Xselect, Xselect)
cube_vent = np.multiply(carre_vent, Xselect)
quatre_vent = np.multiply(carre_vent, carre_vent)

model =  [lineaire_vent, carre_vent, cube_vent, quatre_vent]
nom_mod = ['X','$X^2$','$X^3$','$X^4$']
err_fold = np.zeros((len(model)))

it_Kfolds = 30
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
    plt.show()
    
    ########## K-Fold #########
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

######### print rmse #########
for index, err in enumerate(err_fold):
    print(f'Erreur par {K}-fold avec le modèle {nom_mod[index]}: ', round(err,2))

############# Retour 1000 individus #############
select=np.random.choice(n,size = 1000, replace=False)
X = X[select,:]
y = y[select]
n = len(y)

################################### Ridge ###################################
from sklearn.linear_model import Ridge
alphas_Ridge =  10.**np.arange(-1,1,0.1) 


######### beta et RMSE contre alpha #########
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
plt.savefig(r'./image_rapport/$\beta$_RMSE_by_$\alpha$.png')

######### datasplitting #########
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
plt.savefig(r'./image_rapport/Ridge_datasplitting_$\alpha$_RMSE.png')

######### K folds #########
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
plt.savefig(fr'./image_rapport/Ridge_{K}-fold_$\alpha$_RMSE.png')

######### Comparaison datasplitting et K-fold #########
plt.figure()
plt.suptitle(r'Ridge. RMSE par validation croisée pour divers $\alpha$.')
plt.semilogx()
plt.plot(alphas_Ridge,rmse_Kfolds)
plt.plot(alphas_Ridge,rmse_Kfolds,'r.',label='5-folds')
plt.plot(alphas_Ridge[rmse_Kfolds.argmin()],min(rmse_Kfolds),'ko', label = 'min 5 folds')
plt.plot(alphas_Ridge,rmse_datasplitting)
plt.plot(alphas_Ridge,rmse_datasplitting,'bx',label = 'datasplitting')
plt.plot(alphas_Ridge[rmse_datasplitting.argmin()],min(rmse_datasplitting),'kD', label = 'min datasplitting')
plt.xlabel(r'$\alpha$')
plt.ylabel('RMSE')
plt.legend()
plt.show()
plt.savefig(f'./image_rapport/Ridge_comp_datasplit_{K}-fold.png')


################################### Lasso ###################################
from sklearn.linear_model import LassoLars

alphas_Lasso =  10.**np.arange(-4,-1,0.3) 
######### datasplitting #########
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
plt.savefig(r'./image_rapport/Lasso_datasplitting_$\alpha$_RMSE.png')

######### K fold #########
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
plt.savefig(fr'./image_rapport/Lasso_{K}-fold_$\alpha$_RMSE.png')

######### Comparaison datasplitting et K-fold #########
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
plt.savefig(f'./image_rapport/Lasso_comp_datasplit_{K}-fold.png')

######### Nbr de variable non nul contre RMSE #########
mse = np.zeros(len(alphas_Lasso))
nco=alphas_Lasso*0.
for j,a in enumerate(alphas_Lasso):
  reg = LassoLars(alpha=a,normalize=True)
  reg.fit(X,y)
  nco[j]=sum(abs(reg.coef_)>1.e-16)
  yh = reg.predict(X)
  mse[j] = np.mean((yh-y)**2)
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
plt.savefig('./image_rapport/Lasso_nbrvariable_rmse.png')



