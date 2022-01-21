# -*- coding: utf-8 -*-

# Adresse des deux fichiers de donnees
# https://perso.univ-rennes1.fr/valerie.monbet/MachineLearning/TCGA-PANCAN-HiSeq-801x20531/data.csv
# https://perso.univ-rennes1.fr/valerie.monbet/MachineLearning/TCGA-PANCAN-HiSeq-801x20531/labels.csv

import numpy as np
import matplotlib.pyplot as plt

labels=None
indiv=[]
varbs=None

print("N'oubliez pas de mettre votre numero d'etudiant") 
etudiant = 18022495 # nombre à  remplacer par votre numéro d'etudiant
np.random.seed(etudiant)

# On lit la premiere ligne pour obtenir le nombre de colonnes
X = np.loadtxt("data.csv",max_rows=1,delimiter=",",dtype=str)
nvars=len(X)-1

# Lecture des donnees correpondant aux nvars premieres variables
# et a la moitie des individus.
print("Vous pouvez choisir le nombre de variables nvars.")
nvars=200
print("Sont lues les donnees correpondant aux nvars=",nvars,
      "premieres variables et a moitie des individus",
      "(tiree aleatoirement sur la base de votre numero d'etudiant)")
X = np.loadtxt("data.csv",skiprows=1,delimiter=",",usecols=np.arange(nvars)+1)
nech=X.shape[0]//2
y =np.loadtxt("labels.csv",delimiter=",",skiprows=1,dtype=str)
per=np.random.permutation(X.shape[0])[:nech]
X,y = X[per,:], y[per,1]
print("Nombre de lignes, nombre de colonnes : ",X.shape)

# Elimination des variables constantes
l=np.std(X,axis=0)>1.e-8
X=X[:,l]
print("Nombre de lignes et colonnes, apres elimination des variables constantes: ",X.shape)


#ACP
""""
def stdise(X):
  #Routine de standardisation
   #  On pourrrait utiliser scikitlearn: Exemple:
    #   from sklearn.preprocessing import StandardScaler
     #  X=np.arange(12).reshape((4,3))
      # print(StandardScaler().fit_transform(X))
  
  Xs=X.astype(float)
  mk=np.mean(Xs,axis=0)
  # Calcul de l'ecart-type avec max pour eviter une division par 0
  sk=np.maximum(np.std(Xs,axis=0),10*np.finfo(float).eps)
  Xs=np.add(Xs,-mk)
  Xs=np.multiply(Xs,1/sk)
  return Xs, mk, sk
# SVD. Axes Composantes
# Apres standardisation les colonnes sont de norme "nb de ligne" et non 1,
# on corrige cela, pour avoir de meilleures echelles.
Xs, mk,sk=stdise(X)
Xs=Xs/np.sqrt(np.shape(X)[0])
(U,D,VT) = np.linalg.svd(Xs,full_matrices=False)
V=VT.T
# Premieres composantes principales
C1 = D[0]*U[:,0]
C2 = D[1]*U[:,1]
C3 = D[2]*U[:,2]
# Axes principaux modifies pour le cercle des correlations
A1 = D[0]*V[:,0]
A2 = D[1]*V[:,1]
# Graphiques
plt.close('all')
plt.figure()
plt.title('Representation des individus dans le plan (C1,C2)')
if y is None:
  plt.scatter(C1,C2)
else:
  vlab=np.unique(y)
  lv=len(vlab)
  #cols=['C0','C1','C2','C3','k'] # un choix de couleurs
  #cols=plt.cm.nipy_spectral(np.arange(lv)/lv) # un choix de couleurs
  for i,vl in enumerate(vlab):
    l=y==vl
    plt.scatter(C1[l],C2[l],s=47,label=vl)#,color=cols[i])
  plt.legend(title="maladies")
for i,nm in enumerate(indiv): plt.text(C1[i],C2[i],nm)
plt.xlabel('C1')
plt.ylabel('C2')

# Inerties
plt.figure() 
plt.bar(np.arange(np.shape(D)[0])+1,100*D**2/sum(D**2))
plt.title('Inerties en %')

# Cercle des correlations
if not varbs is None:
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
"""

#cluster
"""
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import  KMeans
from sklearn.preprocessing import StandardScaler

##### Exemple de programation de CAH #####
print("\n******* Classification Ascendante hiérarchique ******* \n")
X=StandardScaler().fit_transform(X)
# Calcul de l'arbre
M=linkage(X,method='ward',metric='euclidean')

# Tracé de l'arbre
seuil=70
plt.figure()
plt.title('CAH. Visualisation des classes au seuil de '+str(seuil))
d=dendrogram(M,labels=list(y),orientation='right',color_threshold=seuil)
print(np.round(M[:,2],2))

##### Récupération des groupes
groupes=fcluster(M,t=seuil,criterion='distance')
for k in range(1,np.max(groupes)+1):
    print('Classe '+str(k).ljust(3,' ')+': ', end='')
    print(*y[np.where(groupes==k)])

#### Décroissance des variances intraclasse
VI=np.cumsum(M[:,2]**2)/2
plt.figure()
plt.plot(np.arange(len(VI))+1,np.flip(VI,axis=0))
plt.xlabel("Nombre de classes")
plt.ylabel("Variance intraclasse")
if 1==0: # Autre figure possible
  plt.figure()
  plt.plot(np.arange(len(VI/max(VI)))+1,np.flip(VI/max(VI),axis=0))
  plt.xlabel("Nombre de classes")
  plt.ylabel("Variance intraclasse/variance totale")

print("\n******* Kmeans ******* \n")

# Comparaison avec les Kmeans
nclus=np.max(groupes)
k_means = KMeans(init='k-means++', n_clusters=nclus, n_init=10)
k_means.fit(X)
for k in range(nclus):
    print('Classe '+str(k+1).ljust(3,' ')+': ', end='')
    print(*y[np.where(k_means.labels_==k)])

print("\n******* Comparaison des inerties ******* \n")

print("Inertie Kmeans",nclus,"centres: ",k_means.inertia_)
print("Inertie CAH",nclus,"classes: ",VI[-nclus])

"""
#analyse discriminante
"""
# Analyse discriminante linéaire ----------------------------------------------
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix,plot_confusion_matrix

lda = LinearDiscriminantAnalysis() #pas d'argument car pas d'initialisation
lda.fit(X,y) # estimation
yhat = lda.predict(X) #prédiction
errl=sum(y!=yhat)/len(y) #somme des erreurs divisé par la longueur
print("Taux d'erreur: ",round(errl,3)) #on imprime le taux d'erreur (on se trompe environ 1 fois sur 4)
plt.rcParams.update({'figure.figsize': (3,3),'font.size': 16})#changement des tailles des caractères
plot_confusion_matrix(lda, X, y,cmap='YlOrBr',colorbar=False)  #cmap = colormap, colorbar pour avoir l'échelle
plt.rcdefaults() 
#l'espace de dimension 8 est découpé par un hyperplan et chaque élément appartient à une partie

C = lda.fit_transform(X, y)# On a 2 classes donc 1 seul axe factoriel
tmp = [C[np.where(y==0)[0]],C[np.where(y==1)[0]]]
plt.figure()
vlab = np.unique(y)
lv = len(vlab)
for i, vl in enumerate(vlab):
    l = (y == vl)
    plt.scatter(C[l, 0], C[l, 1], s = 4, label = vl)
plt.legend()



#Avec validation croisée 
#on retire des personnes dans les données et on regarde si les prédictions sont justes
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
print("Taux d'erreur moyen: ",round(errm,3))


# Analyse discriminante quadratique ----------------------------------------------
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
qda = QuadraticDiscriminantAnalysis() #pas d'argument car pas d'initialisation
qda.fit(X,y) # estimation
yhat = qda.predict(X) #prédiction
errl=sum(y!=yhat)/len(y) #somme des erreurs divisé par la longueur
print("Taux d'erreur: ",round(errl,3)) #on imprime le taux d'erreur (on se trompe environ 1 fois sur 4)
#plt.rcParams.update({'figure.figsize': (3,3),'font.size': 16})#changement des tailles des caractères
#plot_confusion_matrix(lda, X, y,cmap='YlOrBr',colorbar=False)  #cmap = colormap, colorbar pour avoir l'échelle
#plt.rcdefaults() 
#l'espace de dimension 8 est découpé par un hyperplan et chaque élément appartient à une partie
#en prenant l'ensemble entier, on pense que l'analyse quadratique est plus efficace mais en prenant l'analyse croisée, on se rend compte que ce n'est pas le cas
err = 0
for i in range (10):
    ntest=np.floor(len(y)//10).astype(int)
    per=np.random.permutation(len(y))#permutation des nombres de 1 à 400
    lt,la=per[:ntest], per[ntest:]#lt = indices des premiers éléments entre 1 et 400 et la des derniers
    Xa,Xt=X[la,:],X[lt,:] 
    ya,yt=y[la],y[lt] 
    qda.fit(Xa,ya)
    yhat = qda.predict(Xt)      
    errl=sum(yt!=yhat)/len(yt)
    err += errl
errm = err / 10
print("Taux d'erreur moyen: ",round(errm,3))
"""


