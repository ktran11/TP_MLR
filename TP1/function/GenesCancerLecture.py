import ACP

import numpy as np 

labels=None
indiv=[]
varbs=None

etudiant = 18022495 # nombre à  remplacer par votre numéro d'etudiant
np.random.seed(etudiant)

# On lit la premiere ligne pour obtenir le nombre de colonnes
X = np.loadtxt("../data/data.csv",max_rows=1,delimiter=",",dtype=str)
nvars=len(X)-1

# Lecture des donnees correpondant aux nvars premieres variables
# et a la moitie des individus.

print("Sont lues les donnees correpondant aux nvars=",nvars,
      "premieres variables et a moitie des individus")
X = np.loadtxt("../data/data.csv",skiprows=1,delimiter=",",usecols=np.arange(nvars)+1)

y =np.loadtxt("../data/labels.csv",delimiter=",",skiprows=1,dtype=str)

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
print("nombre de composantes à prendre pour retrouver 90% de l'information : ",
      ACP.nbr_comp_inert(D,90))

X = X[:,:ACP.nbr_comp_inert(D,90)[1]]



