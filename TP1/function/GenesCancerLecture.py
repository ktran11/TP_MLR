import ACP
import AnalyseDiscriminante as ad
import Clustering as cl
import matplotlib.pyplot as plt
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
r,k = ACP.nbr_comp_inert(D,90)

print("nombre de composantes à prendre pour retrouver 90% de l'information : ",
      k)

X = C[:,:k]

print(X.shape)
print(y)
### Analyse Discriminante
n = 100
errl = ad.Taux_erreur_analyse_discriminante(X,y)
errq = ad.Taux_erreur_analyse_discriminante(X,y,quadratic=True)
errqv, barq = ad.Taux_erreur_analyse_discriminante(X,y,quadratic=True,validation=True,n=n)
errlv, barl = ad.Taux_erreur_analyse_discriminante(X,y,validation=True,n=n)
print("taux d'erreur analyse discriminante linéaire: ", errl)
print("taux d'erreur analyse discriminante quadratique: ", errq)
print("taux d'erreur analyse discriminante quadratique avec validation: ", errqv)
print("taux d'erreur analyse discriminante linéaire avec validation: ", errlv)
# Barplot
plt.figure()
plt.boxplot(barq)
plt.figure()
plt.boxplot(barl)
#Matrice de confusion
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
sc01 = ad.Scatter_plot_analyse_discriminant(X, y, index1 = 0, index2 = 1)
sc02 = ad.Scatter_plot_analyse_discriminant(X, y, index1 = 0, index2 = 2)
sc12 = ad.Scatter_plot_analyse_discriminant(X, y, index1 = 1, index2 = 2)
sc01
sc02
sc12
