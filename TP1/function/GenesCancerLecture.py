import ACP
import AnalyseDiscriminante as ad
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


### Scatter plot ACP
vlab = list(set(y))
for i,vl in enumerate(vlab):
   l=y==vl
   plt.scatter(C[l,1], C[l,2], s = 4, label = vl)
   plt.legend()



X = C[:,:k]

### Analyse Discriminante
n = 500
errl = ad.Taux_erreur_analyse_discriminante(X,y,n=n)
errq = ad.Taux_erreur_analyse_discriminante(X,y,quadratic=True,n=n)
errqv, barq = ad.Taux_erreur_analyse_discriminante(X,y,quadratic=True,validation=True,n=n)
errlv, barl = ad.Taux_erreur_analyse_discriminante(X,y,validation=True,n=n)



### Clustering
import Clustering as cl
n_init = 10
cl.Kmeans_func(X,y,n_init=n_init)
errkv,bark = cl.Kmeans_func(X, y,n_init=n_init,val=True, n = 200)


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


