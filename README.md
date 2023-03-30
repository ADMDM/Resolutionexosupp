# Resolutionexosupp

#%% A.3
a = 2
b = 3
c = 4
d = (a-(b**(c-2)))/(1-((a*b-2)/3)+c)
print(d)

#%% A.4
name , surname = "SARAH" , "PACKMAN"
fullname = name + " " + surname 
print(" My sister's name is",fullname)
print(" My sister's name is {} ".format(fullname))

#%% B.5

age = [12 , 13 , 14 , 9 ,12 ,15 ,12 ,11 ,13, 14]
print(age)
#age du 3e au 6eme jeune 
print(age[2:6])
#age du 6eme au dernier jeune 
print(age[5:]) 
#age du 3e en partant de la fin de la liste jusqu au debut avec index negatif 
print(age[-3:])
#ajouter un element en fin de liste 
age.append(10)
print(age)
#corriger un age 

#Methode 1 
age[2]=13 #il est passe de 14 a 13
print(age)

#Methode 2
age[2]=age[2]-1
print(age)

#Methode 3 
age[2]-=1
print(age)
#
r = max(age) - min(age)
print(r)
print(sorted(age)) #Trier par odre
print(r)
#nombre de jeune ayant plus de 12 ans 
print(age.count(12))
print("The oldest is" +" "+ str(max(age)) + " and the youngest is"+" "+ str(min(age)))
print("The oldest is",max(age), "and the youngest is",min(age))
print("The oldest is {}".format(max(age)) + " and youngest is {}".format(min(age)))
#%% C.2
#rayon en metre 
radius = np.array([1.23 ,1.65 ,2.1,1.38,1.91])
print(radius[radius>1.5])
# aires des cercles 
areas= 2*pi*radius**2
print(areas)
print("The area of circle number 2 is {} squared meters".format(areas[2]))

#%% C.3

I0 = 10 # intensité initiale du courant en A
R = 1000 # résistance en ohms
C = 50e-6 # capacité en farads
t = 0.1 # temps en secondes

I_t = I0 * np.exp(-t/(R*C)) # calcul de l'intensité du courant à l'instant t

print("L'intensité du courant 0.1 secondes après avoir déconnecté la pile vaut {:.2f} A.".format(I_t)) #{:.2f} 2 decimale

#%% D 
#D.1
# Données météorologiques pour Louvain-La-Neuve et Ostende
mois = ['Janvier', 'Février', 'Mars', 
        'Avril','Mai', 'Juin',
        'Juillet', 'Août', 'Septembre', 
        'Octobre', 'Novembre', 'Décembre'
        ]
temp_lln = np.array([3.4, 4.1, 4.4, 6.6, 9.7, 13.5, 16.4, 18.5, 14.8, 11.1, 6.9, 4])
temp_ost = np.array([4.8, 5.3, 7.2, 9.3, 13.2, 16.2, 18.2, 18.1, 15.8, 12.1, 8.2, 5.8])
# Création du premier graphique de type scatter plot pour les températures à Louvain-La-Neuve tout au long de l'an
fig, ax = plt.subplots()
#scatter plot
ax.scatter(mois, temp_lln)
#titre aux axes 
ax.set(title='Températures à Louvain-La-Neuve en 2020', xlabel='Mois', ylabel='Température (°C)')
#Mettre les mois a l'horizontale
ax.set_xticklabels(mois, rotation=45, ha='right')
#Limites des axes des ordonnes 
ax.set_ylim([0, 25])

# Personnalisation des couleurs et du style des points
ax.scatter(mois, temp_lln, marker='x', color='black', linewidths=2, edgecolors='red')
# Création du deuxième graphique de type scatter plot pour les températures à Louvain-La-Neuve et Ostende
fig, ax = plt.subplots(figsize=(10, 8)) #applique a tout le graphe 

ax.scatter(mois, temp_lln, marker='x', color='black', label='Louvain-La-Neuve')
#Losange pour Ostende 
ax.scatter(mois, temp_ost, marker='D', color='red', label='Ostende')
ax.set(title='Températures en Belgique en 2020', xlabel='Mois', ylabel='Température (°C)')
ax.set_xticklabels(mois, rotation=45, ha='right')
ax.set_ylim([0, 25])
ax.legend(loc='upper right')


# Personnalisation des couleurs et du style des points
ax.plot(mois, temp_lln, marker='x', color='black', linewidths=2, edgecolors='red',)
ax.plot(mois, temp_ost, marker='D', color='red', linewidths=2, edgecolors='black')
plt.plot(mois)
plt.show()

#%% D.2

import numpy as np
import matplotlib.pyplot as plt

# Définition des paramètres du signal
A = 1       # Amplitude
f1 = 500    # Fréquence du premier signal en Hz
f2 = 1000   # Fréquence du deuxième signal en Hz
phi = 0     # Constante de phase

# Définition des paramètres du temps
t_min = 0   # Temps minimal en secondes
t_max = 0.01    # Temps maximal en secondes
N = 1000    # Nombre de points pour le graphe

# Génération des valeurs de temps
t = np.linspace(t_min, t_max, N)

# Calcul du signal 1 et du signal 2
y1 = A * np.sin(2 * np.pi * f1 * t + phi)
y2 = A * np.sin(2 * np.pi * f2 * t + phi)

# Calcul de la somme des deux signaux
y_sum = y1 + y2

# Création d'un graphe de type "line plot" pour le signal 1
fig, ax = plt.subplots()
ax.plot(t, y1)
ax.set_title('Signal à ' + str(f1) + ' Hz')
ax.set_xlabel('Temps (s)')
ax.set_ylabel('Tension (V)')
ax.set_xlim([t_min, t_max])
ax.grid(True)

# Sauvegarde du graphique en tant que fichier PNG
plt.savefig('signal_500.png', dpi=300)

# Création d'un graphe de type "line plot" pour le signal 2
fig, ax = plt.subplots()
ax.plot(t, y2)
ax.set_title('Signal à ' + str(f2) + ' Hz')
ax.set_xlabel('Temps (s)')
ax.set_ylabel('Tension (V)')
ax.set_xlim([t_min, t_max])
ax.grid(True)

# Sauvegarde du graphique en tant que fichier PNG
plt.savefig('signal_1000.png', dpi=300)

# Création d'un graphe de type "line plot" pour la somme des deux signaux
fig, ax = plt.subplots()
ax.plot(t, y_sum)
ax.set_title('Somme des signaux à ' + str(f1) + ' Hz et ' + str(f2) + ' Hz')
ax.set_xlabel('Temps (s)')
ax.set_ylabel('Tension (V)')
ax.set_xlim([t_min, t_max])
ax.grid(True)

# Sauvegarde du graphique en tant que fichier PNG
plt.savefig('somme.png', dpi=300)

# Affichage des graphiques
plt.show()

#%% D.3
import numpy as np
import matplotlib.pyplot as plt

# Définition des paramètres du circuit
R = 1000 # Résistance en ohms
I0 = 10 # Intensité initiale en ampères

# Définition des valeurs de capacité à tester
C1 = 10e-6 # 10 microfarads
C2 = 50e-6 # 50 microfarads
C3 = 100e-6 # 100 microfarads

# Définition des valeurs de temps pour le graphique
t = np.linspace(0, 0.5, 500) # 500 points de 0 à 0.5 s

# Calcul des courants pour chaque valeur de capacité
I1 = I0 * np.exp(-t / (R * C1))
I2 = I0 * np.exp(-t / (R * C2))
I3 = I0 * np.exp(-t / (R * C3))

# Création du graphique
fig, ax = plt.subplots()
ax.plot(t, I1, label='C1 = 10 $\mu$F')
ax.plot(t, I2, label='C2 = 50 $\mu$F')
ax.plot(t, I3, label='C3 = 100 $\mu$F')
ax.set_title('Décroissance du courant en fonction du temps')
ax.set_xlabel('Temps (s)')
ax.set_ylabel('Courant (A)')
ax.legend()

# Affichage du graphique
plt.show()



#%% E1 
# Création du dictionnaire 'communes'
communes = {'Anderlecht': 118241,
            'Etterbeek': 47414,
            'Forest': 55746,
            'Ixelles': 86244,
            'Evere': 4394,
            'Ottignies': 30283}

# Affichage des index du dictionnaire
print(communes.keys())

# Affichage du nombre d'habitants à Ixelles
print(f"Il y a {communes['Ixelles']} habitants à Ixelles.")

# Correction des données erronées
communes['Evere'] = 40394
del communes['Ottignies']

# Affichage du dictionnaire mis à jour
print(communes)

#%% E2
import pandas as pd

# Chargement des données
emissions = pd.read_csv('emissions co2.csv', index_col=0)

# Différence entre variable1 et variable2
variable1 = emissions['2000']  # variable1 est une Series, correspondant à la colonne '2000'
variable2 = emissions[['2000']]  # variable2 est un DataFrame, avec une seule colonne '2000'

# Affichage des émissions de CO2 de la Belgique pour les années 1990, 2000 et 2010
belgium = emissions.loc['Belgium', ['1990', '2000', '2010']]
print(belgium)

# Affichage des émissions de CO2 des 50e, 100e et 150e pays pour les années 1970 et 1971
emissions_50 = emissions.iloc[[49, 99, 149], [0, 1]]
print(emissions_50)

# Affichage des émissions de CO2 des 30 premiers pays pour les 20 dernières années
last_20_years = emissions.iloc[:, -20:]
top_30 = last_20_years.mean(axis=1).sort_values(ascending=False)[:30]
print(emissions.loc[top_30.index, last_20_years.columns])


#%% F1
# Création de la liste des jours de la semaine
semaine = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']

# Boucle pour parcourir les jours de la semaine et afficher les messages correspondants
for jour in semaine:
    if jour in ['Lundi', 'Mardi', 'Mercredi', 'Jeudi']:
        print(jour + " : Au travail")
    elif jour == 'Vendredi':
        print(jour + " : Chouette c'est vendredi")
    else:
        print(jour + " : Repos ce weekend")

#%% F2
# Création de la liste impairs
impairs = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]

# Création de la liste pairs en incrémentant chaque élément de 1
pairs = [n+1 for n in impairs]

# Affichage de la liste pairs
print(pairs)

#%% F3
seq = ["A", "C", "G", "T", "T", "A", "G", "C", "T", "A", "A", "C", "G"]
complement = []

for nucleotide in seq:
    if nucleotide == "A":
        complement.append("T")
    elif nucleotide == "T":
        complement.append("A")
    elif nucleotide == "C":
        complement.append("G")
    elif nucleotide == "G":
        complement.append("C")

print(complement)

#%% F4
for i in range(1, 10):
    print('*' * i)

#%% F5 
# Liste des points des étudiants
points = [1, 12, 5, 12, 13, 15, 7, 18, 14, 9, 10, 9, 16]

# Boucle pour mettre à jour les notes des étudiants en échec
for i in range(len(points)):
    if points[i] < 10:
        points[i] = 10

# Affichage de la liste mise à jour
print(points)

#exercice 5 
import pandas as pd

# Chargement des données
emissions = pd.read_csv('emissions_co2.csv', index_col=0)

# Émissions de CO2 de la Belgique en 2018
emissions_belgique_2018 = emissions.loc['Belgium', '2018']

# Structure conditionnelle
if emissions_belgique_2018 > 10:
    print('La Belgique émet beaucoup de CO2')
elif emissions_belgique_2018 >= 5:
    print('Les émissions de CO2 de la Belgique sont moyennes')
else:
    print('La Belgique émet moins de CO2 que la moyenne mondiale')

# Sélection des pays selon les critères
pays_sel = emissions[(emissions['2018'] > 20) | (emissions['2018'] < 0.05)]

# Affichage des émissions pour les pays sélectionnés
print(pays_sel)

# Données des 5 dernières années triées sur base de 2018
pays_sel_last5 = pays_sel.iloc[:, -5:].sort_values(by='2018')
print(pays_sel_last5)

# F6
import pandas as pd

# Chargement des données
emissions = pd.read_csv('emissions_co2.csv', index_col=0)

# Affichage des émissions par personne de la Belgique pour chaque année
for year in emissions.columns:
    belgium_emissions = emissions.loc['Belgium', year]
    print(f"La Belgique a émis {belgium_emissions} tonnes de CO2 par personne en {year}.")

# Correction des émissions supérieures à 50 pour l'année 1970
for country in emissions.index:
    if emissions.loc[country, '1970'] > 50:
        emissions.loc[country, '1970'] = 50

# Ajout de la colonne MAX contenant la valeur maximale d'émissions de chaque pays depuis 1970
emissions['MAX'] = emissions.loc[:, '1970':].max(axis=1)

# Affichage du dataframe avec la colonne MAX ajoutée
print(emissions)




