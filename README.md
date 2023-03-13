# KeoClean

Bienvenue sur le projet Keoclean. Keoclean est une application Streamlit permettant de diagnostiquer l'état d'un dataset, de nettoyer le datatset et de l'enrichir avec de nouvelles variables.

## Instructions pour exécuter l'application sous Linux

**1. Cloner le repository et créer un environnement**

**2. Se positionner dans le dossier contenant le repository et installer les requirements**

**IMPORTANT:** Si des difficultés sont rencontrés à cette étape, commenter dans le fichier requirements la ligne `pywin32==300`

**3. Démarrer l'application**

`streamlit run main.py`

**Note:** la première fois, il sera demandé de renseigner son adresse mail, vous pouvez laisser ce champ vide et appuyer directement sur la touche Entrée.

Une page devrait s'ouvrir dans votre navigateur. Si ce n'est pas le cas, entrez dans votre navigateur web (Google Chrome ou Firefox) l'adresse : http://localhost:8501/

## Instructions pour éxécuter l'application sous Windows

**1. Cloner le repository**

`git clone https://github.com/HugoBroucke/Keoclean.git`

**2. Créer et activer l'environnement**

`virtualenv -p python3 keoclean-env`

`cd <chemin de l'environnement>\Scripts`

`activate.bat`

**3. Se positionner dans le dossier Keoclean et installer les librairies**

`cd <chemin vers la racine du depository>`

`pip install -r requirements.txt`

**4. Démarrer l'application**

`streamlit run main.py`

**Note:** la première fois, il sera demandé de renseigner son adresse mail, vous pouvez laisser ce champ vide et appuyer directement sur la touche Entrée.

Une page devrait s'ouvrir dans votre navigateur. Si ce n'est pas le cas, entrez dans votre navigateur web (Google Chrome ou Firefox) l'adresse : http://localhost:8501/

## Changer le thème

Le thème par défaut proposé respecte la charte graphique de Kaduceo. Pour le modifier, cliquer sur l'icône avec les 3 traits > Settings > Theme

## Changelog

### Update du 14/06

**Requirements mis à jour !! N'oubliez pas d'installer les dernières librairies**
- Général:
    - Ajout bouton pour réordonner les données (order by)
    - Ajout d'un boutonn pour modifier la valeur des cellules directement depuis le dataset
- Import des données:
    - Optimisation de la sélection des colonnes
    - Ajout d'un bouton pour sélectionner manuellement certaines lignes à importer
    - Utilisation de la librairie AgGrid pour sélectionner le dataset à importer (permet de faire des filtres)
- Enrichir à partir de variables numériques:
    - Ajout d'une fonction pour créer des formules custom (intégration d'une interface de code Python)
- Enrichir à partir de conditions:
    - Possibilité de créer un nouvelle variable qui hérite des valeurs d'une var existante si la condition n'est pas respecté
    - Ajout d'une fonctionnalité pour initialiser une nouvelle colonne
- Enrichir à partir de variable temporelle
    - Ajout d'une fonction pour calculer les différences entre deux dates

### Update du 28/05

- Possibilité de rentrer manuellement le format d'une date lors du renseignement des meta
- Possibilité de modifier les métadonnées de plusieurs variables en même temps (ie. avec les mêmes informations)
- Dans la page Traitement des données manquantes - traitement des lignes avec trop de données manquantes, ajout d'un graphique interactif pour voir les modalités des différentes variables des lignes concernées

### Update du 10/05

- Ajout de stats descriptives au moment de l'import des données (utile pour ne pas avoir à réaliser le Profiling afin de renseigner les métadonnées)
- Ajout de fonctions d'imputations pour le traitement des outliers et des données manquantes
- Correction d'un bug lié au colonne avec uniquement des données manquantes au niveau du Profiling

### Hotfix 04/05

- Résolution de problèmes page Profiling survenant lorsqu'une colonne du df est totalement vide

### Update du 29/04

- Possibilité de voir dans le profiling, si des variables définies comme unique ou booléenne dans les métadonnées sont en adéquation ou non avec leur définition dans les méta
- Dans la page "Traitement des valeurs incompatibles", ajout d'une fonction pour traiter les données qui ont des pattern exclus (possibilité de supprimer les lignes ou de remplacer la valeur exclue par une autre ou une donnée manquante)

### Hotfix 26/04

- Ajout d'un fichier gitignore et création du dossier tempProfiling

### Update du 23/04

- Ajout d'un lien dans la sidebar pour accéder à la documentation de l'outil (doc wiki)
- Ajout d'une fonction de concaténation dans la page "Enrichir à partir d'un autre dataset"
- Ajout de l'option "contient" et "valeur manquante" pour l'édition de contraintes

### Update du 16/04

- Le dataset des métadonnées s'initialise au moment du chargement des données
- Ajout d'une fonction permettant de créer ou de modifier une variable existante à l'aide d'une condition (Disponible à la page "Enrichir à partir de conditions")
- Désormais, à chaque modification effectuée sur le dataset, une table s'incrémente avec la liste des modifications. Il est possible de visualiser cette table en cliquant sur "Afficher le rapport des modifications effectuées" (dans la sidebar)
    - Ajout de la possibilité de charger le dataset à une version donnée (ie. Annuler certaines modifications effectuées) : cela annulera toute les modifications apportées au dataset postérieure à la version chargée
    - Ajout d'une fonctionnalité permettant d'exporter ce rapport au format PDF

### Update du 13/04

Release de la V1 de l'application + mise à disposition du repository pour toute l'équipe
