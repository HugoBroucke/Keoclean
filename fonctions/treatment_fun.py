import fonctions.general_fun as gf
import pandas as pd
import recordlinkage
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
import unidecode
import stop_words
import re
import spacy
import string
import streamlit as st

#https://github.com/J535D165/recordlinkage
def add_variable_to_key(compare_cl, nb_keys_rl, dataset, variable, method='levenshtein', threshold=40):
    """
    Permet d'ajouter une clé à l'algorithme de record linkage.
    
    Input : dataset, variable à ajouter comme clé, méthode utilisée (uniquement pour les variables textuelles) <'jaro',
    'jarowinkler', 'levenshtein', 'damerau_levenshtein', 'qgram','cosine', 'smith_waterman', 'lcs'>, le seuil de
    comparaison à partir duquel on considère une modalité identique à une autre selon la méthode choisie (uniquement pour les
    variables textuelles)
    Output : objet compare_cl implémenté par la clé
    """
    data_type = gf.return_type(dataset, variable)
    
    if data_type == 'object':
        compare_cl.string(variable, variable, method=method, threshold=threshold/100, label=variable)
    
    elif (data_type == 'float64') | (data_type == 'Int64'):
        compare_cl.numeric(variable, variable, label=variable)
        
    elif data_type == 'datetime64[ns]':
        compare_cl.date(variable, variable, label=variable)
    
    nb_keys_rl = nb_keys_rl+1
        
    return compare_cl, nb_keys_rl

def eliminate_records_identified_as_duplicates(dataset, dataset_blocks):
    """
    Permet de supprimer les records identifiés comme doublons.
    
    Input : dataset, liste des index à supprimer (output de la fonction compute_record_linkage)
    Output : dataset avec les lignes identifiées comme doublon supprimées
    """
    idx_list_to_del = []
    for idx in dataset_blocks.index:
        if dataset_blocks.at[dataset_blocks.index[idx], 'master'] == False :
            idx_list_to_del.append(dataset_blocks.at[dataset_blocks.index[idx], 'index'])
    
    dataset = dataset.drop(idx_list_to_del)
    
    return dataset

def compute_record_linkage(compare_cl, nb_keys_rl, dataset, block=None, min_score=80, classifier_method='naive'):
    """
    Permet de faire tourner l'agorithme record linkage.
    
    Input : dataset, variable sur laquelle on décide de bloquer (optionnel, recommandé pour de large datasets), score minimal
    global à atteindre pour considérer que deux records sont identique <doit être entre 0 et 1> (seulement pour la méthode de
    classifciation naive), méthode de classification utilisée <naive, ECM>
    Output : tableau regroupant toutes les paires dont la valeur de match est supérieure au score minimal, liste des groupes
    formés ainsi que les index des records appartenant à chaque groupe, liste des records à supprimer, liste des records
    conservés ainsi que le numéro du groupe
    """
    indexer = recordlinkage.Index()
    indexer.full()
    
    if not block == False:
        indexer = recordlinkage.BlockIndex(on=block)
        
    
    candidate_links = indexer.index(dataset)

    features = compare_cl.compute(candidate_links, dataset)
    
    if classifier_method == 'ECM':
        ecm = recordlinkage.ECMClassifier()
        matches_idx = ecm.fit_predict(features)
        groups = return_matches_groups(matches_idx)
        idx_list_to_del, idx_list_to_keep = get_idx_to_remove_or_keep(groups)
        
        #On reset les deux variables globales
        compare_cl = recordlinkage.Compare()
        nb_keys_rl = 0  
    
    elif classifier_method == 'naive' : #Méthode naive
        min_score = (min_score*nb_keys_rl)/100
        matches = features[features.sum(axis=1) >= min_score]
        groups = return_matches_groups(matches.index)
        idx_list_to_del, idx_list_to_keep = get_idx_to_remove_or_keep(groups)
        
        #On reset les deux variables globales
        compare_cl = recordlinkage.Compare()
        nb_keys_rl = 0
        
    df_blocks = pd.DataFrame(columns=['groupe', 'index', 'master'])
    df_blocks2 = dataset.copy().drop(dataset.index)
    for group in groups:
        for record in group[1]:
            if record in idx_list_to_del:
                master = False
            else: 
                master = True
            df_blocks = df_blocks.append({'groupe':group[0], 'index':record, 'master': master}, ignore_index=True)
            df_blocks2 = df_blocks2.append(dataset.loc[record], ignore_index=True)

    df_rl_merged = pd.merge(df_blocks, df_blocks2, how='outer', right_index=True, left_index=True)
    df_rl_merged = df_rl_merged.sort_values(by=['groupe','master'], ascending=[True, False])
        
    return df_rl_merged

#Utilisée pour retourner la liste des appariemments réalisés : :!\ input = les index des paires !! (matches.index)
def return_matches_groups(matches_idx) : 
    l = [] #Liste contenant tous les appariemment
    grp_name = 0 #Nom d'un nouveau groupe
    cpt = -1 #Compteur pour chaque paires

    for idx in matches_idx:
        cpt = cpt+1
        if cpt == 0: #On initialise la première paire
            l.append(['grp_0', matches_idx[0], (1,1)])
            continue

        else:
            key_list = [] #Liste des index déjà groupés
            for val in range(len(l)):
                for tup_idx in range(len(l[val][1])):
                    key_list.append(l[val][1][tup_idx])
                    key_list.append(l[val][1][tup_idx])

            if (matches_idx[cpt][0] not in key_list) & (matches_idx[cpt][1] not in key_list):
                grp_name = grp_name+1
                l.append(['grp_'+str(grp_name), matches_idx[cpt], (1,1)])             
                continue

            elif (matches_idx[cpt][0] in key_list) & (matches_idx[cpt][1] not in key_list):
                val_in = matches_idx[cpt][0]
                val_out = matches_idx[cpt][1]
                for a in range(len(l)): #On boucle sur les index des groupes formés
                    for t in range(len(l[a][1])): #On boucle sur le premier tuple représentant les index des lignes apparéiées        
                        if l[a][1][t] != val_in:
                            continue
                        else: #On apparie l'index sans groupe au groupe de l'index avec lequel il est apparéié
                            l[a][1] = l[a][1]+(val_out,)
                            l[a][2] = l[a][2]+(1,)
                            for j in range(len(l[a][1])):
                                if l[a][1][j] != val_in:
                                    continue
                                else:
                                    l[a][2] = list(l[a][2]) #Obligé de passer par des listes pour incrémenter des valeurs
                                    l[a][2][j] = l[a][2][j]+1 #On incrémente de +1 l'occurence
                                    l[a][2] = tuple(l[a][2]) #On repasse sous forme de tuple

            elif (matches_idx[cpt][1] in key_list) & (matches_idx[cpt][0] not in key_list):
                val_in = matches_idx[cpt][1]
                val_out = matches_idx[cpt][0]
                for a in range(len(l)): #On boucle sur les index des groupes formés
                    for t in range(len(l[a][1])): #On boucle sur le premier tuple représentant les index des lignes apparéiées        
                        if l[a][1][t] != val_in:
                            continue
                        else: #On apparie l'index sans groupe au groupe de l'index avec lequel il est apparéié
                            l[a][1] = l[a][1]+(val_out,)
                            l[a][2] = l[a][2]+(1,)
                            for j in range(len(l[a][1])):
                                if l[a][1][j] != val_in:
                                    continue
                                else:
                                    l[a][2] = list(l[a][2]) #Obligé de passer par des listes pour incrémenter des valeurs
                                    l[a][2][j] = l[a][2][j]+1 #On incrémente de +1 l'occurence
                                    l[a][2] = tuple(l[a][2]) #On repasse sous forme de tuple

            else:
                val1 = matches_idx[cpt][0]
                val2 = matches_idx[cpt][1]
                for b in range(len(l)):
                    for tup_ix in range(len(l[b][1])):
                        if l[b][1][tup_ix] == val1:
                            l[b][2] = list(l[b][2])
                            l[b][2][tup_ix] = l[b][2][tup_ix]+1
                            l[b][2] = tuple(l[b][2])
                            continue
                        
                        elif l[b][1][tup_ix] == val2:
                            l[b][2] = list(l[b][2])
                            l[b][2][tup_ix] = l[b][2][tup_ix]+1
                            l[b][2] = tuple(l[b][2])
                            continue

    return l

def get_idx_to_remove_or_keep(groups):
    idx_list_to_del = []
    idx_list_to_keep = []
    for index in range(len(groups)):
        max_val = 0
        for s in range (len(groups[index][2])) :
            val = groups[index][2][s]
            if max_val >= val:
                idx_list_to_del.append(groups[index][1][s])
            else :
                if max_val == 0:
                    max_val = val
                    idx_to_keep = groups[index][1][s]
                else:
                    idx_to_keep = groups[index][1][s]
                    idx_list_to_del.append(groups[index][1][s-1])
        idx_list_to_keep.append([groups[index][0], idx_to_keep])
    
    return idx_list_to_del, idx_list_to_keep        

def fill_missing_univariate(dataset, variable, method='value', value=None, window=10, min_periods=5):
    serie = dataset[variable]
    
    if method=='value':
        serie = serie.fillna(value)
        
    elif method=='forward_fill':
        serie = serie.fillna('ffill')
        
    elif method=='backward_fill':
        serie = serie.fillna('bfill')
        
    elif method=='mean':
        serie = serie.fillna(serie.mean())
    
    elif method=='median':
        serie = serie.fillna(serie.median())
    
    elif method=='mode':
        serie = serie.fillna(serie.mode())
        
    elif method=='rolling_avg':
        serie = serie.fillna(serie.rolling(window=window, min_periods=min_periods).mean())
        
    dataset[variable] = serie
    
    return dataset

def impute_outliers(dataset, variable, method='mean'):
    serie = dataset[variable]
        
    if method=='mean':
        serie = serie.mean()
    
    elif method=='median':
        serie = serie.median()
    
    elif method=='mode':
        serie = serie.mode()
    
    return serie

def get_clusters(dataset, var, ngram=(1,1), n_clus=None):
    
    if type(n_clus) is int:
        n_clus = [n_clus]
    
    dataset_copy = dataset.loc[:,[var]].reset_index(drop=True)
    lst = dataset_copy[var].unique().astype('U')
    X = TfidfVectorizer(analyzer='char', ngram_range=ngram).fit_transform(lst)
    sil_score_max = -1 #this is the minimum possible score
    #On cap à 10 000 le sample size du score silhouette dans le cas où le dataset est volumineux
    if len(lst) > 10000:
        sample_size = 10000
    else:
        sample_size = None

    #Get optimal number of clusters
    for n_clusters in range(2, len(lst)):
        model = KMeans(n_clusters = n_clusters).fit(X)
        labels = model.fit_predict(X)
        sil_score = silhouette_score(X, labels, sample_size = sample_size)
        if sil_score > sil_score_max:
            sil_score_max = sil_score
            best_n_clusters = n_clusters
    
    #Création du modèle optimal
    opti = dataset_copy[var].values.astype('U')
    X_opti = TfidfVectorizer(analyzer='char', ngram_range=ngram).fit_transform(opti)
    model_opti = KMeans(n_clusters = best_n_clusters).fit(X_opti)
    #labels_opti = model_opti.fit_predict(X_opti)
    
    #Création du df transitoire
    df_transitoire = dataset_copy.loc[:,[var]]
    df_transitoire['cluster'] = np.nan
    df_transitoire['count'] = 1

    #Clustering des observations
    for cluster in range(best_n_clusters):
        for i, label in enumerate(model_opti.labels_):
            if label == cluster:
                df_transitoire.at[i, 'cluster'] = int(label)
    
    #Création du dataset qui servira a remplacer par la variable nettoyé
    df_transitoire.reset_index(inplace=True)
    df_tomerge = df_transitoire.loc[:,['index', 'cluster']]
    df_transitoire.drop(columns=['index'], inplace=True)
            
    #clusters = dataset to display to the user
    clusters = df_transitoire.groupby(['cluster',var]).count().sort_values(['cluster','count'], ascending=[True, False])
    df_transitoire = clusters.reset_index()
    
    return df_transitoire, df_tomerge

def replace_with(dataset, var, df_filtered, method='most_representative', value=None):
    if method == 'most_representative':
        df_filtered_new_ix = df_filtered.reset_index(drop=True)
        list_to_change=[]
        for ix in df_filtered_new_ix.index:
            if ix == 0:
                master = df_filtered_new_ix.at[ix,var]
            else:
                #Si même cluster, on delete
                if df_filtered_new_ix.at[ix,'cluster'] == df_filtered_new_ix.at[ix-1, 'cluster']: 
                    list_to_change.append(df_filtered_new_ix.at[ix,var])
                #Si cluster différent, on modifie le dataset et on 
                else :
                    dataset[var] = np.where(dataset[var].isin(list_to_change), master, dataset[var])
                    list_to_change=[]
                    master = df_filtered.at[ix,var]
            if ix == len(df_filtered_new_ix) - 1:
                dataset[var] = np.where(dataset[var].isin(list_to_change), master, dataset[var])
    
    elif method == 'custom':
        dataset[var] = np.where(dataset[var].isin(df_filtered[var].values), value, dataset[var])
    
    return dataset

def harmonisation_str(dataset, var, fn, pattern_out='', pattern_in=''):
    if fn == 'Uppercase':
        dataset[var] = dataset[var].str.upper()
    
    elif fn == 'Lowercase':
        dataset[var] = dataset[var].str.lower()
        
    elif fn == 'Proppercase':
        dataset[var] = dataset[var].str.title()
        
    elif fn == 'Enlever les accents':
        dataset[var] = dataset[var].apply(lambda x : unidecode.unidecode(str(x)))
        
    elif fn == 'Enlever la ponctuation':
        dataset[var] = dataset[var].str.translate(str.maketrans('', '', string.punctuation))
        
    elif fn == 'Enlever les espaces de début et de fin':
        dataset[var] = dataset[var].str.strip()
        
    elif fn == 'Remplacer un pattern':
        dataset[var] = dataset[var].str.replace(pattern_out, pattern_in)
        
    elif fn == 'Standardiser le texte':
        modele_spacy = spacy.load("fr_core_news_md")
        liste_mots_inutiles = stop_words.get_stop_words('fr', cache=False)
        liste_mots_a_garder = ["pas", "n", "ni", "ne", "bon", "avec", "sans","aucun","bon","tres"] 
        for mot in liste_mots_a_garder : 
            try :
                liste_mots_inutiles.remove(mot)
            except :
                pass
            
        dataset[var] = dataset[var].apply(lambda x : unidecode.unidecode(str(x)).lower()) #enlever accents et mettre min
        dataset[var] = dataset[var].apply(enlever_mots_inutiles, args=[liste_mots_inutiles]) #enlever stop words
        dataset[var] = dataset[var].apply(lemmatiser_phrase_, args=[modele_spacy]) #lemmatiser
    
    return dataset

def split_column(dataset, var, pattern):
    nb_col = len(dataset[var].str.split(pattern, expand=True).columns)
    dataset[[var+'_'+str(x) for x in range(nb_col)]] = dataset[var].str.split(pattern, expand=True)
    
    return dataset

def enlever_mots_inutiles(phrase, liste_stopwords_) :
    chaine = ""
    
    lettres = "([a-z]{1,})"  #Pour ne garder que les lettres et séparer à chaque espace
    for mot in re.findall(lettres, phrase) : 
        if mot in liste_stopwords_ or len(mot) < 2 : pass
        else : chaine += " " + mot
            
    return chaine.strip()

def lemmatiser_phrase_(phrase, lemmatiseur_) :
    try :
        if phrase == "" : return phrase
        
        liste_ = []
        for mot in phrase.split(" ") :
            decode_ = unidecode.unidecode(lemmatiseur_(mot)[0].lemma_.lower())
            if len(decode_) >= 2 : liste_.append(decode_)
        return " ".join( liste_ ).strip()
    
    except : 
        raise ValueError("ERREUR : ***" + phrase + "***")
        
def discretiser(dataset, var, bins, method='auto', labels=False):
    dataset[var+'_bins'] = pd.cut(dataset[var], bins, labels).astype(str)
    
    return dataset

def calculate_percentil(dataset, var):
    df = dataset.loc[:,[var]].copy()
    df.dropna(inplace=True)
    # sort it by the desired series and caculate the percentile
    sdf = df.sort_values(var).reset_index()
    sdf[var+'_percentile'] = (sdf.index / float(len(sdf) - 1))*100
    # setup the interpolator using the value as the index
    sdf = sdf.set_index('index')
    sdf = sdf.round({var+'_percentile':0})
    dataset = pd.merge(dataset, sdf.loc[:,[var+'_percentile']], how='left', left_index=True, right_index=True)
    
    return dataset

def extract_date_pattern(dataset, var, pattern):
    if pattern == 'Weekday abbr (lun.)':
        dataset[var+'_wdab'] = dataset[var].dt.strftime("%a") #weekday abbr
    elif pattern == 'Weekday complet (lundi)':
        dataset[var+'_wdcp'] = dataset[var].dt.strftime("%A") #weekday complet
    elif pattern == 'Weekday chiffre (01)':
        dataset[var+'_wdcf'] = dataset[var].dt.strftime("%w") #weekday chiffre
    elif pattern == 'Jour (18)':
        dataset[var+'_jcf'] = dataset[var].dt.strftime("%d") #jour chiffre
    elif pattern == 'Mois abbr (jan.)':
        dataset[var+'_mab'] = dataset[var].dt.strftime("%b") #mois abbr
    elif pattern == 'Mois complet (janvier)':
        dataset[var+'_mcp'] = dataset[var].dt.strftime("%B") #mois complet
    elif pattern == 'Mois chiffre (01)':
        dataset[var+'_mcf'] = dataset[var].dt.strftime("%m") #mois chiffre
    elif pattern == 'Année derniers chiffres (21)':
        dataset[var+'_a2cf'] = dataset[var].dt.strftime("%y") #année 2 derniers chiffres
    elif pattern == 'Année complète (2021)':
        dataset[var+'_acp'] = dataset[var].dt.strftime("%Y") #année complet
        
    return dataset

def get_NER(dataset, col):
    model = spacy.load("kaduceo_nettoye.model")
    data_copied = dataset.loc[:,[col]].copy()
    ## nettoyage ##
    modele_spacy = spacy.load("fr_core_news_md")
    liste_mots_inutiles = stop_words.get_stop_words('fr', cache=False)
    liste_mots_a_garder = ["pas", "n", "ni", "ne", "bon", "avec", "sans","aucun","bon","très","-","+"] 

    for mot in liste_mots_a_garder : 
        try :
            liste_mots_inutiles.remove(mot)
        except :
            pass
        
    data_copied['nettoye'] = data_copied[col].apply(lambda x : unidecode.unidecode(str(x)).lower())
    data_copied['nettoye'] = data_copied['nettoye'].apply(enlever_mots_inutiles, args=[liste_mots_inutiles])
    data_copied['nettoye'] = data_copied['nettoye'].apply(lemmatiser_phrase_, args=[modele_spacy]) #lemmatiser
    liste_mots_inutiles = [ unidecode.unidecode(modele_spacy(mot)[0].lemma_.lower()) for mot in liste_mots_inutiles ]
    data_copied['nettoye'] = data_copied['nettoye'].apply(enlever_mots_inutiles, args=[liste_mots_inutiles])
    regex = re.compile(r'[\n\r\t]')
    data_copied['nettoye'] = data_copied['nettoye'].str.replace(regex," ")
    
    ## Reconnaissance NER ##
    data_copied['EXAM'] = ''
    data_copied['SIT_DIS'] = ''
    data_copied['SIT_NORM'] = ''
    
    for ligne in data_copied.index:
        doc = model(data_copied.at[ligne, 'nettoye'])

        EXAM = ""
        SIT_DIS = ""
        SIT_NORM = ""
        for ent in doc.ents:
            if ent.label_ == 'EXAM':
                EXAM = EXAM + str(ent.text) + ";"
            if ent.label_ == 'SIT_DIS':
                SIT_DIS = SIT_DIS + str(ent.text) + ";"
            if ent.label_ == 'SIT_NORM':
                SIT_NORM = SIT_NORM + str(ent.text) + ";"
        EXAM = EXAM[:-1]
        SIT_DIS = SIT_DIS[:-1]
        SIT_NORM = SIT_NORM[:-1]

        data_copied.at[ligne, 'EXAM'] = EXAM
        data_copied.at[ligne, 'SIT_DIS'] = SIT_DIS
        data_copied.at[ligne, 'SIT_NORM'] = SIT_NORM
        
    data_copied = data_copied.drop(columns=['nettoye'])
    
    return data_copied

def get_dummies_from_NER(dataset, col, NER):
    df = get_NER(dataset, col)
    tabsplit = df[NER].str.split(';', expand=True).reset_index()
    tabsplit[0] = np.where(tabsplit[0]=='',None,tabsplit[0])
    melt = pd.melt(tabsplit, id_vars=['index']).drop(columns='variable')
    dummies = pd.get_dummies(melt, prefix=NER).groupby('index').sum()
    
    
    df = df.drop(columns=['EXAM', 'SIT_DIS', 'SIT_NORM'])
    dataset = pd.merge(df, dummies, left_index=True, right_index=True)
    dataset[NER+'_BOOL'] = np.where(dataset.sum(axis=1) > 0, True, False)
    
    return dataset

def get_cp_insee(data_, var):
    """Renvoie le dataframe qui a une colonne de codes postaux, avec les codes postaux qui se trouvent dans le fichier insee.
       Ainsi on pourra ensuite faire des jointures sur les codes postaux avec des données insee.

    Paramètres :
    data_ -- dataframe avec une colonne "code_postal"
    insee_ -- dataframe des données insee de codes postaux avec une colonne "code_postal"

    """
    insee_ = pd.read_csv("static/data/data_insee.csv")
    data_ = data_.rename(columns={var:'CODE_POSTAL_TO_REPLACE'})
    data=data_.loc[:,["CODE_POSTAL_TO_REPLACE"]].copy()
    data = data.rename(columns={"CODE_POSTAL_TO_REPLACE":'code_postal'})
    data["code_postal"] = data["code_postal"].astype("int32").astype("str")
    
    #Pour les codes postaux de longueur 4 au lieu de 5, au rajoute un 0 devant
    data.loc[:,"code_postal_insee"] = np.where(data['code_postal'].str.len()==4, '0'+data['code_postal'], data['code_postal'])

    #Nous récupérons dans le fichier insee les codes postaux qui ne sont pas de longueur 5
    #exemple de code postal: "92200/92210"
    cp_groupes = list(insee_[insee_['code_postal'].apply(len)!=5]['code_postal'])
    #Liste des code postaux qui font partie d'un "aggregat" postal  dans les donnees insee
    cp_groupes_split = [nb for line in cp_groupes for nb in line.split('/')]
    
    #Nous mettons de côté les codes postaux de data qui n'appartiennent pas à cp_groupes
    data_sans_prob = data[~data.loc[:,"code_postal"].isin(cp_groupes_split)].copy(deep=True)

    #Nous remplaçons les codes postaux de data présents sous format type "92200/92210"
    #par leurs codes postaux insee correspondants
    data_prob = data[data.loc[:,"code_postal"].isin(cp_groupes_split)].copy(deep=True)
    cp_insee_prob = insee_[insee_["code_postal"].isin(cp_groupes)]
    data_prob["code_postal_insee"] = list(data_prob.apply(lambda x: cp_insee_prob.loc[cp_insee_prob['code_postal'].str.contains(x.code_postal),"code_postal"].item(),                                       axis=1))

    #Nous concategnons les dataframe data_sans_prob et data_prob
    data_clean = pd.concat([data_sans_prob,data_prob])
    data_clean = data_clean.sort_index()
    data_["CODE_POSTAL_TO_REPLACE"] = data_clean["code_postal_insee"].values
    data_ = data_.rename(columns={'CODE_POSTAL_TO_REPLACE':var})
    
    return data_

def extract_from_cp_insee(data, var, method):
    insee_ = pd.read_csv("static/data/data_insee.csv")
    if method in data.columns:
        data = data.drop(columns=[method])
    data = pd.merge(data, insee_.loc[:,['code_postal', method]], how='left', left_on = var, right_on='code_postal').drop_duplicates()
    
    return data        

def format_date(dataset, var, pattern):
    #Defaut : Y-m-d
    if pattern == '18 Janvier 2021':
        dataset[var+'_formatted'] = dataset[var].dt.strftime('%d %B %Y')
    elif pattern == 'lundi 18 Janvier 2021':
        dataset[var+'_formatted'] = dataset[var].dt.strftime('%A %d %B %Y')
    elif pattern == '18/01/2021':
        dataset[var+'_formatted'] = dataset[var].dt.strftime('%d/%m/%Y')
    elif pattern == '18-01-2021':
        dataset[var+'_formatted'] = dataset[var].dt.strftime('%d-%m-%Y')
    elif pattern == '01/18/2021':
        dataset[var+'_formatted'] = dataset[var].dt.strftime('%m/%d/%Y')
    elif pattern == '01-18-2021':
        dataset[var+'_formatted'] = dataset[var].dt.strftime('%m-%d-%Y')
        
    return dataset

def create_var_from_cdt(dataset, df_cdt, col, val, inherit=False, old_var=None):
    df_copy = dataset.copy()
    for ix in df_cdt.index:
        if df_cdt.at[ix, 'connecteur'] == 'egal':
            df_copy = df_copy.loc[(df_copy[df_cdt.at[ix, 'variable']]) == df_cdt.at[ix, 'comparison_value']]
            continue
        elif df_cdt.at[ix, 'connecteur'] == 'différent':
            df_copy = df_copy.loc[(df_copy[df_cdt.at[ix, 'variable']]) != df_cdt.at[ix, 'comparison_value']]
            continue
        elif df_cdt.at[ix, 'connecteur'] == 'supérieur':
            df_copy = df_copy.loc[(df_copy[df_cdt.at[ix, 'variable']]) > df_cdt.at[ix, 'comparison_value']]
            continue
        elif df_cdt.at[ix, 'connecteur'] == 'inférieur':
            df_copy = df_copy.loc[(df_copy[df_cdt.at[ix, 'variable']]) < df_cdt.at[ix, 'comparison_value']]
            continue
        elif df_cdt.at[ix, 'connecteur'] == 'supérieur ou égal':
            df_copy = df_copy.loc[(df_copy[df_cdt.at[ix, 'variable']]) >= df_cdt.at[ix, 'comparison_value']]
            continue
        elif df_cdt.at[ix, 'connecteur'] == 'inférieur ou égal':
            df_copy = df_copy.loc[(df_copy[df_cdt.at[ix, 'variable']]) <= df_cdt.at[ix, 'comparison_value']]
            continue
        elif df_cdt.at[ix, 'connecteur'] == 'commence par':
            df_copy = df_copy.loc[(df_copy[df_cdt.at[ix, 'variable']]).str.startswith(df_cdt.at[ix, 'comparison_value'], na=False)]
            continue
        elif df_cdt.at[ix, 'connecteur'] == 'fini par':
            df_copy = df_copy.loc[(df_copy[df_cdt.at[ix, 'variable']]).str.endswith(df_cdt.at[ix, 'comparison_value'], na=False)]
            continue
        elif df_cdt.at[ix, 'connecteur'] == 'contient':
            df_copy = df_copy.loc[(df_copy[df_cdt.at[ix, 'variable']]).str.contains(df_cdt.at[ix, 'comparison_value'], na=False)]
            continue
    
    if inherit == False:
        dataset[col] = np.where(dataset.index.isin(df_copy.index), val, dataset[col])
    else:
        dataset[col] = np.where(dataset.index.isin(df_copy.index), val, dataset[old_var])
                            
    return dataset

def map_type(dataset, dataset_used_to_enrich, dataset_keys):
    for i in dataset_keys.index:
        origin_type = gf.return_type(dataset, dataset_keys.at[i, 'Mapping_df_origine'])
        var_to_map = dataset_keys.at[i, 'Mapping_nouveau_df']
        excepetions_list = []
        try:
            if origin_type == 'object':
                try:
                    dataset_used_to_enrich[var_to_map] = dataset_used_to_enrich[var_to_map].astype('Int64').astype('str').replace('<NA>', np.nan)
                except : 
                    dataset_used_to_enrich[var_to_map] = dataset_used_to_enrich[var_to_map].astype(object)
            elif origin_type == 'float64':
                dataset_used_to_enrich[var_to_map] = dataset_used_to_enrich[var_to_map].astype('float64')
            else:
                dataset_used_to_enrich[var_to_map] = pd.to_datetime(dataset_used_to_enrich[var_to_map], errors='coerce', infer_datetime_format=True)
        except:
            excepetions_list.append(var_to_map)
            pass
    if excepetions_list:
        st.warning("Les variables suivantes n'ont pas pu être correctement mappées: "+excepetions_list)
    return dataset_used_to_enrich

def calculate_diff_between_dates(dataset, date1, date2, returned_unit):
    if returned_unit == 'jours':
        dataset["difference_entre_"+date1+"_et_"+date2+"_en_jours"] = (dataset[date1] - dataset[date2]).dt.days
    elif returned_unit == 'minutes':
        dataset["difference_entre_"+date1+"_et_"+date2+"_en_minutes"] = (dataset[date1] - dataset[date2]).dt.days * 24 * 60
    elif returned_unit == 'secondes':
        dataset["difference_entre_"+date1+"_et_"+date2+"_en_secondes"] = (dataset[date1] - dataset[date2]).dt.days * 24 * 60 * 60
        
    return dataset