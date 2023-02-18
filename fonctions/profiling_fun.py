import streamlit as st
import pandas as pd
import seaborn as sns
import phik #Ne pas commenter
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import fonctions.general_fun as gf
import fonctions.diagnosis_fun as diaf
from pylab import savefig
import unidecode
import re

def describe_df(df, DF_META):
    nb_var = len(df.columns)
    nb_lignes = len(df)
    
    num = 0
    cat = 0
    date = 0
    for i in df.dtypes:
        # if i == 'Int64':
        #     num = num+1
        #     continue
        if i == 'float64':
            num = num+1
            continue
        elif i == 'object':
            cat = cat+1
        else:
            date = date+1
    
    nullable = 0
    boolean = 0
    unique = 0
    master = 0
    min_value = 0
    max_value=0
    exclude_values=0
    for i in DF_META.index:
        if DF_META.at[i, 'nullable'] == True:
            nullable = nullable+1
        if DF_META.at[i, 'boolean'] == True:
            boolean = boolean+1
        if DF_META.at[i, 'unique'] == True:
            unique = unique+1
        if DF_META.at[i, 'master'] == True:
            master = master+1
        if (DF_META.at[i, 'min_value'] != None) & (pd.notna(DF_META.at[i, 'min_value'])):
            min_value = min_value+1
        if (DF_META.at[i, 'max_value'] != None) & (pd.notna(DF_META.at[i, 'max_value'])):
            max_value = max_value+1
        if DF_META.at[i, 'exclude_values'] != None:
            exclude_values = exclude_values+1

    #Metadata
    df_analyse_meta = pd.DataFrame(columns=['Type', 'Nombre de variables'])
    df_analyse_meta = df_analyse_meta.append({'Type':'Nullable','Nombre de variables':round((nullable/nb_var)*100,2)},ignore_index=True)
    df_analyse_meta = df_analyse_meta.append({'Type':'Boolean','Nombre de variables':round((boolean/nb_var)*100,2)},ignore_index=True)
    df_analyse_meta = df_analyse_meta.append({'Type':'Unique','Nombre de variables':round((unique/nb_var)*100,2)},ignore_index=True)
    df_analyse_meta = df_analyse_meta.append({'Type':'Master','Nombre de variables':round((master/nb_var)*100,2)},ignore_index=True)
    df_analyse_meta = df_analyse_meta.append({'Type':'Avec une valeur minimale','Nombre de variables':round((min_value/nb_var)*100,2)},ignore_index=True)
    df_analyse_meta = df_analyse_meta.append({'Type':'Avec une valeur maximale','Nombre de variables':round((max_value/nb_var)*100,2)},ignore_index=True)
    df_analyse_meta = df_analyse_meta.append({'Type':'Avec des pattern exclus','Nombre de variables':round((exclude_values/nb_var)*100,2)},ignore_index=True)
    df_analyse_meta = df_analyse_meta.set_index('Type')
    
    fig0, ax = plt.subplots()
    ax = df_analyse_meta.plot.bar( ax=plt.gca(), width=0.8, color=["cornflowerblue"] )
    ax.set_title("Analyse des métadonnées", fontdict={'fontsize': 12, 'fontweight': 'medium'})
    ax.legend().set_visible(False)
    plt.ylabel("Pourcentage dans le dataset")
    plt.xlabel("Variable")
    plt.tick_params(axis='x', labelrotation=90)
    plt.yticks(np.arange(0, 110, 10))
    
    for rect in ax.patches:
        height = rect.get_height() ; width = rect.get_width()
        x = rect.get_x() ; y = rect.get_y()
        
        label_x = x + width/2 ; label_y = y + height/2
        label_text = round(height, 1)
    
        #Plot only when height is greater than specified value
        if height > 0 : ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=8, color="white")
        
    savefig("tempDir/tempProfiling/anaylse_meta1.png", bbox_inches='tight')   
    
    #Type données
    labels=['Données textuelles', 'Données numériques', 'Données temporelles']
    sizes = [round((cat/nb_var)*100,2), round((num/nb_var)*100,2), round((date/nb_var)*100,2)]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',  colors=["orange", "mediumorchid", "forestgreen"])
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title("Répartition des types dans le dataset", fontdict={'fontsize': 12, 'fontweight': 'medium'})
    savefig('tempDir/tempProfiling/anaylse_meta2.png', bbox_inches='tight')
    
    return nb_var, nb_lignes, fig0, fig1

def doublons(df, big_df=False):
    lignes_dup = str(len(df[df.duplicated(keep=False)].reset_index()))+" lignes dupliquées"

    if big_df == False: #Exécuter seulement si l'utilisateur le décide
        if len(df) > 1000:
            df_samp = df.sample(1000)
            matrix = df_samp.phik_matrix()
            st.warning("Pour des raisons d'optimisation, les corrélations sont effectuées sur un échantillon du dataset (n=1000)")
        else:
            matrix = df.phik_matrix()
            
        hm =plt.figure()
        plt.title("Corrélation entre les variables (Phik)")
        sns.heatmap(matrix)
        savefig('tempDir/tempProfiling/doublons1.png', bbox_inches='tight')
        
        return lignes_dup, hm
    
    else:
        return lignes_dup

def missing(df, DF_META):
    list_var_meta = DF_META['variable'].to_list()
    df_miss = pd.DataFrame(columns=['classe', 'col', 'percent', 'nb'])
    for col in df.columns:
        if col in list_var_meta:
            variable_index = DF_META.loc[DF_META['variable'] == col].index[0]
            nullable = DF_META.at[variable_index, 'nullable']
        else:
            nullable = False
            
        if nullable == True:
            for classe in ['Valeur manquante nullable', 'Valeur non manquante']:
                if classe == 'Valeur manquante nullable':
                    nb_missing_nullable = df[col].isnull().sum()
                    percent_missing_nullable = round((nb_missing_nullable/len(df))*100, 2)
                    df_miss = df_miss.append({'classe':classe,
                                              'col':col,
                                              'percent':percent_missing_nullable,
                                             'nb':nb_missing_nullable},
                                                                  ignore_index=True)
                else:
                    nb_nullable = len(df) - nb_missing_nullable
                    percent_not_missing_nullable = round(100 - percent_missing_nullable, 2)
                    df_miss = df_miss.append({'classe':classe,
                                              'col':col,
                                              'percent':percent_not_missing_nullable,
                                             'nb':nb_nullable},
                                                                  ignore_index=True)
        else :
            for classe in ['Valeur manquante', 'Valeur non manquante']:
                if classe == 'Valeur manquante':
                    nb_missing = df[col].isnull().sum()
                    percent_missing = round((nb_missing/len(df))*100, 2)
                    df_miss = df_miss.append({'classe':classe,
                                              'col':col,
                                              'percent':percent_missing,
                                             'nb':nb_missing},
                                                                  ignore_index=True)
                else:
                    nb = len(df) - nb_missing
                    percent_not_missing = round(100 - percent_missing, 2)
                    df_miss = df_miss.append({'classe':classe,
                                              'col':col,
                                              'percent':percent_not_missing,
                                             'nb':nb},
                                                                  ignore_index=True)
    
    pivot_table = df_miss.pivot(index = 'col', columns='classe', values='percent')
    pivot_table = pivot_table.fillna(0)
    cols = pivot_table.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    pivot_table = pivot_table[cols]
    
    #Histogram par variable
    fig0, ax = plt.subplots()
    ax = pivot_table.plot.bar( ax=plt.gca(), stacked=True, width=0.8, color=["green", "lightgrey", "lightgreen"] )
    l = ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.ylabel("Pourcentage")
    plt.xlabel("Variable")
    ax.set_title('Proportion de valeurs manquantes dans chaque variable', fontdict={'fontsize': 12, 'fontweight': 'medium'})
    plt.tick_params(axis='x', labelrotation=90)
        #Annotations :
    for rect in ax.patches:
        height = rect.get_height() ; width = rect.get_width()
        x = rect.get_x() ; y = rect.get_y()
        
        label_x = x + width/2 ; label_y = y + height/2
        label_text = round(height, 1)
    
        #Plot only when height is greater than specified value
        if height > 0 : ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=8, color="black")
    savefig('tempDir/tempProfiling/missing1.png', bbox_inches='tight')
    
    #Camembert
    vm = 0
    vnm = 0
    vmn = 0
    for l in df_miss.index:
        if df_miss.at[l, 'classe'] == 'Valeur non manquante':
            vnm = vnm + df_miss.at[l, 'nb']
            continue
        elif df_miss.at[l, 'classe'] == 'Valeur manquante':
            vm = vm + df_miss.at[l, 'nb']
            continue
        elif df_miss.at[l, 'classe'] == 'Valeur manquante nullable':
            vmn = vmn + df_miss.at[l, 'nb']
    
    labels=['Valeur non manquante', 'Valeur manquante', 'Valeur manquante nullable']
    sizes = [vnm, vm, vmn]
    
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',  colors=["green", "lightgrey", "lightgreen"])
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title("Proportion de valeurs manquantes dans le dataset", fontdict={'fontsize': 12, 'fontweight': 'medium'})
    savefig('tempDir/tempProfiling/missing2.png', bbox_inches='tight')
    
    return fig0, fig1

def hors_limite(df, DF_META):
    list_var_meta = DF_META['variable'].to_list()
    df_oor = pd.DataFrame(columns=['classe', 'col', 'percent', 'nb'])   
    
    for col in df.columns:
        data_type = gf.return_type(df, col)
        #if (data_type == 'Int64') | (data_type == 'float64'): #Check si c'est une colonne numérique
        if data_type == 'float64': #Check si c'est une colonne numérique
            df_copy = df.loc[:,[col]].dropna() #On réalise l'étude sur les données non manquantes
            if len(df_copy) == 0:
                continue
            for classe in ['Valeur hors limite', 'Valeur outlier', 'Valeur correcte']:
                if classe == 'Valeur hors limite':
                    if col not in list_var_meta:
                        nb_flagged = 0
                        percent_flagged = 0
                        nb_out = 0
                        percent_out = 0
                        index_flagged=[]
                        continue
                    else: #Si la var est présente dans les metadata
                        variable_index = DF_META.loc[DF_META['variable'] == col].index[0]
                        min_val = DF_META.at[variable_index,'min_value']
                        max_val = DF_META.at[variable_index,'max_value']
                        df_flagged = df_copy.loc[(df_copy[col] < min_val) | (df_copy[col] > max_val)]
                        index_flagged = df_flagged.index
                        nb_flagged = len(df_flagged)
                        percent_flagged = round((nb_flagged / len(df_copy)*100), 2)
                        df_oor = df_oor.append({'classe':classe,
                                                  'col':col,
                                                  'percent':percent_flagged,
                                                 'nb':nb_flagged},ignore_index=True)
                        
                elif classe == 'Valeur outlier': #Method a 3 ecarts types
                    #On retire du dataset les observations flaggués comme hors limite
                    df_copy[col] = df_copy[col].apply('float64')
                    y_hat_df = pd.DataFrame(np.abs(stats.zscore(df_copy.loc[~df_copy.index.isin(index_flagged),[col]])), columns=['y_hat'])
                    nb_out = len(y_hat_df.loc[y_hat_df['y_hat'] > 3])
                    percent_out = round((nb_out / len(df_copy))*100, 2)
                    df_oor = df_oor.append({'classe':classe,
                                  'col':col,
                                  'percent':percent_out,
                                 'nb':nb_out},ignore_index=True)
                
                elif classe == 'Valeur correcte':
                    nb = len(df_copy) - nb_flagged - nb_out
                    percent = round(100 - percent_flagged - percent_out, 2)
                    df_oor = df_oor.append({'classe':classe,
                                              'col':col,
                                              'percent':percent,
                                             'nb':nb}, ignore_index=True)
                    nb_flagged = 0
                    percent_flagged = 0
                    nb_out = 0
                    percent_out = 0
                    index_flagged=[]
                    
        else:
            continue
        
    pivot_table = df_oor.pivot(index = 'col', columns='classe', values='percent')
    pivot_table = pivot_table.fillna(0)
    if 'Valeur hors limite' in df_oor['classe'].unique():
        cols = ['Valeur correcte', 'Valeur outlier', 'Valeur hors limite']
    else:
        cols = ['Valeur correcte', 'Valeur outlier']
    pivot_table = pivot_table[cols] 
    
    #Histogram par variable
    fig0, ax = plt.subplots()
    ax = pivot_table.plot.bar( ax=plt.gca(), stacked=True, width=0.8, color=["green", "orange", "red"] )
    l = ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    
    plt.ylabel("Pourcentage (parmis les données non manquantes)")
    plt.xlabel("Variable numérique")
    ax.set_title('Proportion de valeurs hors limite dans chaque variable numérique parmis les données non manquantes', fontdict={'fontsize': 12, 'fontweight': 'medium'})
    plt.tick_params(axis='x', labelrotation=90)
            
    #Annotations :
    for rect in ax.patches:
        height = rect.get_height() ; width = rect.get_width()
        x = rect.get_x() ; y = rect.get_y()
        
        label_x = x + width/2 ; label_y = y + height/2
        label_text = round(height, 1)
    
        #Plot only when height is greater than specified value
        if height > 0 : ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=8, color="black")
    savefig('tempDir/tempProfiling/hl1.png', bbox_inches='tight')
    
    #Camembert
    vo = 0
    vc = 0
    vhl = 0
    for l in df_oor.index:
        if df_oor.at[l, 'classe'] == 'Valeur correcte':
            vc = vc + df_oor.at[l, 'nb']
            continue
        elif df_oor.at[l, 'classe'] == 'Valeur outlier':
            vo = vo + df_oor.at[l, 'nb']
            continue
        elif df_oor.at[l, 'classe'] == 'Valeur hors limite':
            vhl = vhl + df_oor.at[l, 'nb']
    
    if 'Valeur hors limite' in df_oor['classe'].unique():
        labels=['Valeur correcte', 'Valeur outlier', 'Valeur hors limite']
        sizes = [vc, vo, vhl]
    else:
        labels=['Valeur correcte', 'Valeur outlier']
        sizes = [vc, vo]
        
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',  colors=["green", "orange", "red"])
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title("Proportion de valeurs hors limite dans le dataset (uniquement pour les valeurs numériques) parmis les données non manquantes", fontdict={'fontsize': 12, 'fontweight': 'medium'})
    savefig('tempDir/tempProfiling/hl2.png', bbox_inches='tight')
    
    return fig0, fig1

def incompatible(df, DF_THEN, DF_IF):
    df_incomp = pd.DataFrame(columns=['Règle','Intitulé', 'Enregistrements incompatibles (%)'])
    for rule in DF_THEN['ruleID'].unique():
        phrase = diaf.write_rule(DF_IF, DF_THEN, rule)
        DF_RULE = diaf.find_incompatibilities(dataset=df, tab1=DF_IF, tab2=DF_THEN, rule=rule)
        nb_ligne = len(DF_RULE['row_breaking_rule'].unique())
        percent = round((nb_ligne / len(df))*100, 2)
        df_incomp = df_incomp.append({'Règle':rule,
                                      'Intitulé': phrase,
                                      'Enregistrements incompatibles (%)': str(nb_ligne) + ' ('+ str(percent)+'%)'}, ignore_index=True)
    
    return df_incomp

def excluded(df, DF_META):
    var_meta = DF_META['variable'].to_list()
    df_excluded = pd.DataFrame(columns=['Variable', 'Pattern Exclus', 'Enregistrements incompatibles (%)'])
    for col in df.columns:
        if col not in var_meta:
            continue
        elif DF_META.at[DF_META.loc[DF_META['variable'] == col].index[0], 'exclude_values'] == None:
            continue
        else:
            df_copy = df.loc[:,[col]]
            variable_index = DF_META.loc[DF_META['variable'] == col].index[0]
            for val in DF_META.at[variable_index, 'exclude_values']:
                nb_val_excluded = len(df_copy.loc[df_copy[col] == val])
                percent_excluded = round((nb_val_excluded / len(df))*100,2)
                df_excluded = df_excluded.append({'Variable':col,
                              'Pattern Exclus':val,
                              'Enregistrements incompatibles (%)':str(nb_val_excluded) + ' ('+ str(percent_excluded)+'%)'}, ignore_index=True)
    
    return df_excluded

def erreur_tempo(df, DF_TEMP):
    df_tempo = pd.DataFrame(columns=['Règle ID', 'Clé', 'Variable Temporelle', 'Résultat'])
    for rule_temp in DF_TEMP.index:
        df_flag = diaf.find_temporal_errors(df, DF_TEMP, rule_temp)
        if df_flag.empty == False:
            df_tempo = df_tempo.append({'Règle ID':rule_temp,
                          'Clé':DF_TEMP.at[rule_temp, 'clé'],
                          'Variable Temporelle': DF_TEMP.at[rule_temp, 'variable_temporelle'],
                          'Résultat':'Erreur détecté'}, ignore_index=True)
        else:
            df_tempo = df_tempo.append({'Règle ID':rule_temp,
                          'Clé':DF_TEMP.at[rule_temp, 'clé'],
                          'Variable Temporelle':DF_TEMP.at[rule_temp, 'variable_temporelle'],
                          'Résultat':'Aucune erreur détecté'}, ignore_index=True)    
    
    return df_tempo

def var_object(df_copy, col):
    nb_unique = len(df_copy[col].unique())
    grouped = df_copy.reset_index().groupby([col]).count().rename(columns={"index": "Pourcentage"}).sort_values(by=["Pourcentage"], ascending=False)
    grouped["Pourcentage"] = round((grouped["Pourcentage"] / len(df_copy))*100,2)
    
    #Head()
    fig0, ax = plt.subplots()
    ax = grouped.head().plot.bar( ax=plt.gca(), width=0.8, color=["cornflowerblue"] )
    ax.set_title('5 modalités les plus représentées pour la variable '+col, fontdict={'fontsize': 12, 'fontweight': 'medium'})
    ax.legend().set_visible(False)
    plt.ylabel("Répartition en %")
    plt.tick_params(axis='x', labelrotation=90)
    
    for rect in ax.patches:
        height = rect.get_height() ; width = rect.get_width()
        x = rect.get_x() ; y = rect.get_y()
        
        label_x = x + width/2 ; label_y = y + height/2
        label_text = round(height, 1)
    
        #Plot only when height is greater than specified value
        if height > 0 : ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=8, color="white")
    savefig("tempDir/tempProfiling/"+col+"_head.png", bbox_inches='tight')
    
    #tail()
    fig1, ax1 = plt.subplots()
    ax1 = grouped.tail().sort_values(by=["Pourcentage"], ascending=True).plot.bar( ax=plt.gca(), width=0.8, color=["cornflowerblue"] )
    ax1.set_title('5 modalités les moins représentées pour la variable '+col, fontdict={'fontsize': 12, 'fontweight': 'medium'})
    ax1.legend().set_visible(False)
    plt.ylabel("Répartition en %")
    plt.tick_params(axis='x', labelrotation=90)
    
    for rect in ax1.patches:
        height = rect.get_height() ; width = rect.get_width()
        x = rect.get_x() ; y = rect.get_y()
        
        label_x = x + width/2 ; label_y = y + height/2
        label_text = round(height, 1)
    
        #Plot only when height is greater than specified value
        if height > 0 : ax1.text(label_x, label_y, label_text, ha='center', va='center', fontsize=8, color="white")
    savefig("tempDir/tempProfiling/"+col+"_tail.png", bbox_inches='tight')
    
    return nb_unique, fig0, fig1

def cs_pattern(string):
    pattern_detected = unidecode.unidecode(string)

    regex_digit = r"[-+]?(\d+([.,]\d*)?|[.,]\d+)([eE][-+]?\d+)?"
    regex_lonely_letter = r"(?!digit)\s[a-z]\s"
    regex_lonely_Letter = r"(?!digit|char)\s[A-Z]\s"
    regex_word_maj = r'(?!digit|char|Char)\b[A-Z]+\b'
    regex_word_propper = r'(?!digit|char|Char|WORD)\b[A-Z].*?\b'
    regex_word_min = r'(?!digit|char|Char|WORD|Word)\b[a-z].*?\b'

    pattern_detected = re.sub(regex_digit, ' {digit} ', pattern_detected)
    pattern_detected = re.sub(regex_lonely_letter, ' {char} ', pattern_detected)
    pattern_detected = re.sub(regex_lonely_Letter, '{Char}', pattern_detected)
    pattern_detected = re.sub(regex_word_maj, ' {WORD} ', pattern_detected)
    pattern_detected = re.sub(regex_word_propper, ' {Word} ', pattern_detected)
    pattern_detected = re.sub(regex_word_min, ' {word} ', pattern_detected)

    pattern_detected = pattern_detected.replace(" ", "")
    pattern_detected = pattern_detected.replace("{{", "{")
    pattern_detected = pattern_detected.replace("}}", "}")
    
    return pattern_detected

def var_object_pattern(df_copy, col):
    len_df = len(df_copy)
    df_copy = df_copy.dropna()
    df_copy['Fréquence du pattern'] = df_copy[col].apply(cs_pattern)
    grouped = df_copy.reset_index().groupby(['Fréquence du pattern']).count().rename(columns={"index": "Pourcentage"}).sort_values(by=["Pourcentage"], ascending=False)
    grouped["Pourcentage"] = round((grouped["Pourcentage"] / len_df)*100,2)
    
        #Head()
    fig2, ax = plt.subplots()
    ax = grouped[['Pourcentage']].head().plot.bar( ax=plt.gca(), width=0.8, color=["cornflowerblue"] )
    ax.set_title('5 pattern les plus représentés pour la variable '+col, fontdict={'fontsize': 12, 'fontweight': 'medium'})
    ax.legend().set_visible(False)
    plt.ylabel("Répartition en %")
    plt.tick_params(axis='x', labelrotation=90)
    
    for rect in ax.patches:
        height = rect.get_height() ; width = rect.get_width()
        x = rect.get_x() ; y = rect.get_y()
        
        label_x = x + width/2 ; label_y = y + height/2
        label_text = round(height, 1)
    
        #Plot only when height is greater than specified value
        if height > 0 : ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=8, color="white")
    savefig("tempDir/tempProfiling/"+col+"_pattern_head.png", bbox_inches='tight')
    
    #tail()
    fig3, ax1 = plt.subplots()
    ax1 = grouped[['Pourcentage']].tail().plot.bar( ax=plt.gca(), width=0.8, color=["cornflowerblue"] )
    ax1.set_title('5 pattern les moins représentées pour la variable '+col, fontdict={'fontsize': 12, 'fontweight': 'medium'})
    ax1.legend().set_visible(False)
    plt.ylabel("Répartition en %")
    plt.tick_params(axis='x', labelrotation=90)
    
    for rect in ax1.patches:
        height = rect.get_height() ; width = rect.get_width()
        x = rect.get_x() ; y = rect.get_y()
        
        label_x = x + width/2 ; label_y = y + height/2
        label_text = round(height, 1)
    
        #Plot only when height is greater than specified value
        if height > 0 : ax1.text(label_x, label_y, label_text, ha='center', va='center', fontsize=8, color="white")
    savefig("tempDir/tempProfiling/"+col+"_pattern_tail.png", bbox_inches='tight')
    
    return fig2, fig3
    
def var_date(df_copy, col):
    nb_unique = len(df_copy[col].unique())
    min_val = df_copy[col].min()
    max_val = df_copy[col].max()
    grouped = df_copy.reset_index().groupby([col]).count().rename(columns={"index": "Pourcentage"}).sort_values(by=["Pourcentage"], ascending=False)
    grouped["Pourcentage"] = round((grouped["Pourcentage"] / len(df_copy))*100,2)
    
    #Head()
    fig0, ax = plt.subplots()
    ax = grouped.head().plot.bar( ax=plt.gca(), width=0.8, color=["cornflowerblue"] )
    ax.set_title('5 modalités les plus représentées pour la variable '+col, fontdict={'fontsize': 12, 'fontweight': 'medium'})
    ax.legend().set_visible(False)
    plt.ylabel("Répartition en %")
    plt.tick_params(axis='x', labelrotation=90)
    
    for rect in ax.patches:
        height = rect.get_height() ; width = rect.get_width()
        x = rect.get_x() ; y = rect.get_y()
        
        label_x = x + width/2 ; label_y = y + height/2
        label_text = round(height, 1)
    
        #Plot only when height is greater than specified value
        if height > 0 : ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=8, color="white")
    savefig("tempDir/tempProfiling/"+col+"_head.png", bbox_inches='tight')    
    #tail()
    fig1, ax1 = plt.subplots()
    ax1 = grouped.tail().sort_values(by=["Pourcentage"], ascending=True).plot.bar( ax=plt.gca(), width=0.8, color=["cornflowerblue"] )
    ax1.legend().set_visible(False)
    ax1.set_title('5 modalités les moins représentées pour la variable '+col, fontdict={'fontsize': 12, 'fontweight': 'medium'})
    plt.ylabel("Répartition en %")
    plt.tick_params(axis='x', labelrotation=90)
    
    for rect in ax1.patches:
        height = rect.get_height() ; width = rect.get_width()
        x = rect.get_x() ; y = rect.get_y()
        
        label_x = x + width/2 ; label_y = y + height/2
        label_text = round(height, 1)
    
        #Plot only when height is greater than specified value
        if height > 0 : ax1.text(label_x, label_y, label_text, ha='center', va='center', fontsize=8, color="white")
    savefig("tempDir/tempProfiling/"+col+"_tail.png", bbox_inches='tight')
    
    return nb_unique, min_val, max_val, fig0, fig1

def var_number(df_copy, col):
    nb_unique = len(df_copy[col].unique())
    min_val = df_copy[col].min()
    max_val = df_copy[col].max()
    mean_val = df_copy[col].mean()
    median_val = df_copy[col].median()
    nb_zero = len(df_copy.loc[df_copy[col]==0])
    
    grouped = df_copy.reset_index().groupby([col]).count().rename(columns={"index": "Pourcentage"}).sort_values(by=["Pourcentage"], ascending=False)
    grouped["Pourcentage"] = round((grouped["Pourcentage"] / len(df_copy))*100,2)
    
    #Head()
    fig0, ax = plt.subplots()
    ax = grouped.head().plot.bar( ax=plt.gca(), width=0.8, color=["cornflowerblue"] )
    ax.set_title('5 modalités les plus représentées pour la variable '+col, fontdict={'fontsize': 12, 'fontweight': 'medium'})
    ax.legend().set_visible(False)
    plt.ylabel("Répartition en %")
    plt.tick_params(axis='x', labelrotation=90)
    
    for rect in ax.patches:
        height = rect.get_height() ; width = rect.get_width()
        x = rect.get_x() ; y = rect.get_y()
        
        label_x = x + width/2 ; label_y = y + height/2
        label_text = round(height, 1)
    
        #Plot only when height is greater than specified value
        if height > 0 : ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=8, color="white")
    savefig("tempDir/tempProfiling/"+col+"_head.png", bbox_inches='tight')    
    #tail()
    fig1, ax1 = plt.subplots()
    ax1 = grouped.tail().sort_values(by=["Pourcentage"], ascending=True).plot.bar( ax=plt.gca(), width=0.8, color=["cornflowerblue"] )
    ax1.set_title('5 modalités les moins représentées pour la variable '+col, fontdict={'fontsize': 12, 'fontweight': 'medium'})
    ax1.legend().set_visible(False)
    plt.ylabel("Répartition en %")
    plt.tick_params(axis='x', labelrotation=90)
    
    for rect in ax1.patches:
        height = rect.get_height() ; width = rect.get_width()
        x = rect.get_x() ; y = rect.get_y()
        
        label_x = x + width/2 ; label_y = y + height/2
        label_text = round(height, 1)
    
        #Plot only when height is greater than specified value
        if height > 0 : ax1.text(label_x, label_y, label_text, ha='center', va='center', fontsize=8, color="white")
    savefig("tempDir/tempProfiling/"+col+"_tail.png", bbox_inches='tight')
    
    return nb_unique, min_val, max_val, mean_val, median_val, nb_zero, fig0, fig1