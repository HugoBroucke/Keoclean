import pandas as pd
import numpy as np
from scipy import stats
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

def recuperer_var_corr_selon_confiance(res, confiance='99%'):
#PhiK https://phik.readthedocs.io/en/latest/phik.html#module-phik.report
#https://connect.springerpub.com/content/book/978-0-8261-9825-9/back-matter/bmatter2 One-Sided Probabilities for z -Scores of the Standard Normal Distribution
    i=1
    result = []
    for col in res.columns:
        for li in range(i, len(res.index)):
            tab = []
            tab.append(col)
            tab.append(res.index[li])
            tab.append(res.loc[res.index[li],[col]].values[0])
            result.append(tab)

        i=i+1
        
    df_res = pd.DataFrame()
    
    if confiance == '99%':
        for i in result:
            if i[2] >= 2.33:
                df_res = df_res.append({'VAR X': i[0], 'VAR Y': i[1], 'Z-score': round(i[2], 3)}, ignore_index=True)
            else:
                continue
    elif confiance == '95%':
        for i in result:
            if i[2] >= 1.65:
                df_res = df_res.append({'VAR X': i[0], 'VAR Y': i[1], 'Z-score': round(i[2], 3)}, ignore_index=True)
            else:
                continue
    elif confiance == '90%':
        for i in result:
            if i[2] >= 1.29:
                df_res = df_res.append({'VAR X': i[0], 'VAR Y': i[1], 'Z-score': round(i[2], 3)}, ignore_index=True)
            else:
                continue
    
    return df_res

def find_missing_in_records(dataset, DF_META):
    """
    Permet de trouver le nombre de données manquantes ainsi que leur pourcentage records.
    
    Input : dataset
    Output : Tableau contenant pour chaque records son nombre de données manquantes en valeur absolu et en pourcentage ainsi que
    son nombre de données manquantes en valeur absolu et en pourcentage pour les variables master
    """
    df_missing_in_records = pd.DataFrame(columns=['index', 'nb_missing', 'percent_missing', 'nb_missing_master_only',
                                                  'percent_missing_master_only'])
    master_cols = DF_META.loc[DF_META['master']==True]['variable'].to_list()
    master_data = dataset.loc[:,master_cols]

    for index in dataset.index:
        nb_missing = dataset.loc[index].isnull().sum()
        percent_missing = round((nb_missing/len(dataset.columns))*100, 2)
        nb_missing_master_only = master_data.loc[index].isnull().sum()
        percent_missing_master_only = round((nb_missing_master_only/len(master_data.columns))*100, 2)
        
        df_missing_in_records = df_missing_in_records.append({'index':index, 'nb_missing':nb_missing,
                                                              'percent_missing':percent_missing,
                                                              'nb_missing_master_only':nb_missing_master_only,
                                                              'percent_missing_master_only':percent_missing_master_only},
                                                              ignore_index=True)
    
    return df_missing_in_records

def find_missing_in_columns(dataset, DF_META):
    df_missing_in_columns = pd.DataFrame(columns=['col', 'master', 'nb_missing', 'percent_missing'])
    for col in dataset.columns:
        if col in DF_META['variable'].values:
            if DF_META.at[DF_META.loc[DF_META['variable']==col].index[0],'nullable'] == True:
                continue
            else:                
                master =  DF_META.at[DF_META.loc[DF_META['variable']==col].index[0],'master']
                nb_missing = dataset[col].isnull().sum()
                percent_missing = round((nb_missing/len(dataset))*100, 2)
        else:
            master = False
            nb_missing = dataset[col].isnull().sum()
            percent_missing = round((nb_missing/len(dataset))*100, 2)
        
        df_missing_in_columns = df_missing_in_columns.append({'col':col,
                                                              'master':master,
                                                              'nb_missing':nb_missing,
                                                              'percent_missing':percent_missing},
                                                              ignore_index=True)
    
    return df_missing_in_columns

def find_out_of_range_values(dataset, DF_META):
    """
    Permet d'identifier les cellules hors des valeurs limite définies dans les méta-données.
    
    Input : dataset
    Output : Tableau contenant les records et les variables contenant les valeurs hors limites ainsi qu'une variable state
    montrant si la valeur est supérieur au seuil max ou inférieur au seuil min
    """
    df_out_of_range_val = pd.DataFrame(columns=['index', 'variable', 'value', 'state'])
    for index in DF_META.index:
        if DF_META.at[DF_META.index[index],'type'] != 'NUMBER':
            continue
        if DF_META.at[DF_META.index[index],'min_value'] is not None:
            min_value = DF_META.at[DF_META.index[index],'min_value']
        else:
            min_value = None
        if DF_META.at[DF_META.index[index],'max_value'] is not None:
            max_value = DF_META.at[DF_META.index[index],'max_value']
        else:
            max_value= None
        
        if (min_value is None) & (max_value is None):
            continue
        else:
            variable = DF_META.at[DF_META.index[index],'variable']
            for idx in dataset.index:
                value = dataset.at[idx,variable]
                if (value is pd.NA) | ((value >= min_value) & (value <= max_value)):
                    continue
                elif value < min_value:
                    state = "< range"
                    df_out_of_range_val = df_out_of_range_val.append({'index':idx, 'variable':variable, 'value':value,
                                                                      'state':state}, ignore_index=True)
                    continue
                elif value > max_value:
                    state = "> range"
                    df_out_of_range_val = df_out_of_range_val.append({'index':idx, 'variable':variable, 'value':value,
                                                                      'state':state}, ignore_index=True)
    return df_out_of_range_val

#https://machinelearningmastery.com/model-based-outlier-detection-and-removal-in-python/ pour qqs méthodes
def find_outliers(dataset, col, method='z_score', contamination=0.01):
    """
    Permet de trouver les outliers pour les données numériques selon plusieurs méthodes.
    
    Input : dataset, methode de détection des outliers <z_score, isolation_forest, elliptiv_envelope, local_outlier_factor>,
    taux d'outliers dans le dataset (indisponible pour la méthode z_score)
    Output : Tableau contenant pour chaque variable numérique, les records outliers ainsi que les valeurs considérées comme 
    outliers
    """
    if method == 'z_score':
        df = dataset.loc[:,[col]].dropna()
        df[col] = df[col].apply('float64') #On s'assure du type de la variable numérique
        y_hat_df = pd.DataFrame(np.abs(stats.zscore(df)), columns=['y_hat'])
        df_outliers = create_outliers_df(dataset, df, col, y_hat_df, method=method)
    
    elif method == 'isolation_forest':
        df = dataset.loc[:,[col]].dropna()
        df[col] = df[col].apply('float64')
        iso = IsolationForest(contamination=contamination)
        y_hat = iso.fit_predict(df)
        y_hat_df = pd.DataFrame(y_hat, columns=['y_hat'])
        df_outliers = create_outliers_df(dataset, df, col, y_hat_df, method=method)
        
    elif method == 'elliptic_envelope':
        df = dataset.loc[:,[col]].dropna()
        df[col] = df[col].apply('float64')
        ee = EllipticEnvelope(contamination=contamination)
        y_hat = ee.fit_predict(df)
        y_hat_df = pd.DataFrame(y_hat, columns=['y_hat'])
        df_outliers = create_outliers_df(dataset, df, col, y_hat_df, method=method)

    elif method == 'local_outlier_factor':
        df = dataset.loc[:,[col]].dropna()
        df[col] = df[col].apply('float64')
        lof = LocalOutlierFactor()
        y_hat = lof.fit_predict(df)
        y_hat_df = pd.DataFrame(y_hat, columns=['y_hat'])
        df_outliers = create_outliers_df(dataset, df, col, y_hat_df, method=method)
                
    return df_outliers

#Utilisée pour créer le dataset contenant les records outliers
def create_outliers_df(dataset, df, col, y_hat_df, method='z_score'):
    df_outliers = pd.DataFrame(columns=['index', 'variable', 'value'])

    df = df.reset_index()
    df_merge = pd.merge(df, y_hat_df, how='inner', left_index=True, right_index=True)
    for idx in df_merge.index:
        if method != 'z_score':
            if df_merge.at[df_merge.index[idx],'y_hat'] == -1:
                index = df_merge.at[df_merge.index[idx],'index']
                variable = col
                value = dataset.at[index, col]

                df_outliers = df_outliers.append({'index':index, 'variable':variable, 'value':value}, ignore_index=True)
        else:
            if df_merge.at[df_merge.index[idx],'y_hat'] > 3:
                index = df_merge.at[df_merge.index[idx],'index']
                variable = col
                value = dataset.at[index, col]

                df_outliers = df_outliers.append({'index':index, 'variable':variable, 'value':value}, ignore_index=True)
    
    return df_outliers

def find_incompatibilities(dataset, tab1, tab2, rule):
    dataset_COPY = dataset.copy()
    dataset_COPY['row_breaking_rule'] = dataset_COPY.index
    df_ifs = dataset_COPY.drop(dataset_COPY.index)
    if_tronq = tab1.loc[tab1['ruleID']==rule].reset_index(drop=True)

    for ix in if_tronq.index:
        if ix == 0:
            df_AND = dataset_COPY.copy()
            if if_tronq.at[ix, 'connecteur'] == 'egal':
                df_AND = df_AND.loc[(df_AND[if_tronq.at[ix, 'variable']]) == if_tronq.at[ix, 'comparison_value']]
                continue
            elif if_tronq.at[ix, 'connecteur'] == 'différent':
                df_AND = df_AND.loc[(df_AND[if_tronq.at[ix, 'variable']]) != if_tronq.at[ix, 'comparison_value']]
                continue
            elif if_tronq.at[ix, 'connecteur'] == 'supérieur':
                df_AND = df_AND.loc[(df_AND[if_tronq.at[ix, 'variable']]) > if_tronq.at[ix, 'comparison_value']]
                continue
            elif if_tronq.at[ix, 'connecteur'] == 'inférieur':
                df_AND = df_AND.loc[(df_AND[if_tronq.at[ix, 'variable']]) < if_tronq.at[ix, 'comparison_value']]
                continue
            elif if_tronq.at[ix, 'connecteur'] == 'supérieur ou égal':
                df_AND = df_AND.loc[(df_AND[if_tronq.at[ix, 'variable']]) >= if_tronq.at[ix, 'comparison_value']]
                continue
            elif if_tronq.at[ix, 'connecteur'] == 'inférieur ou égal':
                df_AND = df_AND.loc[(df_AND[if_tronq.at[ix, 'variable']]) <= if_tronq.at[ix, 'comparison_value']]
                continue
            elif if_tronq.at[ix, 'connecteur'] == 'commence par':
                df_AND = df_AND.loc[(df_AND[if_tronq.at[ix, 'variable']]).str.startswith(if_tronq.at[ix, 'comparison_value'], na=False)]
                continue
            elif if_tronq.at[ix, 'connecteur'] == 'fini par':
                df_AND = df_AND.loc[(df_AND[if_tronq.at[ix, 'variable']]).str.endswith(if_tronq.at[ix, 'comparison_value'], na=False)]
                continue
            elif if_tronq.at[ix, 'connecteur'] == 'contient':
                df_AND = df_AND.loc[(df_AND[if_tronq.at[ix, 'variable']]).str.contains(if_tronq.at[ix, 'comparison_value'], na=False)]
                continue
            elif if_tronq.at[ix, 'connecteur'] == 'is missing':
                df_AND = df_AND.loc[((df_AND[if_tronq.at[ix, 'variable']]).isna())]
                continue
                
        if ix > 0:
            if if_tronq.at[ix-1, 'logique'] == 'AND':
                if if_tronq.at[ix, 'connecteur'] == 'egal':
                    df_AND = df_AND.loc[(df_AND[if_tronq.at[ix, 'variable']]) == if_tronq.at[ix, 'comparison_value']]
                    continue
                elif if_tronq.at[ix, 'connecteur'] == 'différent':
                    df_AND = df_AND.loc[(df_AND[if_tronq.at[ix, 'variable']]) != if_tronq.at[ix, 'comparison_value']]
                    continue
                elif if_tronq.at[ix, 'connecteur'] == 'supérieur':
                    df_AND = df_AND.loc[(df_AND[if_tronq.at[ix, 'variable']]) > if_tronq.at[ix, 'comparison_value']]
                    continue
                elif if_tronq.at[ix, 'connecteur'] == 'inférieur':
                    df_AND = df_AND.loc[(df_AND[if_tronq.at[ix, 'variable']]) < if_tronq.at[ix, 'comparison_value']]
                    continue
                elif if_tronq.at[ix, 'connecteur'] == 'supérieur ou égal':
                    df_AND = df_AND.loc[(df_AND[if_tronq.at[ix, 'variable']]) >= if_tronq.at[ix, 'comparison_value']]
                    continue
                elif if_tronq.at[ix, 'connecteur'] == 'inférieur ou égal':
                    df_AND = df_AND.loc[(df_AND[if_tronq.at[ix, 'variable']]) <= if_tronq.at[ix, 'comparison_value']]
                    continue
                elif if_tronq.at[ix, 'connecteur'] == 'commence par':
                    df_AND = df_AND.loc[(df_AND[if_tronq.at[ix, 'variable']]).str.startswith(if_tronq.at[ix, 'comparison_value'], na=False)]
                    continue
                elif if_tronq.at[ix, 'connecteur'] == 'fini par':
                    df_AND = df_AND.loc[(df_AND[if_tronq.at[ix, 'variable']]).str.endswith(if_tronq.at[ix, 'comparison_value'], na=False)]
                    continue
                elif if_tronq.at[ix, 'connecteur'] == 'contient':
                    df_AND = df_AND.loc[(df_AND[if_tronq.at[ix, 'variable']]).str.contains(if_tronq.at[ix, 'comparison_value'], na=False)]
                    continue
                elif if_tronq.at[ix, 'connecteur'] == 'is missing':
                    df_AND = df_AND.loc[((df_AND[if_tronq.at[ix, 'variable']]).isna())]
                    continue
                    
            # if if_tronq.at[ix-1, 'logique'] == 'OR':
            #     df_ifs = pd.merge(df_ifs, df_AND, how='outer')
            #     df_AND = dataset_COPY.copy()
            #     if if_tronq.at[ix, 'connecteur'] == 'egal':
            #         df_AND = df_AND.loc[(df_AND[if_tronq.at[ix, 'variable']]) == if_tronq.at[ix, 'comparison_value']]
            #         continue
            #     elif if_tronq.at[ix, 'connecteur'] == 'différent':
            #         df_AND = df_AND.loc[(df_AND[if_tronq.at[ix, 'variable']]) != if_tronq.at[ix, 'comparison_value']]
            #         continue
            #     elif if_tronq.at[ix, 'connecteur'] == 'supérieur':
            #         df_AND = df_AND.loc[(df_AND[if_tronq.at[ix, 'variable']]) > if_tronq.at[ix, 'comparison_value']]
            #         continue
            #     elif if_tronq.at[ix, 'connecteur'] == 'inférieur':
            #         df_AND = df_AND.loc[(df_AND[if_tronq.at[ix, 'variable']]) < if_tronq.at[ix, 'comparison_value']]
            #         continue
            #     elif if_tronq.at[ix, 'connecteur'] == 'supérieur ou égal':
            #         df_AND = df_AND.loc[(df_AND[if_tronq.at[ix, 'variable']]) >= if_tronq.at[ix, 'comparison_value']]
            #         continue
            #     elif if_tronq.at[ix, 'connecteur'] == 'inférieur ou égal':
            #         df_AND = df_AND.loc[(df_AND[if_tronq.at[ix, 'variable']]) <= if_tronq.at[ix, 'comparison_value']]
            #         continue
            #     elif if_tronq.at[ix, 'connecteur'] == 'commence par':
            #         df_AND = df_AND.loc[(df_AND[if_tronq.at[ix, 'variable']]).str.startswith(if_tronq.at[ix, 'comparison_value'], na=False)]
            #         continue
            #     elif if_tronq.at[ix, 'connecteur'] == 'fini par':
            #         df_AND = df_AND.loc[(df_AND[if_tronq.at[ix, 'variable']]).str.endswith(if_tronq.at[ix, 'comparison_value'], na=False)]
            #         continue

    if if_tronq.at[ix, 'logique'] == 'THEN':
        df_ifs = pd.merge(df_ifs, df_AND, how='outer')

    df_thens = dataset_COPY.drop(dataset_COPY.index)
    then_tronq = tab2.loc[tab2['ruleID']==rule].reset_index(drop=True)

    for ix in then_tronq.index:
        if ix == 0:
            df_AND = dataset_COPY.copy()
            if then_tronq.at[ix, 'connecteur'] == 'egal':
                df_AND = df_AND.loc[(df_AND[then_tronq.at[ix, 'variable']]) == then_tronq.at[ix, 'comparison_value']]
                continue
            elif then_tronq.at[ix, 'connecteur'] == 'différent':
                df_AND = df_AND.loc[(df_AND[then_tronq.at[ix, 'variable']]) != then_tronq.at[ix, 'comparison_value']]
                continue
            elif then_tronq.at[ix, 'connecteur'] == 'supérieur':
                df_AND = df_AND.loc[(df_AND[then_tronq.at[ix, 'variable']]) > then_tronq.at[ix, 'comparison_value']]
                continue
            elif then_tronq.at[ix, 'connecteur'] == 'inférieur':
                df_AND = df_AND.loc[(df_AND[then_tronq.at[ix, 'variable']]) < then_tronq.at[ix, 'comparison_value']]
                continue
            elif then_tronq.at[ix, 'connecteur'] == 'supérieur ou égal':
                df_AND = df_AND.loc[(df_AND[then_tronq.at[ix, 'variable']]) >= then_tronq.at[ix, 'comparison_value']]
                continue
            elif then_tronq.at[ix, 'connecteur'] == 'inférieur ou égal':
                df_AND = df_AND.loc[(df_AND[then_tronq.at[ix, 'variable']]) <= then_tronq.at[ix, 'comparison_value']]
                continue
            elif then_tronq.at[ix, 'connecteur'] == 'commence par':
                df_AND = df_AND.loc[(df_AND[then_tronq.at[ix, 'variable']]).str.startswith(then_tronq.at[ix, 'comparison_value'], na=False)]
                continue
            elif then_tronq.at[ix, 'connecteur'] == 'fini par':
                df_AND = df_AND.loc[(df_AND[then_tronq.at[ix, 'variable']]).str.endswith(then_tronq.at[ix, 'comparison_value'], na=False)]
                continue
            elif if_tronq.at[ix, 'connecteur'] == 'contient':
                df_AND = df_AND.loc[(df_AND[then_tronq.at[ix, 'variable']]).str.contains(then_tronq.at[ix, 'comparison_value'], na=False)]
                continue
            elif if_tronq.at[ix, 'connecteur'] == 'is missing':
                df_AND = df_AND.loc[((df_AND[then_tronq.at[ix, 'variable']]).isna())]
                continue
                
        if ix > 0:
            if then_tronq.at[ix-1, 'logique'] == 'AND':
                if then_tronq.at[ix, 'connecteur'] == 'egal':
                    df_AND = df_AND.loc[(df_AND[then_tronq.at[ix, 'variable']]) == then_tronq.at[ix, 'comparison_value']]
                    continue
                elif then_tronq.at[ix, 'connecteur'] == 'différent':
                    df_AND = df_AND.loc[(df_AND[then_tronq.at[ix, 'variable']]) != then_tronq.at[ix, 'comparison_value']]
                    continue
                elif then_tronq.at[ix, 'connecteur'] == 'supérieur':
                    df_AND = df_AND.loc[(df_AND[then_tronq.at[ix, 'variable']]) > then_tronq.at[ix, 'comparison_value']]
                    continue
                elif then_tronq.at[ix, 'connecteur'] == 'inférieur':
                    df_AND = df_AND.loc[(df_AND[then_tronq.at[ix, 'variable']]) < then_tronq.at[ix, 'comparison_value']]
                    continue
                elif then_tronq.at[ix, 'connecteur'] == 'supérieur ou égal':
                    df_AND = df_AND.loc[(df_AND[then_tronq.at[ix, 'variable']]) >= then_tronq.at[ix, 'comparison_value']]
                    continue
                elif then_tronq.at[ix, 'connecteur'] == 'inférieur ou égal':
                    df_AND = df_AND.loc[(df_AND[then_tronq.at[ix, 'variable']]) <= then_tronq.at[ix, 'comparison_value']]
                    continue
                elif then_tronq.at[ix, 'connecteur'] == 'commence par':
                    df_AND = df_AND.loc[(df_AND[then_tronq.at[ix, 'variable']]).str.startswith(then_tronq.at[ix, 'comparison_value'], na=False)]
                    continue
                elif then_tronq.at[ix, 'connecteur'] == 'fini par':
                    df_AND = df_AND.loc[(df_AND[then_tronq.at[ix, 'variable']]).str.endswith(then_tronq.at[ix, 'comparison_value'], na=False)]
                    continue
                elif if_tronq.at[ix, 'connecteur'] == 'contient':
                    df_AND = df_AND.loc[(df_AND[then_tronq.at[ix, 'variable']]).str.contains(then_tronq.at[ix, 'comparison_value'], na=False)]
                    continue
                elif if_tronq.at[ix, 'connecteur'] == 'is missing':
                    df_AND = df_AND.loc[((df_AND[then_tronq.at[ix, 'variable']]).isna())]
                    continue
                    
            # if then_tronq.at[ix-1, 'logique'] == 'OR':
            #     df_thens = pd.merge(df_thens, df_AND, how='outer')
            #     df_AND = dataset_COPY.copy()
            #     if then_tronq.at[ix, 'connecteur'] == 'egal':
            #         df_AND = df_AND.loc[(df_AND[then_tronq.at[ix, 'variable']]) == then_tronq.at[ix, 'comparison_value']]
            #         continue
            #     elif then_tronq.at[ix, 'connecteur'] == 'différent':
            #         df_AND = df_AND.loc[(df_AND[then_tronq.at[ix, 'variable']]) != then_tronq.at[ix, 'comparison_value']]
            #         continue
            #     elif then_tronq.at[ix, 'connecteur'] == 'supérieur':
            #         df_AND = df_AND.loc[(df_AND[then_tronq.at[ix, 'variable']]) > then_tronq.at[ix, 'comparison_value']]
            #         continue
            #     elif then_tronq.at[ix, 'connecteur'] == 'inférieur':
            #         df_AND = df_AND.loc[(df_AND[then_tronq.at[ix, 'variable']]) < then_tronq.at[ix, 'comparison_value']]
            #         continue
            #     elif then_tronq.at[ix, 'connecteur'] == 'supérieur ou égal':
            #         df_AND = df_AND.loc[(df_AND[then_tronq.at[ix, 'variable']]) >= then_tronq.at[ix, 'comparison_value']]
            #         continue
            #     elif then_tronq.at[ix, 'connecteur'] == 'inférieur ou égal':
            #         df_AND = df_AND.loc[(df_AND[then_tronq.at[ix, 'variable']]) <= then_tronq.at[ix, 'comparison_value']]
            #         continue
            #     elif then_tronq.at[ix, 'connecteur'] == 'commence par':
            #         df_AND = df_AND.loc[(df_AND[then_tronq.at[ix, 'variable']]).str.startswith(then_tronq.at[ix, 'comparison_value'], na=False)]
            #         continue
            #     elif then_tronq.at[ix, 'connecteur'] == 'fini par':
            #         df_AND = df_AND.loc[(df_AND[then_tronq.at[ix, 'variable']]).str.endswith(then_tronq.at[ix, 'comparison_value'], na=False)]
            #         continue

    if then_tronq.at[ix, 'logique'] == 'END':
        df_thens = pd.merge(df_thens, df_AND, how='outer')
    
    df_ifs['FLAG_FOR_RULES'] = np.where(df_ifs['row_breaking_rule'].isin(df_thens['row_breaking_rule'].to_list()), False, True)
    DF_RULE = df_ifs.loc[df_ifs['FLAG_FOR_RULES']==True].drop(columns=['FLAG_FOR_RULES']).set_index('row_breaking_rule')
    DF_RULE = DF_RULE.reset_index()
    
    return DF_RULE

def write_rule(DF_IF, DF_THEN, rule):
    phrase ='SI '
    if_tronq = DF_IF.loc[DF_IF['ruleID']==rule]
    then_tronq = DF_THEN.loc[DF_THEN['ruleID']==rule]
    for row in if_tronq.index:
        phrase = phrase+if_tronq.at[row, 'variable']+' '+if_tronq.at[row, 'connecteur']+' '+str(if_tronq.at[row, 'comparison_value'])+' '+if_tronq.at[row, 'logique']+' '
    for row in then_tronq.index:
        phrase = phrase+then_tronq.at[row, 'variable']+' '+then_tronq.at[row, 'connecteur']+' '+str(then_tronq.at[row, 'comparison_value'])+' '+then_tronq.at[row, 'logique']+' '
    phrase = phrase.split(' END ')[0]
    phrase = phrase.replace('nan ', '') #Si contient 'nan', on l'enlève
    
    return phrase

def find_temporal_errors(dataset, DF_TEMP, rule):
    cle = DF_TEMP.at[rule, 'clé']
    if len(cle) > 1:
        groupby_obj = cle[0]
        secondary_keys = cle[1:]
    else:
        groupby_obj = cle[0]
        secondary_keys = []
    var_temp = [DF_TEMP.at[rule, 'variable_temporelle']] + secondary_keys
    dataset_copy = dataset.copy()
    dataset_copy = dataset_copy.reset_index().dropna()
    dataset_copy = dataset_copy.groupby([groupby_obj]).apply(lambda x: (x.sort_values(var_temp))).reset_index(drop=True)

    dataset_copy['flag'] = False
    for ix in range(len(dataset_copy)-1):
        if dataset_copy.at[ix, groupby_obj] != dataset_copy.at[ix+1, groupby_obj]:
            continue
        else:
            if dataset_copy.at[ix, 'index'] > dataset_copy.at[ix+1, 'index']:
                dataset_copy['flag'][ix] = True
                dataset_copy['flag'][ix+1] = True
                if ix != 0:
                    if dataset_copy.at[ix, groupby_obj] == dataset_copy.at[ix-1, groupby_obj]:
                        dataset_copy['flag'][ix-1] = True

    df_flag = dataset_copy.loc[dataset_copy.flag == True, dataset_copy.drop(columns=['flag']).columns].set_index(['index']).sort_index()
    df_flag = df_flag.reset_index()
    df_flag = df_flag.rename(columns={"index": "row_breaking_rule"})
    
    return df_flag














