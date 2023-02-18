import fonctions.general_fun as gf
import numpy as np
import pandas as pd

def init_META(df, DF_META):
    for col in df.columns:
        data_type = gf.return_type(df, col)
        if data_type == 'object':
            DF_META = update_META_DATA(DF_META, col, 'STRING')
            try:
                df[col] = df[col].astype('Int64').astype('str').replace('<NA>', np.nan)
            except : 
                df[col] = df[col].astype(object)
        elif data_type == 'datetime64[ns]':
            DF_META = update_META_DATA(DF_META, col, 'DATE')
            df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
        else:
            DF_META = update_META_DATA(DF_META, col, 'NUMBER')
            df[col] = df[col].astype('float64')
    
    return df, DF_META

def update_META_DATA(DF_META, variable, data_type, nullable=False, boolean=False, unique=False, master=False, min_value=None,
                     max_value=None, authorized_values= None, exclude_values=None):
    """
    Modifie la table des méta-données
    
    Input : variable, type de la variable, la variable peut elle avoir des valeurs manquantes?, la variable est-elle booléenne?
    , la variable est-elle clé primaire?, la variable est elle une variable master ?
    
    """
    
    if data_type == 'STRING':        
        DF_META = DF_META.append({'variable':variable, 'type':data_type, 'master':master, 'nullable':nullable,
                                  'boolean':boolean,'unique':unique, 'min_value':None, 'max_value':None,'exclude_values':exclude_values},
                                 ignore_index=True)
        
    elif data_type == 'NUMBER':

        DF_META = DF_META.append({'variable':variable, 'type':data_type, 'master':master, 'nullable':nullable,
                                  'boolean':boolean,'unique':unique, 'min_value':min_value, 'max_value':max_value, 'exclude_values':exclude_values},
                                 ignore_index=True)
    
    elif data_type == 'DATE':

        DF_META = DF_META.append({'variable':variable, 'type':data_type, 'master':master, 'nullable':nullable,
                                  'boolean':boolean,'unique':unique, 'min_value':min_value, 'max_value':max_value, 'exclude_values':exclude_values},
                                 ignore_index=True)
    
    return DF_META

def rapid_analysis(df):
    returned_df = pd.DataFrame(columns=['Variable', 'Type détecté', 'Nb missing (%)', 'Nb unique'])
    int_df = pd.DataFrame(columns=['Variable', 'Mean', 'Min', '1st Quart', 'Median', '3rd Quart', 'Max'])
    obj_df = pd.DataFrame(columns=['Variable', 'Modalité 1', 'Modalité 2', 'Modalité 3', 'Modalité 4', 'Modalité 5'])
    len_df = len(df)
    nb_cols = len(df.columns)
    for col in df.columns:
        col_type = gf.return_type(df, col)
        nb_missing = df[col].isna().sum()
        nb_missing_percent = round((nb_missing / len_df*100), 2)
        nb_unique = len(df[col].unique())
        returned_df = returned_df.append({'Variable':col,
                                          'Type détecté':col_type,
                                          'Nb missing (%)': str(nb_missing)+' ('+str(nb_missing_percent)+' %)',
                                          'Nb unique':nb_unique}, ignore_index=True)
        if nb_missing ==len_df:
            continue
        
        if (col_type == 'object') | (col_type =='datetime64[ns]'):
            df_copy = df.loc[:,[col]]
            df_copy = df_copy.fillna('NaN')
            grouped = df_copy.reset_index().groupby([col]).count().rename(columns={"index": "Pourcentage"}).sort_values(by=["Pourcentage"], ascending=False)
            grouped["Pourcentage"] = round((grouped["Pourcentage"] / len(df_copy))*100,2)
            grouped.reset_index(inplace=True)
            
            modalite1 = grouped.at[0, col]
            pourcentage1 = grouped.at[0, 'Pourcentage']
            
            try:
                modalite2 = grouped.at[1, col]
                pourcentage2 = grouped.at[1, 'Pourcentage']
            except:
                modalite2 = 'XXXXX NAN XXXXX'
                pourcentage2= 'XXXXX NAN XXXXX'
            try:
                modalite3 = grouped.at[2, col]
                pourcentage3 = grouped.at[2, 'Pourcentage']
            except:
                modalite3 = 'XXXXX NAN XXXXX'
                pourcentage3= 'XXXXX NAN XXXXX'
            try:
                modalite4 = grouped.at[3, col]
                pourcentage4 = grouped.at[3, 'Pourcentage']
            except:
                modalite4 = 'XXXXX NAN XXXXX'
                pourcentage4= 'XXXXX NAN XXXXX'
            try:
                modalite5 = grouped.at[4, col]
                pourcentage5 = grouped.at[4, 'Pourcentage']
            except:
                modalite5 = 'XXXXX NAN XXXXX'
                pourcentage5= 'XXXXX NAN XXXXX'
                
            obj_df = obj_df.append({'Variable': col, 'Modalité 1':"("+str(pourcentage1)+" %)"+" "+str(modalite1),
                                    'Modalité 2':"("+str(pourcentage2)+" %)"+" "+str(modalite2), 
                                    'Modalité 3':"("+str(pourcentage3)+" %)"+" "+str(modalite3), 
                                    'Modalité 4':"("+str(pourcentage4)+" %)"+" "+str(modalite4), 
                                    'Modalité 5':"("+str(pourcentage5)+" %)"+" "+str(modalite5)}, ignore_index=True)
        else:
            mean = df[col].dropna().mean()
            mini = df[col].dropna().min()
            quart1 = df[col].dropna().quantile(.25)
            median = df[col].dropna().quantile(.5)
            quart3 = df[col].dropna().quantile(.75)
            maxi = df[col].dropna().max()
            
            int_df = int_df.append({'Variable':col, 'Mean':mean, 'Min':mini, '1st Quart':quart1, 'Median':median, '3rd Quart':quart3, 'Max':maxi},
                                   ignore_index=True)
            
        obj_df['Modalité 2'] = np.where(obj_df['Modalité 2'] == "(XXXXX NAN XXXXX %) XXXXX NAN XXXXX", "Pas d'autres modalités", obj_df['Modalité 2'])
        obj_df['Modalité 3'] = np.where(obj_df['Modalité 3'] == "(XXXXX NAN XXXXX %) XXXXX NAN XXXXX", "Pas d'autres modalités", obj_df['Modalité 3'])
        obj_df['Modalité 4'] = np.where(obj_df['Modalité 4'] == "(XXXXX NAN XXXXX %) XXXXX NAN XXXXX", "Pas d'autres modalités", obj_df['Modalité 4'])
        obj_df['Modalité 5'] = np.where(obj_df['Modalité 5'] == "(XXXXX NAN XXXXX %) XXXXX NAN XXXXX", "Pas d'autres modalités", obj_df['Modalité 5'])
    
    return returned_df, len_df, nb_cols, int_df, obj_df