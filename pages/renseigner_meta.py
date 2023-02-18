import streamlit as st
import pandas as pd
import datetime
import fonctions.screening_fun as sf
import fonctions.general_fun as gf
import numpy as np
import time

def screening_meta(state):    
    st.header("Renseigner les métadonnées")      
    gf.check_data_init(state.dataset_clean)    
    modif_multi = st.checkbox("Modifier plusieurs variables en même temps ?", key='check1001')
    if modif_multi == True:
        selected_vars = st.multiselect("Sélectionner les colonnes à modifier", options=list(state.dataset_clean.columns), default=list(state.dataset_clean.columns))
    else:
        selected_var = st.selectbox("Choix de la variable", state.dataset_clean.columns, key='sb24')
    type_list = ['STRING', 'NUMBER', 'DATE']
    selected_type = st.selectbox("Renseigner le type", type_list, key='sb25')
    container = st.beta_container()
    col1, col2 = st.beta_columns([2,3])
    with col1:
        nullable = st.checkbox("Variable nullable ?", key='check7')
        boolean = st.checkbox("Variable booléenne ?", key='check8')
        unique = st.checkbox("Variable clé primaire ?", key='check9')
        master = st.checkbox("Variable master ?", key='check10')
        min_value = None
        max_value = None   
        
        #Ajout de l'étendue pour variable numérique
        if selected_type == 'NUMBER':
            if st.checkbox("Ajouter une étendue ?", key='check11') == True:
                min_value = st.number_input("Entrer la valeur minimale", key='ni8')
                max_value = st.number_input("Entrer la valeur maximale", min_value=min_value, key='ni9')
        
        #Ajout de l'étendue pour variable date
        elif selected_type == 'DATE':
            format_date = container.checkbox("Définir manuellement le format de la date ?", key='check1000')
            if format_date == True:
                container.info("Exemple: pour la date 20150601000000, le format a sélectionner est %Y%m%d%H%M%S \n [Lien vers la documentation](https://docs.python.org/fr/3/library/datetime.html)")
                formated_date = container.text_input("Entrer le format désiré", value="%Y%m%d")
            if st.checkbox("Ajouter une étendue ?", key='check12') == True:
                min_value = st.date_input("Entrer la date initiale", max_value=max_value, min_value=datetime.date(1800, 1, 1), key='di3')
                max_value = st.date_input("Entrer la date finale", min_value=min_value, key='di4')
        
        elif selected_type == 'STRING':
            min_value = None
            max_value = None
    
    
    with col2:
        val_to_exclude = st.text_input("Entrer une valeur à exclure", key='ti8')
        if st.button('Ajouter aux valeurs exclues', key='but26') : 
            if modif_multi == True:
                for selected_var in selected_vars:
                    state.dic_excluded_values = gf.add_value_to_dic(state.dic_excluded_values, key=selected_var, value=val_to_exclude, i=set())
                    #exclude_values = state.dic_excluded_values.get(selected_var)
                    st.success(val_to_exclude + " a été ajouté aux valeurs exclues pour la variable: "+selected_var)
            else:
                state.dic_excluded_values = gf.add_value_to_dic(state.dic_excluded_values, key=selected_var, value=val_to_exclude, i=set())
                st.success(val_to_exclude + " a été ajouté aux valeurs exclues pour la variable: "+selected_var)
                #exclude_values = state.dic_excluded_values.get(selected_var)
    
        
    
    st.write("Note: La notion de variable Master est subjective, il s'agit d'une variable d'une plus grande importance dans le dataset")
    
    st.markdown("<h3 style='text-align: center; '>Table des métadonnées</h1>", unsafe_allow_html=True)       
    st.dataframe(state.DF_META)

    if st.button("Ajouter la variable aux métadonnées", key='but27'):
        if modif_multi == True:
            for selected_var in selected_vars:
                exclude_values = state.dic_excluded_values.get(selected_var)
                if selected_var in state.DF_META['variable'].to_list():
                    state.DF_META.drop(state.DF_META.loc[state.DF_META.variable == selected_var].index, inplace=True)
                
                state.DF_META = sf.update_META_DATA(state.DF_META, variable = selected_var, data_type = selected_type, nullable=nullable, boolean=boolean,
                                                    unique=unique, master=master, min_value=min_value, max_value=max_value, exclude_values=exclude_values)
                
                
                if selected_type == 'NUMBER':
                    state.dataset_clean[selected_var] = state.dataset_clean[selected_var].astype('float64')
                if selected_type == 'STRING':         
                    try:
                        state.dataset_clean[selected_var] = state.dataset_clean[selected_var].astype('Int64').astype('str').replace('<NA>', np.nan)
                    except : 
                        state.dataset_clean[selected_var] = state.dataset_clean[selected_var].astype(object)
                if selected_type == 'DATE':
                    if format_date == False:
                        state.dataset_clean[selected_var] = pd.to_datetime(state.dataset_clean[selected_var], errors='coerce', infer_datetime_format=True)
                    else:
                        for i in state.dataset_clean.index:
                            try:
                                state.dataset_clean.at[i,selected_var] = pd.to_datetime(state.dataset_clean.at[i,selected_var], errors='raise', format=formated_date)
                            except:
                                state.dataset_clean.at[i,selected_var] = np.nan
                        state.dataset_clean[selected_var] = pd.to_datetime(state.dataset_clean[selected_var], errors='coerce', infer_datetime_format=True)
        else:
            exclude_values = state.dic_excluded_values.get(selected_var)
            if selected_var in state.DF_META['variable'].to_list():
                state.DF_META.drop(state.DF_META.loc[state.DF_META.variable == selected_var].index, inplace=True)
            
            state.DF_META = sf.update_META_DATA(state.DF_META, variable = selected_var, data_type = selected_type, nullable=nullable, boolean=boolean,
                                                unique=unique, master=master, min_value=min_value, max_value=max_value, exclude_values=exclude_values)
            
            
            if selected_type == 'NUMBER':
                state.dataset_clean[selected_var] = state.dataset_clean[selected_var].astype('float64')
            if selected_type == 'STRING':         
                try:
                    state.dataset_clean[selected_var] = state.dataset_clean[selected_var].astype('Int64').astype('str').replace('<NA>', np.nan)
                except : 
                    state.dataset_clean[selected_var] = state.dataset_clean[selected_var].astype(object)
            if selected_type == 'DATE':
                if format_date == False:
                    state.dataset_clean[selected_var] = pd.to_datetime(state.dataset_clean[selected_var], errors='coerce', infer_datetime_format=True)
                else:
                    for i in state.dataset_clean.index:
                        try:
                            state.dataset_clean.at[i,selected_var] = pd.to_datetime(state.dataset_clean.at[i,selected_var], errors='raise', format=formated_date)
                        except:
                            state.dataset_clean.at[i,selected_var] = np.nan
                    state.dataset_clean[selected_var] = pd.to_datetime(state.dataset_clean[selected_var], errors='coerce', infer_datetime_format=True)

    if state.display_df == True:
        gf.display_data(state.dataset_clean)
        
    if state.reorder_cols == True:
        state.dataset_clean, state.reorder_var, state.reorder_how = gf.reorder_columns(state.dataset_clean, state.reorder_var, state.reorder_how)
        
    if state.change_value == True:
        state.dataset_clean = gf.change_cell_value(state.dataset_clean)
        
    if state.display_state_df == True:
        state.dataset_clean, state.DF_STATE = gf.display_df_states(state.dataset_clean, state.DF_STATE)