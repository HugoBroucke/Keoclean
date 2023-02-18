import streamlit as st
import fonctions.general_fun as gf
import fonctions.diagnosis_fun as df
import fonctions.treatment_fun as tf
import pandas as pd
import numpy as np
import altair as alt

def treatment_missing(state):
    st.header("Traitement des données manquantes")
    gf.check_data_init(state.dataset_clean)
    with st.beta_expander("Traitement des lignes avec trop de données manquantes"):  
        thr_rec = st.number_input("Entrer le pourcentage de données manquantes minimum", min_value=0, max_value=100, value=int(50), key='ni23')
        master_rec = st.checkbox("Réaliser l'analyse en prenant en compte les variables master ?", key='check17')
        if master_rec == True:
            thr_master_rec = st.number_input("Entrer le pourcentage de données MASTER manquantes minimum", min_value=0, max_value=100, value=int(30), key='ni24')
        else :
            thr_master_rec = None
            
        if st.button("Démarrer l'analyse", key='but54'):
            df_missing_in_rec = df.find_missing_in_records(state.dataset_clean, state.DF_META)
            st.write('Valeurs manquantes dans chaque lignes')
            if master_rec == True:
                state.df_lig_mis = df_missing_in_rec.loc[(df_missing_in_rec['percent_missing'] >= thr_rec) & (df_missing_in_rec['percent_missing_master_only'] >= thr_master_rec)]

            else:
                state.df_lig_mis = df_missing_in_rec.loc[df_missing_in_rec['percent_missing'] >= thr_rec]
            state.b3onclick = True
            
        if state.b3onclick == True:
            st.markdown("<h3 style='text-align: center; '>Liste des enregistrements avec trop de données manquantes</h1>", unsafe_allow_html=True)
            grid_response = gf.create_grid(state.df_lig_mis)
            data = grid_response['data']
            selected = grid_response['selected_rows']
            df_filtered = pd.DataFrame(selected)
            
            miss_l1, miss_l2 = st.beta_columns(2)
            with miss_l1:
                if st.button("Supprimer toutes les lignes identifiées", key='but55'):
                    #state.df_precedent_state = state.dataset_clean.copy()
                    nb_mis = len(data['index'].values)
                    nature_modif = "Supression des lignes avec trop de données manquantes"
                    if master_rec == True:
                        modif = "Suppression de "+str(nb_mis)+" lignes avec au moins "+str(thr_rec)+" % de données manquantes et "+str(thr_master_rec)+" % de données master manquantes"
                    else:
                        modif = "Suppression de "+str(nb_mis)+" lignes avec au moins "+str(thr_rec)+" % de données manquantes"
                    state.dataset_clean = state.dataset_clean.drop(data['index'].values)
                    state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                    state.b3onclick = False
            with miss_l2:
                if st.button("Supprimer les lignes sélectionnées", key='but56'):
                    nb_mis = len(df_filtered['index'].values)
                    nature_modif = "Supression de certaines lignes avec trop de données manquantes"
                    if master_rec == True:
                        modif = "Suppression de "+str(nb_mis)+" lignes avec au moins "+str(thr_rec)+" % de données manquantes et "+str(thr_master_rec)+" % de données master manquantes"
                    else:
                        modif = "Suppression de "+str(nb_mis)+" lignes avec au moins "+str(thr_rec)+" % de données manquantes"
                    state.dataset_clean = state.dataset_clean.drop(df_filtered['index'].values)
                    state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                    state.df_lig_mis = state.df_lig_mis.drop(state.df_lig_mis.loc[state.df_lig_mis['index'].isin(df_filtered['index'].values) == True].index)
            
            col_to_analyse = st.selectbox("Choisir la colonne à analyser", state.dataset_clean.columns)
            if not df_filtered.empty:
                df_chart = state.dataset_clean.loc[state.dataset_clean.index.isin(data['index'].values), [col_to_analyse]]
                df_chart[col_to_analyse] = df_chart[col_to_analyse].fillna("NaN")
                df_chart['selected'] = np.where(df_chart.index.isin(df_filtered['index'].values), 'Oui', 'Non')
    
                chart = alt.Chart(df_chart).mark_bar().encode(
                    x=alt.X(col_to_analyse, sort=alt.EncodingSortField(field=col_to_analyse, op="count", order='descending')),
                    y='count('+col_to_analyse+')',
                    color='selected'
                )
            
            else:
                df_chart = state.dataset_clean.loc[state.dataset_clean.index.isin(data['index'].values), [col_to_analyse]]
                df_chart[col_to_analyse] = df_chart[col_to_analyse].fillna("NaN")
                chart = alt.Chart(df_chart).mark_bar().encode(
                    x=alt.X(col_to_analyse, sort=alt.EncodingSortField(field=col_to_analyse, op="count", order='descending')),
                    y='count('+col_to_analyse+')'
                )
                
            st.markdown("<h3 style='text-align: center; '>Analyse des modalités de la variable "+col_to_analyse+" pour les enregistrements avec trop de données manquantes</h1>", unsafe_allow_html=True)
            st.altair_chart(chart, use_container_width=True)
        ##### TRAITEMENT DES COLONNES AVEC TROP DE DONNEES MANQUANTES #####
       
    with st.beta_expander("Traitement des colonnes avec trop de données manquantes"):  
        thr_col = st.number_input("Entrer le pourcentage de données manquantes minimum", min_value=0, max_value=100, value=int(50), key='ni25')
        master_col = st.checkbox("Réaliser l'analyse en prenant en compte les variables master ?", key='check18')
        if master_col == True:
            thr_master_col = st.number_input("Entrer le pourcentage de données MASTER manquantes minimum", min_value=0, max_value=100, value=int(30), key='ni26')
        else :
            thr_master_col = None
            
        if st.button("Démarrer l'analyse", key='but57'):
            df_missing_in_col = df.find_missing_in_columns(state.dataset_clean, state.DF_META)
            st.write('Valeurs manquantes dans chaque colonne')
            if master_col == True:
                state.df_col_mis = df_missing_in_col.loc[((df_missing_in_col['percent_missing'] >= thr_col) & (df_missing_in_col['master'] == False)) |
                                                         ((df_missing_in_col['percent_missing'] >= thr_master_col) & (df_missing_in_col['master'] == True))]

            else:
                state.df_col_mis = df_missing_in_col.loc[df_missing_in_col['percent_missing'] >= thr_col]
            state.b4onclick = True
            
        if state.b4onclick == True:
            st.markdown("<h3 style='text-align: center; '>Liste des colonnes avec trop de données manquantes</h1>", unsafe_allow_html=True)
            grid_response = gf.create_grid(state.df_col_mis)
            data = grid_response['data']
            selected = grid_response['selected_rows']
            df_filtered = pd.DataFrame(selected)
            
            miss_l1, miss_l2 = st.beta_columns(2)
            with miss_l1:
                if st.button("Supprimer toutes les colonnes identifiées", key='but58'):
                    #state.df_precedent_state = state.dataset_clean.copy()
                    nature_modif = "Supression des colonnes avec trop de données manquantes"
                    modif = "Suppression des colonnes: "
                    for v in data['col'].to_list():
                        add_var = v+' '
                        modif = modif + add_var
                    if master_col == True:
                        modif = modif+" possédant au moins "+str(thr_col)+" % de données manquantes et "+str(master_col)+" % de données manquantes s'il s'agit d'une variable MASTER"
                    else:
                        modif = modif+" possédant au moins "+str(thr_col)+" % de données manquantes"
                    state.dataset_clean = state.dataset_clean.drop(columns=data['col'].to_list())
                    state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                    state.b4onclick = False
            with miss_l2:
                if st.button("Supprimer les colonnes sélectionnées", key='but59'):
                    #state.df_precedent_state = state.dataset_clean.copy()
                    nature_modif = "Supression de certaines colonnes avec trop de données manquantes"
                    modif = "Suppression des colonnes: "
                    for v in df_filtered['col'].to_list():
                        add_var = v+' '
                        modif = modif + add_var
                    if master_col == True:
                        modif = modif+" possédant au moins "+str(thr_col)+" % de données manquantes et "+str(master_col)+" % de données manquantes s'il s'agit d'une variable MASTER"
                    else:
                        modif = modif+" possédant au moins "+str(thr_col)+" % de données manquantes"
                    state.dataset_clean = state.dataset_clean.drop(columns=df_filtered['col'].to_list())
                    state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                    state.df_col_mis = state.df_col_mis.drop(state.df_col_mis.loc[state.df_col_mis['col'].isin(df_filtered['col'].values) == True].index)
                        
                        ##### IMPUTATION UNIVARIE #####
    with st.beta_expander("Inputation univariée d'une variable"):  
        selected_col = st.selectbox("Choisir une variable à imputer", state.dataset_clean.columns, key='sb41')
        var_type = gf.return_type(state.dataset_clean, selected_col)
        
        if var_type == 'float64':
            methods = ['value', 'forward_fill', 'bacward_fill', 'mean', 'median', 'mode', 'rolling_avg']
            selected_method = st.selectbox("Choisir une méthode d'inputation univariée", methods, key='sb42')
        else :
            methods = ['value', 'forward_fill', 'bacward_fill', 'mode']
            selected_method = st.selectbox("Choisir une méthode d'inputation univariée", methods, key='sb42')
        
        if selected_method == 'value':     
            if var_type == 'object':
                value = st.text_input("Entrer la valeur de remplacement", key='ti11')
            elif var_type == 'datetime64[ns]':
                value = st.date_input("Entrer la valeur de remplacement", key='di5')
            else:
                value = st.number_input("Entrer la valeur de remplacement", key='ni27')
        else:
            value = None
        
        if selected_method == 'rolling_avg':
            window = st.number_input("Entrer la taille de la fenêtre glissante", value=int(10), key='ni28')
            min_periods = st.number_input("Entrer le nombre minimum d'observations dans la fenêtre", value=int(3), key='ni29')
        else:
            window = None
            min_periods = None
                
        if st.button("Imputer la variable", key='but60'):
            #state.df_precedent_state = state.dataset_clean.copy()
            nb_mis = state.dataset_clean[selected_col].isnull().sum()
            nature_modif = "Imputation d'une colonne avec trop de données manquantes"
            if selected_method != 'value':
                modif = "Imputation de "+str(nb_mis)+" valeurs manquantes pour la colonne "+selected_col+" par la méthode "+selected_method
            else : 
                modif = "Imputation de "+str(nb_mis)+" valeurs manquantes pour la colonne "+selected_col+" par la valeur "+str(value)
            state.dataset_clean = tf.fill_missing_univariate(state.dataset_clean, selected_col, method=selected_method, value=value, window=window, min_periods=min_periods)
            state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
            st.success('Modification effectuée')
            
    if state.display_df == True:
        gf.display_data(state.dataset_clean)

    if state.reorder_cols == True:
        state.dataset_clean, state.reorder_var, state.reorder_how = gf.reorder_columns(state.dataset_clean, state.reorder_var, state.reorder_how)

    if state.change_value == True:
        state.dataset_clean = gf.change_cell_value(state.dataset_clean)
        
    if state.display_state_df == True:
        state.dataset_clean, state.DF_STATE = gf.display_df_states(state.dataset_clean, state.DF_STATE)