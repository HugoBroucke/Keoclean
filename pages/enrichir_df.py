import streamlit as st
import fonctions.general_fun as gf
import fonctions.treatment_fun as tf
import chardet
import os
import os.path
import csv
import pandas as pd

def enrich_df(state):
    st.header("Enrichir à partir d'un autre dataset")
    gf.check_data_init(state.dataset_clean)
    uploaded_newfile = st.file_uploader("Choisir un fichier CSV", key='fu2')
    header = st.number_input("Entrer la ligne d'en tête du fichier", 0, key='ni6')
    encod_bool = st.checkbox("Définir manuellement l'encodage ?", key='check4')
    delim_bool = st.checkbox("Définir manuellement le délimiteur ?", key='check5')
    
    if uploaded_newfile is not None:
        if os.path.isfile("tempDir/"+uploaded_newfile.name):
            print('File exists')
        else:
            with st.spinner('Veuillez patienter, le fichier sera bientôt chargé !'):
                gf.save_uploaded_file(uploaded_newfile) 
        
    if encod_bool == True:
        encoding = st.text_input("Entrer l'encodage'du fichier CSV", key='ti5')
    else:
        if uploaded_newfile is not None:
            with open("tempDir/"+uploaded_newfile.name, 'rb') as rawdata:
                result = chardet.detect(rawdata.read(1000))
                encoding = result['encoding']
    
    if delim_bool == True:
        sep = st.text_input("Entrer le délimiteur du fichier csv", key='ti6')
    else:
        if uploaded_newfile is not None:
            with open("tempDir/"+uploaded_newfile.name, 'r') as csvfile:
                dialect = csv.Sniffer().sniff(csvfile.readline(1000))
                sep = dialect.delimiter
    
    if uploaded_newfile is not None:
        st.warning("Encodage détecté : "+encoding+" --- Séparateur détecté : "+'"'+sep+'"')
        try:
            with st.spinner("Chargement du dataset, weuillez patienter ..."):
                state.dataset_added = pd.read_csv(uploaded_newfile, header=header, encoding=encoding, na_values='', sep=sep)
                filtered = st.multiselect("Filter columns", options=list(state.dataset_added.columns), default=list(state.dataset_added.columns))
                st.markdown("<h3 style='text-align: center; '>Aperçu du dataset à fusionner</h1>", unsafe_allow_html=True)

                col1, col2, col3 = st.beta_columns([1,2,1])
                with col2:
                    st.dataframe(state.dataset_added[filtered].head(100))
        except:
            st.error("Le dataset n'a pas pu être chargé, veuillez modifier manuellement l'encodage et/ou le délimiteur")

        method = ['Merge dataset', 'Concaténer dataset']
        method_selected = st.selectbox("Choix de la méthode", method, key='sb12')
        if method_selected == 'Merge dataset':
            st.info("Mapper les variables servant à réaliser la jointure")
        else:
            st.info("Mapper les colonnes similaires (nécessaire uniquement si les colonnes n'ont pas le même nom)")
        
        b1, b2 = st.beta_columns(2)
        with b1:
            key_left = st.selectbox("Choix de la variable de la table d'origine à ajouter comme clé de fusion", state.dataset_clean.columns, key='sb13')
            
        with b2:
            key_right = st.selectbox("Choix de la variable de la table à fusionner à ajouter comme clé de fusion", state.dataset_added[filtered].columns, key='sb14')
        
        if st.button("Ajouter", key='but13'):
            if key_left in state.df_cles['Mapping_df_origine'].values:
                 state.df_cles=state.df_cles.drop(state.df_cles.loc[state.df_cles['Mapping_df_origine'] == key_left].index)
            if key_right in state.df_cles['Mapping_nouveau_df'].values:
                 state.df_cles=state.df_cles.drop(state.df_cles.loc[state.df_cles['Mapping_nouveau_df'] == key_right].index)
            state.df_cles = state.df_cles.append({'Mapping_df_origine':key_left, 'Mapping_nouveau_df':key_right}, ignore_index=True)
        

        col4, col5, col6 = st.beta_columns([1,2,1])
        with col5:
            st.markdown("<h3 style='text-align: center; '>Mapping des variables</h3>", unsafe_allow_html=True)
            st.dataframe(state.df_cles)
            
            ##### MERGE #####
        
        col7, col8, col9 = st.beta_columns([2,1,2])        
        if method_selected == 'Merge dataset':
            with col7:
                if st.button("Fusionner les colonnes sélectionnées", key='but14'):
                    if state.df_cles.empty:
                        st.error('Veuillez renseigner les clés de fusion')                       
                    else:
                        state.dataset_added[filtered] = tf.map_type(state.dataset_clean, state.dataset_added[filtered], state.df_cles)
                        nature_modif = "Fusion de colonnes à partir d'un autre dataset"
                        nb_col_before = len(state.dataset_clean.columns)
                        allcols = state.dataset_clean.columns.append(state.dataset_added[filtered].columns).tolist()
                        coldups = list(set([x for x in allcols if allcols.count(x) > 1]))
                        mergedcols = state.df_cles['Mapping_df_origine'].values.tolist()
                        for mergedcol in mergedcols:
                            coldups.remove(mergedcol)
                        if coldups:
                            state.dataset_clean = state.dataset_clean.drop(columns=coldups)
                        state.dataset_clean = pd.merge(state.dataset_clean, state.dataset_added[filtered], how='left', left_on=state.df_cles['Mapping_df_origine'].values.tolist(), right_on=state.df_cles['Mapping_nouveau_df'].values.tolist())
                        nb_col_after = len(state.dataset_clean.columns)
                        modif = "Insertion de "+str(nb_col_after-nb_col_before)+" colonnes à partir d'un autre dataset"
                        state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                        state.df_cles = state.df_cles.drop(state.df_cles.index)
                        uploaded_newfile = None
            with col9:
                if st.button("Supprimmer les colonnes mappées", key='supmap1'):
                    state.df_cles = state.df_cles.drop(state.df_cles.index)
                    
                            ##### CONCATENER #####
        col10, col11, col12 = st.beta_columns([2,1,2])
        if method_selected == 'Concaténer dataset':
            with col10:
                if st.button("Concaténer les datasets", key='204'):
                    if state.df_cles.empty == False:
                        state.dataset_added[filtered] = tf.map_type(state.dataset_clean, state.dataset_added[filtered], state.df_cles)
                    nature_modif = "Concaténation de lignes à partir d'un autre dataset"
                    nb_l_before = len(state.dataset_clean)
                    for i in state.df_cles.index:
                        col_to_replace = state.df_cles.at[i, 'Mapping_nouveau_df']
                        col_to_copy = state.df_cles.at[i, 'Mapping_df_origine']
                        state.dataset_clean = state.dataset_clean.rename(columns={col_to_replace: col_to_copy})
                    state.dataset_clean = pd.concat([state.dataset_clean, state.dataset_added[filtered]], ignore_index=True)
                    nb_l_after = len(state.dataset_clean)
                    modif = "Insertion de "+str(nb_l_after-nb_l_before)+" lignes à partir d'un autre dataset"
                    state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                    state.df_cles = state.df_cles.drop(state.df_cles.index)
                    uploaded_newfile = None
            with col12:
                if st.button("Supprimmer les colonnes mappées", key='supmap2'):
                    state.df_cles = state.df_cles.drop(state.df_cles.index)             

            
    if state.display_df == True:
        gf.display_data(state.dataset_clean)

    if state.reorder_cols == True:
        state.dataset_clean, state.reorder_var, state.reorder_how = gf.reorder_columns(state.dataset_clean, state.reorder_var, state.reorder_how)
        
    if state.change_value == True:
        state.dataset_clean = gf.change_cell_value(state.dataset_clean)
        
    if state.display_state_df == True:
        state.dataset_clean, state.DF_STATE = gf.display_df_states(state.dataset_clean, state.DF_STATE)
                    


