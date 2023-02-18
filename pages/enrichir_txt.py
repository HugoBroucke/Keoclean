import streamlit as st
import fonctions.general_fun as gf
import fonctions.treatment_fun as tf
import pandas as pd

def enrich_txt(state):
    st.header("Enrichir à partir des variables textuelles")
    gf.check_data_init(state.dataset_clean)
    cols_txt = []
    for col in state.dataset_clean.columns:
        data_type = gf.return_type(state.dataset_clean, col)
        if data_type == 'object':
            cols_txt.append(col)
            
    with st.beta_expander("Extraire les entités reconnues (NER)"):
        selected_col_txt = st.selectbox("Choisir la variable à utiliser", cols_txt, key='sb17')
        b_ner1, b_ner2 = st.beta_columns(2)
        with b_ner1:
            st.markdown("<h3 style='text-align: center; '>Extraction d'entités</h3>", unsafe_allow_html=True)
            if st.button('Extraire les entités reconnues', key='but17'):
                with st.spinner("Extraction en cours ..."):
                    state.df_dummy = tf.get_NER(state.dataset_clean, selected_col_txt)
                    state.b_ner1_onclick = True
                    state.b_ner2_onclick = False
                    state.df_dummy2 = pd.DataFrame()
        
        with b_ner2:
            st.markdown("<h3 style='text-align: center; '>OneHotEncoding d'entités</h3>", unsafe_allow_html=True)
            entities = ['EXAM', 'SIT_DIS']
            selected_entity = st.selectbox("Choisir l'entité avec laquelle travailler", entities, key='sb18')
            if st.button("Extraire les variable dummies de l'entité "+selected_entity, key='but18'):
                with st.spinner("Création des dummies en cours ..."):
                    state.df_dummy2 = tf.get_dummies_from_NER(state.dataset_clean, selected_col_txt, selected_entity)
                    state.b_ner2_onclick = True
                    state.b_ner1_onclick = False
                    state.df_dummy = pd.DataFrame() 
                
        if state.b_ner1_onclick == True: 
            #st.write("Fusionner le dataset ?")                                                                             

            st.markdown("<h4 style='text-align: left; '>Fusionner les entités reconnues au dataset ?</h4>", unsafe_allow_html=True)
            st.dataframe(state.df_dummy.head(50))
            
            col1, col2 = st.beta_columns(2)
            with col1:
                selected_ner_to_del = st.selectbox("Choisir une variable dummy à supprimmer", state.df_dummy.columns[1:], key='sb19')
                if st.button("Supprimmer la variable sélectionné", key='but19'):
                    state.df_dummy = state.df_dummy.drop(columns=[selected_ner_to_del])
            with col2:
                if st.button("Ajouter les variables au dataset", key='but20'):
                    #state.df_precedent_state = state.dataset_clean.copy()
                    nature_modif = "Extraction d'entités reconnues d'une variable textuelle"
                    modif = "Extraction des entités: "
                    for v in state.df_dummy.columns[1:]:
                        add = v+' '
                        modif = modif + add    
                    modif = modif+'pour la variable '+selected_col_txt
                    state.dataset_clean = pd.merge(state.dataset_clean, state.df_dummy.loc[:,state.df_dummy.columns[1:]], left_index=True, right_index=True)
                    state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                    state.df_dummy = pd.DataFrame()
                    state.b_ner1_onclick = False
        
        if state.b_ner2_onclick == True: 
            #st.write("Fusionner le dataset ?")
            st.markdown("<h4 style='text-align: left; '>Fusionner les entités reconnues au dataset ?</h4>", unsafe_allow_html=True)                                                                            
            st.dataframe(state.df_dummy2.head(50))
            
            col3, col4 = st.beta_columns(2)
            with col3:
                selected_dummy = st.selectbox("Choisir une variable dummy à supprimmer", state.df_dummy2.columns[1:], key='sb20')
                if st.button("Supprimmer la variable dummy sélectionné", key='but21'):
                    state.df_dummy2 = state.df_dummy2.drop(columns=[selected_dummy])
            with col4:
                if st.button("Ajouter les variables au dataset", key='but22'):
                    #state.df_precedent_state = state.dataset_clean.copy()
                    nb_dummies = len(state.df_dummy2.columns[1:])
                    nature_modif = "Extraction des dummies d'une entité reconnue d'une variable textuelle"
                    modif = "Extraction de "+nb_dummies+" d'entités "+selected_entity+" pour la variable "+selected_col_txt
                    state.dataset_clean = pd.merge(state.dataset_clean, state.df_dummy2.loc[:,state.df_dummy2.columns[1:]], left_index=True, right_index=True)
                    state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                    state.df_dummy2 = pd.DataFrame()
                    state.b_ner2_onclick = False
        
        ##### EXTRACTION D UN PATTERN #####
        
    with st.beta_expander("Créer des colonnes à partir d'un pattern"):
        selected_col_txt = st.selectbox("Choisir la variable à utiliser", cols_txt, key='sb21')
        pattern_split = st.text_input("Entrer le pattern sur lequel diviser la variable", key='ti7')
        if st.button("Diviser la variable", key='but23'):
            nature_modif = "Création de colonnes à partir d'un pattern d'une variable textuelle"
            nb_col_before = len(state.dataset_clean.columns)
            #state.df_precedent_state = state.dataset_clean.copy()
            state.dataset_clean = tf.split_column(state.dataset_clean, selected_col_txt, pattern_split)
            nb_col_after = len(state.dataset_clean.columns)
            modif = "Création de "+str(nb_col_before-nb_col_after)+" à partir du pattern "+pattern_split+" pour la variable "+selected_col_txt
            state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
            st.success('Modification effectuée')
            
            ##### REGROUPER PLUSIEURS COLONNES #####
        
    with st.beta_expander("Regrouper des colonnes"):
        selected_col_txt = st.selectbox("Sélectionner une variable à regrouper", cols_txt, key='200')
        if st.button("Ajouter la colonne"):
            state.group_cols.append(selected_col_txt)
        st.info("Liste des colonnes à regrouper: "+str(state.group_cols))   
        pattern_divide = st.text_input("Entrer le pattern pour séparer les modalités", key='201')
        nom_var = st.text_input("Entrer le nom de la colonne à créer", key='202')
        if st.button("Regrouper les colonnes", key='203'):
            if not state.group_cols:
                st.error("Veuillez entrer les colonnes à regrouper")
            elif not nom_var:
                st.error("Veuillez entrer le nom de la variable à créer")
            else:
                nature_modif = "Création de colonnes à partir de plusieurs variables textuelles" 
                df2merge = pd.DataFrame(state.dataset_clean[state.group_cols].dropna().agg(pattern_divide.join, axis=1), columns=[nom_var])
                if nom_var in state.dataset_clean.columns:
                    state.dataset_clean = state.dataset_clean.drop(columns=[nom_var])
                state.dataset_clean = pd.merge(state.dataset_clean, df2merge, left_index=True, right_index=True, how='left')
                modif = "Création de la colonne "+nom_var+" regroupant les variables: "
                for v in state.group_cols:
                        add_var = v+' '
                        modif = modif + add_var
                state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                state.group_cols = []
                st.success('Colonne créée')
            
        ##### EXTRACTION A PARTIR DE CODES POSTAUX #####
        
    with st.beta_expander("Créer des colonnes à partir de codes postaux standardisés"):
        selected_col_txt_cp = st.selectbox("Choisir la variable correspondant au code postal standardisé", cols_txt, key='sb22')
        method_cp = ['Commune', 'Code Département', 'Département', 'Code Région', 'Région']
        method = st.selectbox("Choisir l'extraction à réaliser", method_cp, key='sb23')
        if st.button('Ajouter la colonne', key='but24'):
            #state.df_precedent_state = state.dataset_clean.copy()
            nature_modif = "Création de colonnes à partir d'un code postal"
            modif = "Création de la colonne "+method+" à partir des codes postaux de la variable "+selected_col_txt_cp
            state.dataset_clean = tf.extract_from_cp_insee(state.dataset_clean, selected_col_txt_cp, method)
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