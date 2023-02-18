import streamlit as st
import fonctions.general_fun as gf
import fonctions.diagnosis_fun as df
import fonctions.treatment_fun as tf
import phik #Ne pas commenter
from phik import resources, report #Ne pas commenter
import pandas as pd

def treatment_doublons(state):
    st.header("Traitement des données dupliquées")
    gf.check_data_init(state.dataset_clean)
    with st.beta_expander("Traitement des lignes dupliquées"):
        if st.button("Démarrer l'analyse", key='but35'):
            state.duplicate_lignes = state.dataset_clean[state.dataset_clean.duplicated(keep=False)].reset_index()
            state.b1onclick = True
            
        if state.b1onclick == True:
            st.write('Nombre de lignes duppliquées : '+str(len(state.duplicate_lignes))+' lignes dupliquées')
            st.markdown("<h3 style='text-align: center; '>Liste des enregistrements duppliqués</h1>", unsafe_allow_html=True)
            grid_response = gf.create_grid(state.duplicate_lignes, key='ag1')
            data = grid_response['data']
            selected = grid_response['selected_rows']
            df_filtered = pd.DataFrame(selected)
            
            dup1, dup2, dup3 = st.beta_columns([5,1,5])
            with dup1:
                if st.button("Supprimer les lignes dupliquées", key='but36'):
                    #state.df_precedent_state = state.dataset_clean.copy()
                    nb_dup = len(data)
                    nature_modif = "Suppression de toutes les lignes dupliquées"
                    modif = "Suppression de "+str(nb_dup)+" lignes dupliquées"
                    state.dataset_clean = state.dataset_clean.drop_duplicates()
                    state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif ,modif, state.dataset_clean)
                    state.b1onclick = False
            with dup3:
                if st.button("Supprimer les lignes sélectionnées", key='but61'):
                    #state.df_precedent_state = state.dataset_clean.copy()
                    nb_dup = len(state.dataset_clean.loc[df_filtered['index'].values].index)
                    nature_modif = "Suppression de certaines lignes dupliquées"
                    modif = "Suppression de "+str(nb_dup)+" lignes dupliquées"
                    state.dataset_clean = state.dataset_clean.drop(state.dataset_clean.loc[df_filtered['index'].values].index)
                    state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                    state.duplicate_lignes = state.duplicate_lignes.loc[~state.duplicate_lignes['index'].isin(df_filtered['index'].values)]

                ##### TRAITEMENT COLONNES DUPLIQUEES #####
    with st.beta_expander("Traitement des colonnes dupliquées"):
        butcoldup = st.button("Démarrer l'analyse", key='but37')
        st.write("[Lien vers la librairie PhiK utilisée dans le calcul les corrélations](https://phik.readthedocs.io/en/latest/index.html)")
        if butcoldup:
            with st.spinner("Analyse en cours, veuillez patienter..."):
                
                if len(state.dataset_clean) > 1000:
                    df_samp = state.dataset_clean.sample(1000)
                    state.sig_matrix = df_samp.significance_matrix()
                    st.warning("Pour des raisons d'optimisation, les corrélations sont effectuées sur un échantillon du dataset (n=1000)")
                else:
                    state.sig_matrix = state.dataset_clean.significance_matrix()
                    
                state.b2onclick = True
        
        if state.b2onclick == True:
            confiance = st.selectbox("Sélectionner la confiance avec laquelle on considère que deux variables sont corrélées", ['90%', '95%', '99%'], key='sb33')
            df_res = df.recuperer_var_corr_selon_confiance(state.sig_matrix, confiance=confiance)
            a = df_res['VAR X'].to_list()
            b = df_res['VAR Y'].to_list()
            a.extend(b)
            cor_vars = list(set(a))

            st.markdown("<h3 style='text-align: center; '>Liste des colonnes avec une forte corrélation</h1>", unsafe_allow_html=True)
            st.dataframe(df_res)

            #var2del = st.selectbox("Sélectionner la variable à supprimer", cor_vars, key='sb34')
            var2keep = st.multiselect("Sélectionner les variables à conserver", options=list(cor_vars), default=list(cor_vars))
            
            if st.button("Valider", key='but38'):
                #state.df_precedent_state = state.dataset_clean.copy()
                var2del = [x for x in cor_vars if x not in var2keep]
                
                nature_modif = "Suppression de colonnes corrélées"
                modif = "Suppression des colonnes: "
                for v in var2del:
                    add_var = v+' '
                    modif = modif + add_var
                
                state.dataset_clean = state.dataset_clean.drop(columns = var2del)
                state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                #state.sig_matrix = state.sig_matrix.drop(columns=var2del, index=var2del)
                df_res = df_res.loc[~(df_res['VAR X'].isin(var2del))]
                df_res = df_res.loc[~(df_res['VAR Y'].isin(var2del))]
                # state.dataset_clean = state.dataset_clean.drop(columns=[var2del])
                # state.sig_matrix = state.sig_matrix.drop(columns=[var2del], index=[var2del])
                
                ##### TRAITEMENT DES ENREGISTREMENTS SIMILAIRES #####
    with st.beta_expander("Traitement des enregistrements similaires"):
        st.write("[Lien vers la librairie RecordLinkage utilisée pour la détection d'enregistrements similaire](https://phik.readthedocs.io/en/latest/index.html)")
        col1, col2 = st.beta_columns(2)
        with col1:
            selected_variable = st.selectbox("Choisir une variable à mettre dans la clé", state.dataset_clean.columns, key='sb35')
            if gf.return_type(state.dataset_clean, selected_variable) == 'object':
                method = ['levenshtein', 'jaro','jarowinkler', 'damerau_levenshtein', 'qgram','cosine', 'smith_waterman', 'lcs']
                selected_method = st.selectbox("Choisir un calcul de distance", method, key='sb36')
                threshold = st.number_input("Entrer un seuil minimal de correspondance par variable en pourcentage", min_value=0, max_value=100, value=int(30), key='ni15')
            else:
                selected_method = None
                threshold = None
            
            if st.button("Ajouter la variable à la clé", key='but39'):
                state.compare_cl, state.nb_keys_rl = tf.add_variable_to_key(state.compare_cl, state.nb_keys_rl,
                                                                            state.dataset_clean, selected_variable,
                                                                            method=selected_method, threshold=threshold)
                if selected_variable in state.df_rl_keys['variable'].tolist():
                    state.df_rl_keys = state.df_rl_keys.drop(state.df_rl_keys[state.df_rl_keys['variable'] == selected_variable].index)
                    state.df_rl_keys = state.df_rl_keys.append({'variable':selected_variable, 'distance':selected_method, 'seuil':threshold}, ignore_index=True)
                else:
                    state.df_rl_keys = state.df_rl_keys.append({'variable':selected_variable, 'distance':selected_method, 'seuil':threshold}, ignore_index=True)
                #Ajouter un truc pour supprimer une variable qui y est déjà
            
            st.markdown("<h3 style='text-align: center; '>Liste des clés</h1>", unsafe_allow_html=True)
            st.dataframe(state.df_rl_keys)
            liste_cle_2del = state.df_rl_keys['variable'].to_list()
            cle2del = st.selectbox("Choisir la clé à supprimer", liste_cle_2del, key='sb45')
            if st.button("Supprimer les clés sélectionnées", key='but64'):
                #state.df_rl_keys = state.df_rl_keys.drop(state.df_rl_keys.loc[state.df_rl_keys['variable'].isin(df_filtered['variable'].values)].index)
                state.df_rl_keys = state.df_rl_keys.drop(state.df_rl_keys.loc[state.df_rl_keys['variable'] == cle2del].index)
                        
        with col2:
            selected_variable_block = st.selectbox("Choisir une variable sur laquelle bloquer", state.dataset_clean.columns, key='sb37')
            if st.button("Bloquer sur la variable", key='but40'):
                state.block.append(selected_variable_block)
                state.block = list(set(state.block))
            
            st.info("Bloqué sur : "+str(state.block))
            if st.button("Vider la liste ?", key='but63'):
                state.block.clear()
        
        classif = ['ECM', 'naive']
        for var in state.df_rl_keys['variable'].to_list():
            if gf.return_type(state.dataset_clean, var) != 'object' :
                classif = ['naive']
                
        selected_classif = st.selectbox("Choisir la méthode de classification (attention, seule la méthode naive est disponible si des variables non textuelles font parties des clés)", classif, key='sb38')
        
        if selected_classif == 'naive':
            min_score = st.number_input("Entrer un seuil minimal de correspondance global en pourcentage", min_value=0, max_value=100, value=int(50), key='ni16')
        else :
            min_score = 0
        
        if st.button("Exécuter le RecordLinkage", key='but41'):
            with st.spinner("Génération des groupes d'enregistrements similaires, veuillez patienter ..."):
                state.df_rl_groups = tf.compute_record_linkage(state.compare_cl, state.nb_keys_rl, state.dataset_clean, block=state.block,
                                                      min_score=min_score, classifier_method=selected_classif)
                state.fill_color_rl = []
                for l in state.df_rl_groups.index:
                    state.fill_color_rl.append(['lightgrey' if (val % 2) == 0 else 'white' for val in [int(state.df_rl_groups['groupe'].str.split('grp_')[l][1])]])
                state.rl_executed = True
                
        if state.rl_executed == True:
            if state.df_rl_groups.empty :
                st.success("Aucun enregistrement dupliqués avec ces paramètres")
            else :
                #st.write("Liste des groupes identifiés")  
                st.markdown("<h3 style='text-align: center; '>Liste des groupes identifiés</h1>", unsafe_allow_html=True)
                st.dataframe(state.df_rl_groups)
                #gf.display_df_streamlit(state.df_rl_groups)
        
            if st.button("Supprimer les lignes identifiées comme non master", key='but42'):
                state.df_precedent_state = state.dataset_clean.copy()
                nature_modif = "Suppression de enregistrements similaires"
                nb_ligne_before_modif = len(state.dataset_clean)
                state.dataset_clean = tf.eliminate_records_identified_as_duplicates(state.dataset_clean, state.df_rl_groups)
                nb_ligne_after_modif = len(state.dataset_clean)
                modif = "Suppression de "+str(nb_ligne_before_modif-nb_ligne_after_modif)+" enregistrements similaires"
                state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                state.df_rl_groups = state.df_rl_groups.drop(state.df_rl_groups.index)
                state.df_rl_keys = state.df_rl_keys.drop(state.df_rl_keys.index)
                state.rl_executed = False
                
    if state.display_df == True:
        gf.display_data(state.dataset_clean)
   
    if state.reorder_cols == True:
        state.dataset_clean, state.reorder_var, state.reorder_how = gf.reorder_columns(state.dataset_clean, state.reorder_var, state.reorder_how)

    if state.change_value == True:
        state.dataset_clean = gf.change_cell_value(state.dataset_clean)
        
    if state.display_state_df == True:
        state.dataset_clean, state.DF_STATE = gf.display_df_states(state.dataset_clean, state.DF_STATE)
        