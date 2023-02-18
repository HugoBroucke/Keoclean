import streamlit as st
import fonctions.general_fun as gf
import fonctions.treatment_fun as tf
import pandas as pd

def standard_txt(state):
    st.header("Standardisation des variables textuelles")
    gf.check_data_init(state.dataset_clean)
    cols_text = []
    for col in state.dataset_clean.columns:
        data_type = gf.return_type(state.dataset_clean, col)
        if data_type == 'object':
            cols_text.append(col)
            
    with st.beta_expander("Clustering de modalités"):
        selected_col_txt = st.selectbox("Choisir une variable à modifier", cols_text, key='sb29')
        if st.checkbox("Définir manuellement un nombre de cluster ? (Recommandé dans le cadre de nombreuses modalités)", key='check14') == True:
            n_clus = st.number_input("Choisir le nombre de clusters", value=int(2), min_value=2, key='ni12')
        else:
            n_clus = None
        n_gram = st.number_input("N-Gram", value=int(1), min_value=1, key='ni13')
        
        st.write("[Lien vers la documentation](https://fr.wikipedia.org/wiki/N-gramme)")
        
        if st.button('Identifier les clusters', key='but30'):
            state.df_transitoire, state.df_tomerge = tf.get_clusters(state.dataset_clean, selected_col_txt, ngram=(n_gram,n_gram), n_clus=n_clus)
            state.show_grid = True
            
        if state.show_grid == True:
            st.markdown("<h3 style='text-align: center; '>Liste des clusters</h1>", unsafe_allow_html=True)
            grid_response = gf.create_grid(state.df_transitoire)       
            df = grid_response['data'] #Print response, ne pas commenter     
            selected = grid_response['selected_rows']
            df_filtered = pd.DataFrame(selected)
            
            b1, b2 = st.beta_columns(2)
            with b1:
                if st.button('Nettoyer les données sélectionnées avec la valeur la plus représentative de chaque cluster', key='but31'):
                    nb_var = df_filtered['count'].sum()
                    nature_modif = "Clustering de modalités par la valeur la plus représentative de chaque groupepour une variable textuelle"
                    modif = "Regroupement de "+str(nb_var)+" observations pour la variable " +selected_col_txt
                    #state.df_precedent_state = state.dataset_clean.copy()
                    state.dataset_clean = tf.replace_with(state.dataset_clean, selected_col_txt, df_filtered, method='most_representative', value=None)
                    state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                    state.df_transitoire = pd.DataFrame()
                    state.show_grid = False
            
            with b2:
                text_replacement = st.text_input('Entrer la valeur de remplacement', key='ni14')
                if st.button('Nettoyer les données sélectionnées avec la valeur entrée', key='but32'):
                    #state.df_precedent_state = state.dataset_clean.copy()
                    nb_var = df_filtered['count'].sum()
                    nature_modif = "Clustering de modalités par une valeur donnée pour une variable textuelle"
                    modif = "Imputation de "+str(nb_var)+" observations pour la variable " +selected_col_txt+" par la valeur: "+text_replacement
                    state.dataset_clean = tf.replace_with(state.dataset_clean, selected_col_txt, df_filtered, method='custom', value=text_replacement)
                    state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                    state.df_transitoire = state.df_transitoire.loc[~state.df_transitoire[selected_col_txt].isin(df_filtered[selected_col_txt].unique())]
                    
                    ##### HARMONISATION FORMAT #####
    with st.beta_expander("Harmonisation du format"):  
        selected_col_txt = st.selectbox("Choisir une variable à modifier", cols_text, key='sb30')
        fn = ['Uppercase', 'Lowercase', 'Proppercase', 'Enlever les accents', 'Enlever la ponctuation', 'Enlever les espaces de début et de fin', 'Remplacer un pattern', 'Standardiser le texte']
        selected_fn = st.selectbox("Choisir une fonction", fn, key='sb31')
        
        if selected_fn == 'Remplacer un pattern':
            pattern_out = st.text_input("Choisir le pattern à remplacer", key='ti9')
            pattern_in = st.text_input("Remplacer le pattern par", key='ti10')
        else:
            pattern_out = None
            pattern_in = None
            
        if selected_fn == 'Standardiser le texte':
            st.info("Retrait des stopwords, retrait des accents, passage en police minuscule, retrait de la ponctuation, lemmatisation")
        
        if st.button('Appliquer', key='but33'):
            #state.df_precedent_state = state.dataset_clean.copy()
            nature_modif = "Harmonisation du format d'une variable textuelle"
            modif = "Harmonisation de la variable "+selected_col_txt+" par la méthode "+selected_fn
            if selected_fn == 'Remplacer un pattern':
                modif = modif+" (remplacement du pattern "+pattern_out+ " par le pattern "+pattern_in+")"
            state.dataset_clean = tf.harmonisation_str(state.dataset_clean, selected_col_txt, selected_fn, pattern_out=pattern_out, pattern_in=pattern_in)
            state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
            st.success('Modification effectuée')

        ##### HARMONISATION CODES POSTAUX #####
    
    with st.beta_expander("Standardisation des codes postaux"):
        selected_col_txt = st.selectbox("Choisir la variable correspondant au code postal", cols_text, key='sb32')
        if st.button("Standardiser les codes postaux", key='but34'):
            try:
                #precedent_state_trying = state.dataset_clean.copy()
                nature_modif = "Harmonisation du code postal d'une variable textuelle"
                modif = "Harmonisation du code postal pour la variable "+selected_col_txt
                state.dataset_clean = tf.get_cp_insee(state.dataset_clean, selected_col_txt)
                state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                #state.df_precedent_state = precedent_state_trying.copy() #On sauvegarde l'état précédent seulement si le try est reussi
                st.success('Modification effectuée')
            except:
                st.error("Erreur : la variable sélectionnée ne correspond pas à un code postal")
                
    if state.display_df == True:
        gf.display_data(state.dataset_clean)

    if state.reorder_cols == True:
        state.dataset_clean, state.reorder_var, state.reorder_how = gf.reorder_columns(state.dataset_clean, state.reorder_var, state.reorder_how)

    if state.change_value == True:
        state.dataset_clean = gf.change_cell_value(state.dataset_clean)
        
    if state.display_state_df == True:
        state.dataset_clean, state.DF_STATE = gf.display_df_states(state.dataset_clean, state.DF_STATE)
        
