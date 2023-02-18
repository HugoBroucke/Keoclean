import streamlit as st
import fonctions.general_fun as gf
import fonctions.diagnosis_fun as df
import fonctions.treatment_fun as tf
import pandas as pd
import numpy as np

def treatment_hl(state):
    st.header("Traitement des valeurs hors limite")
    gf.check_data_init(state.dataset_clean)
    imp_methods = ['mean', 'median', 'mode']
    with st.beta_expander("Gestion des valeurs hors limite"): 
        if st.button("Démarrer l'analyse", key='but43'):
            state.df_out_of_range_val = df.find_out_of_range_values(state.dataset_clean, state.DF_META)
            state.b5onclick = True
            
        if state.b5onclick == True:
            state.df_out_of_range_val = state.df_out_of_range_val.applymap(gf.set_default)
            st.markdown("<h3 style='text-align: center; '>Liste des valeurs hors limites</h1>", unsafe_allow_html=True)
            grid_response = gf.create_grid(state.df_out_of_range_val)
            data = grid_response['data']
            selected = grid_response['selected_rows']
            df_filtered = pd.DataFrame(selected)
            
            apply_all = st.checkbox("Appliquer les modifications sur toutes les valeurs hors limite identifiées?", key='check15')
            if apply_all == True:
                apllyall1, apllyall2 = st.beta_columns(2)
                with apllyall1: 
                    if st.button("Supprimer les valeurs hors limites", key='but44'):
                        #state.df_precedent_state = state.dataset_clean.copy()
                        nb_oor = len(state.df_out_of_range_val)
                        nature_modif = "Suppression des lignes avec des valeurs hors limites"
                        state.dataset_clean = state.dataset_clean.drop(state.df_out_of_range_val['index'].values)
                        modif = "Suppression de "+str(nb_oor)+" lignes avec des valeurs hors limites"
                        state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                        state.df_out_of_range_val = pd.DataFrame()
                        state.b5onclick = False
                with apllyall2:
                    if st.checkbox("Imputer par une valeur spécifique ?", key='700'):
                        replaceall = st.number_input("Choisir la valeur par laquelle remplacer", key='ni17')
                        if st.button("Remplacer les valeurs hors limite par la valeur renseignée", key='but45'):
                            #state.df_precedent_state = state.dataset_clean.copy()
                            nb_oor = len(state.df_out_of_range_val)
                            nature_modif = "Imputation des lignes avec des valeurs hors limites"
                            for i in state.df_out_of_range_val.index:
                                state.dataset_clean.at[state.df_out_of_range_val.at[i, 'index'], state.df_out_of_range_val.at[i, 'variable']] = replaceall
                            state.df_out_of_range_val = pd.DataFrame()
                            state.b5onclick = False
                            modif = "Imputation de "+str(nb_oor)+" lignes avec des valeurs hors limites par la valeur "+str(replaceall)
                            state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                    else:
                        selected_imp = st.selectbox("Choisir la méthode d'imputation", imp_methods, key='701')
                        if st.button("Remplacer les valeurs hors limite par la méthode sélectionnée", key='702'):
                            nb_oor = len(state.df_out_of_range_val)
                            nature_modif = "Imputation des lignes avec des valeurs hors limites"
                            modif = "Imputation de "+str(nb_oor)+" lignes avec des valeurs hors limites par la méthode "+str(selected_imp)
                            for c in state.df_out_of_range_val['variable'].unique():
                                state.dataset_clean[c] = np.where(state.dataset_clean.index.isin(state.df_out_of_range_val['index'].values),
                                                                         tf.impute_outliers(state.dataset_clean, c, method = selected_imp),
                                                                         state.dataset_clean[c])
                            state.df_out_of_range_val = pd.DataFrame()
                            state.b5onclick = False
                            state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
            
            else:
                applysel1, applysel2 = st.beta_columns(2)
                with applysel1: 
                    if st.button("Supprimer les valeurs hors limites", key='but46'):
                        nb_oor = len(df_filtered['index'].values)
                        nature_modif = "Suppression de certaines lignes avec des valeurs hors limites"
                        #state.df_precedent_state = state.dataset_clean.copy()
                        state.dataset_clean = state.dataset_clean.drop(df_filtered['index'].values)
                        modif = "Suppression de "+str(nb_oor)+" lignes avec des valeurs hors limites"
                        state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                        state.df_out_of_range_val = state.df_out_of_range_val.drop(state.df_out_of_range_val.loc[state.df_out_of_range_val['index'].isin(df_filtered['index'].values) == True].index)
                with applysel2:
                    if st.checkbox("Imputer par une valeur spécifique ?", key='800'):
                        replaceall = st.number_input("Choisir la valeur par laquelle remplacer", key='ni18')
                        if st.button("Remplacer les valeurs hors limite par la valeur renseignée", key='but47'):
                            nb_oor = len(df_filtered['index'].values)
                            nature_modif = "Imputation de certaines lignes avec des valeurs hors limites"
                            for i in df_filtered.index:
                                state.dataset_clean.at[df_filtered.at[i, 'index'], df_filtered.at[i, 'variable']] = replaceall
                                state.df_out_of_range_val = state.df_out_of_range_val.drop(state.df_out_of_range_val.loc[state.df_out_of_range_val['index'].isin(df_filtered['index'].values) == True].index)
                            modif = "Imputation de "+str(nb_oor)+" lignes avec des valeurs hors limites par la valeur "+str(replaceall)
                            state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                    else:
                        selected_imp = st.selectbox("Choisir la méthode d'imputation", imp_methods, key='801')
                        if st.button("Remplacer les valeurs hors limite par la méthode sélectionnée", key='802'):
                            nb_oor = len(state.df_out_of_range_val)
                            nature_modif = "Imputation de certaines lignes avec des valeurs hors limites"
                            for c in df_filtered['variable'].unique():
                                state.dataset_clean[c] = np.where(state.dataset_clean.index.isin(df_filtered['index'].values),
                                                                         tf.impute_outliers(state.dataset_clean, c, method = selected_imp),
                                                                         state.dataset_clean[c])
                            modif = "Imputation de "+str(nb_oor)+" lignes avec des valeurs hors limites par la méthode "+str(selected_imp)
                            #state.b5onclick = False
                            state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                    ##### TRAITEMENT DES OUTLIERS #####
    with st.beta_expander("Gestion des outliers"): 
        col_num = []
        for col in state.dataset_clean.columns:
            data_type = gf.return_type(state.dataset_clean, col)
            #if (data_type == 'Int64') | (data_type =='float64'):
            if data_type =='float64':
                col_num.append(col)
        selected_col = st.selectbox("Choix de la variable à analyser", col_num, key='sb39')
        methods = ['z_score', 'isolation_forest', 'elliptic_envelope', 'local_outlier_factor']
        selected_method = st.selectbox("Choix de la méthode de détection des outliers", methods, key='sb40')
        
        if selected_method != 'z_score':
            contamination = st.number_input("Entrer le niveau de contamination en pourcentage", min_value=0, max_value=100, value=int(5), key='ni19')
            if selected_method == 'isolation_forest':
                st.write("[Lien vers la librairie utilisée](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)")
            elif selected_method == 'elliptic_envelope':
                st.write("[Lien vers la librairie utilisée](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html)")
            elif selected_method == 'local_outlier_factor':
                st.write("[Lien vers la librairie utilisée](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)")        
        
        if st.button("Démarrer l'analyse", key='but48'):
            if selected_method != 'z_score':
                state.df_outliers = df.find_outliers(state.dataset_clean, col = selected_col, method=selected_method, contamination=contamination/100)
            else :
                state.df_outliers = df.find_outliers(state.dataset_clean, col=selected_col, method=selected_method)
            state.b6onclick = True
            
        if state.b6onclick == True:
            state.df_outliers = state.df_outliers.applymap(gf.set_default)
            st.markdown("<h3 style='text-align: center; '>Liste des outliers</h1>", unsafe_allow_html=True)
            grid_response = gf.create_grid(state.df_outliers)
            data = grid_response['data'] #Ne pas commenter
            selected = grid_response['selected_rows']
            df_filtered = pd.DataFrame(selected)
            
            apply_all = st.checkbox("Appliquer les modifications sur tous les outliers identifiés?", key='check16')
            if apply_all == True:
                apllyall1, apllyall2 = st.beta_columns(2)
                with apllyall1: 
                    if st.button("Supprimer les outliers", key='but49'):
                        #state.df_precedent_state = state.dataset_clean.copy()
                        nb_oor = len(state.df_outliers['index'].values)
                        nature_modif = "Suppression des lignes avec des outliers"
                        state.dataset_clean = state.dataset_clean.drop(state.df_outliers['index'].values)
                        modif = "Suppression de "+str(nb_oor)+" lignes avec des outliers (méthode de détection des outliers: "+selected_method+")"
                        state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                        state.df_outliers = pd.DataFrame()
                        state.b6onclick = False
                        
                with apllyall2:
                    if st.checkbox("Imputer par une valeur spécifique ?", key='500'):
                        replaceall = st.number_input("Choisir la valeur par laquelle remplacer", key='ni20')
                        if st.button("Remplacer les ouliers par la valeur renseignée", key='but50'):
                            nb_oor = len(state.df_outliers['index'].values)
                            nature_modif = "Imputation des lignes avec des outliers"
                            modif = "Imputation de "+str(nb_oor)+" lignes avec des outliers par la valeur "+str(replaceall)+" (méthode de détection des outliers: "+selected_method+")"
                            for i in state.df_outliers.index:
                                state.dataset_clean.at[state.df_outliers.at[i, 'index'], state.df_outliers.at[i, 'variable']] = replaceall
                            state.df_outliers = pd.DataFrame()
                            state.b6onclick = False
                            state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                    else:
                        selected_imp = st.selectbox("Choisir la méthode d'imputation", imp_methods, key='501')
                        if st.button("Remplacer les ouliers par la méthode sélectionnée", key='502'):
                            nb_oor = len(state.df_outliers['index'].values)
                            nature_modif = "Imputation des lignes avec des outliers"
                            modif = "Imputation de "+str(nb_oor)+" lignes avec des outliers par la méthode "+str(selected_imp)+" (méthode de détection des outliers: "+selected_method+")"
                            state.dataset_clean[selected_col] = np.where(state.dataset_clean.index.isin(state.df_outliers['index'].values),
                                                                         tf.impute_outliers(state.dataset_clean, selected_col, method = selected_imp),
                                                                         state.dataset_clean[selected_col])
                            state.df_outliers = pd.DataFrame()
                            state.b6onclick = False 
                            state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                        
            else:
                applysel1, applysel2 = st.beta_columns(2)
                with applysel1: 
                    if st.button("Supprimer les outliers", key='but51'):
                        #state.df_precedent_state = state.dataset_clean.copy()
                        nb_oor = len(df_filtered['index'].values)
                        nature_modif = "Suppression de certaines lignes avec des outliers"
                        state.dataset_clean = state.dataset_clean.drop(df_filtered['index'].values)
                        modif = "Suppression de "+str(nb_oor)+" lignes avec des outliers (méthode de détection des outliers: "+selected_method+")"
                        state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                        state.df_outliers = state.df_outliers.drop(state.df_outliers.loc[state.df_outliers['index'].isin(df_filtered['index'].values) == True].index)
                
                with applysel2:
                    if st.checkbox("Imputer par une valeur spécifique ?", key='600'):
                        replaceall = st.number_input("Choisir la valeur par laquelle remplacer", key='ni21')
                        if st.button("Remplacer les outliers par la valeur renseignée", key='but52'):
                            nb_oor = len(df_filtered['index'].values)
                            nature_modif = "Imputation de certaines lignes avec des outliers"
                            modif = "Imputation de "+str(nb_oor)+" lignes avec des outliers par la valeur "+str(replaceall)+" (méthode de détection des outliers: "+selected_method+")"
                            for i in df_filtered.index:
                                state.dataset_clean.at[df_filtered.at[i, 'index'], df_filtered.at[i, 'variable']] = replaceall
                            state.df_outliers = state.df_outliers.drop(state.df_outliers.loc[state.df_outliers['index'].isin(df_filtered['index'].values) == True].index)
                            state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                    else:
                        selected_imp = st.selectbox("Choisir la méthode d'inputation", imp_methods, key='601')
                        if st.button("Remplacer les ouliers par la méthode sélectionnée", key='602'):
                            nb_oor = len(state.df_outliers['index'].values)
                            nature_modif = "Imputation des lignes avec des outliers"
                            modif = "Imputation de "+str(nb_oor)+" lignes avec des outliers par la méthode "+str(selected_imp)+" (méthode de détection des outliers: "+selected_method+")"
                            state.dataset_clean[selected_col] = np.where(state.dataset_clean.index.isin(state.df_outliers['index'].values),
                                                                         tf.impute_outliers(state.dataset_clean, selected_col, method=selected_imp),
                                                                         state.dataset_clean[selected_col])
                            state.df_outliers = state.df_outliers.drop(state.df_outliers.loc[state.df_outliers['index'].isin(df_filtered['index'].values) == True].index)
                            state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
    if state.display_df == True:
        gf.display_data(state.dataset_clean)

    if state.reorder_cols == True:
        state.dataset_clean, state.reorder_var, state.reorder_how = gf.reorder_columns(state.dataset_clean, state.reorder_var, state.reorder_how)

    if state.change_value == True:
        state.dataset_clean = gf.change_cell_value(state.dataset_clean)
        
    if state.display_state_df == True:
        state.dataset_clean, state.DF_STATE = gf.display_df_states(state.dataset_clean, state.DF_STATE)