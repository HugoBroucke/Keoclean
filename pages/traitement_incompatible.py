import streamlit as st
import fonctions.general_fun as gf
import fonctions.diagnosis_fun as df
import pandas as pd
import numpy as np
import fonctions.profiling_fun as pf

def treatment_incomp(state):
    st.header("Traitement des données incompatibles")
    gf.check_data_init(state.dataset_clean)
    with st.beta_expander("Traitement des contraintes métiers"):
        if state.DF_THEN.empty == True:
            st.warning("Aucune règles n'a été renseignée")
        else :
            for rule in state.DF_THEN['ruleID'].unique():
                phrase = df.write_rule(state.DF_IF, state.DF_THEN, rule)
                #"Règle "+str(rule)
                if st.checkbox(phrase, key='checkloop1_'+str(rule)) == True:
                    #phrase = df.write_rule(state.DF_IF, state.DF_THEN, rule)
                    #st.write(phrase)
    
                    DF_RULE = df.find_incompatibilities(dataset=state.dataset_clean, tab1=state.DF_IF, tab2=state.DF_THEN, rule=rule)
                    st.markdown("<h3 style='text-align: center; '>Liste des enregistrements incompatibles</h1>", unsafe_allow_html=True)
                    grid_response = gf.create_grid(DF_RULE, key='agloop1_'+str(rule))
                    data = grid_response['data']
                    selected = grid_response['selected_rows']
                    df_filtered = pd.DataFrame(selected)
    
                    col1, col2, col3 = st.beta_columns([1, 1, 3])
                    with col1:
                        if st.button("Supprimer toutes les lignes concernées", key='butloop1_'+str(rule)):
                            #state.df_precedent_state = state.dataset_clean.copy()
                            nb_lignes = len(data['row_breaking_rule'].values)
                            nature_modif = "Suppression des lignes violant une règle"
                            modif = "Suppression de "+str(nb_lignes)+" violant la règle: "+phrase
                            state.dataset_clean = state.dataset_clean.drop(data['row_breaking_rule'].values)
                            state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                    with col2:
                        if st.button("Supprimer les lignes sélectionnées", key='butloop2_'+str(rule)):
                            #state.df_precedent_state = state.dataset_clean.copy()
                            nb_lignes = len(df_filtered['row_breaking_rule'].values)
                            nature_modif = "Suppression de certaines lignes violant une règle"
                            modif = "Suppression de "+str(nb_lignes)+" violant la règle: "+phrase
                            state.dataset_clean = state.dataset_clean.drop(df_filtered['row_breaking_rule'].values)
                            state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                            DF_RULE.reset_index(inplace=True)
                            df_filtered.reset_index(inplace=True)
                            ds1 = set([tuple(line) for line in DF_RULE.values])
                            ds2 = set([tuple(line) for line in df_filtered.values])
                            res = pd.DataFrame(list(ds1.difference(ds2)))
                            res.columns= DF_RULE.columns
                            DF_RULE = DF_RULE.set_index('index')
                    with col3:
                        if st.checkbox("Souhaitez vous modifier la valeur des cellules concernées ?", key='checkloop2_'+str(rule)):
                            selected_var = st.selectbox('sur quelle variable ?', state.DF_THEN.loc[state.DF_THEN['ruleID']==rule]['variable'].to_list(), key='sbloop1_'+str(rule))
                            data_type = gf.return_type(state.dataset_clean, selected_var)
                            if data_type == 'object':
                                val = st.text_input('Entrez la valeur souhaitée', key='tiloop1_'+str(rule))
                            elif data_type == 'datetime64[ns]':
                                val = st.date_input('Entrez la valeur souhaitée', key='diloop1_'+str(rule))
                            else :
                                val= st.number_input('Entrez la valeur souhaitée', key='diloop1_'+str(rule))
                        
                            if st.button("Modifier la valeur", key='butloop3_'+str(rule)):
                                #state.df_precedent_state = state.dataset_clean.copy()
                                nb_lignes = len(df_filtered['row_breaking_rule'].values)
                                nature_modif = "Imputation de certaines lignes violant une règle"
                                modif = "Imputation de "+str(nb_lignes)+" lignes violant la règle: "+phrase+". Valeur imputée: "+str(val)
                                state.dataset_clean[selected_var] = np.where(state.dataset_clean.index.isin(df_filtered['row_breaking_rule'].to_list()),
                                                                             val, state.dataset_clean[selected_var])
                                state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)

    with st.beta_expander("Traitement des pattern exclus"):
        if (all(element == None for element in state.DF_META['exclude_values'].to_list())) or (state.DF_META.empty == True):
            st.warning("Aucun pattern à exclure n'a été défini")
        else:
            df_excluded = pf.excluded(state.dataset_clean, state.DF_META)
            for pattern in df_excluded.index:
                pattern_name = df_excluded.at[pattern, 'Pattern Exclus']
                var_name = df_excluded.at[pattern, 'Variable']
                if st.checkbox("Pattern "+pattern_name+" pour la variable "+var_name, key='checkloop10_'+str(pattern)) == True:
                    st.markdown("<h3 style='text-align: center; '>Liste des enregistrements avec un pattern exclus</h1>", unsafe_allow_html=True)
                    DF_INCOMP = state.dataset_clean.loc[state.dataset_clean[var_name]==pattern_name].reset_index()
                    DF_INCOMP = DF_INCOMP.rename(columns={'index':'row_breaking_rule'})
                    grid_response = gf.create_grid(DF_INCOMP, key='agloop10_'+str(pattern))
                    data = grid_response['data']
                    selected = grid_response['selected_rows']
                    df_filtered = pd.DataFrame(selected)
                    
                    col1, col2 = st.beta_columns(2)
                    with col1:
                        if st.button("Supprimer les lignes sélectionnées", key='butloop10_'+str(pattern)):
                            nb_lignes = len(df_filtered)
                            nature_modif = "Suppression de certaines lignes avec un pattern exclu"
                            modif = "Suppression de "+str(nb_lignes)+" lignes où la variable "+var_name+" a pour valeur le pattern exclu "+pattern_name
                            state.dataset_clean = state.dataset_clean.drop(df_filtered['row_breaking_rule'].values)
                            state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                    with col2:
                            data_type = gf.return_type(state.dataset_clean, var_name)
                            if st.checkbox("Remplacer la valeur par une valeur manquante ?", key='checkloop100_'+str(pattern)) == True:
                                val = np.nan
                            else:
                                if data_type == 'object':
                                    val = st.text_input('Entrez la valeur de remplacement', key='tiloop10_'+str(pattern))
                                elif data_type == 'datetime64[ns]':
                                    val = st.date_input('Entrez la valeur de remplacement', key='diloop10_'+str(pattern))
                                else :
                                    val= st.number_input('Entrez la valeur de remplacement', key='diloop10_'+str(pattern))
                                
                            if st.button("Remplacer la valeur des lignes sélectionnées", key='butloop20_'+str(pattern)):
                                nb_lignes = len(df_filtered)
                                nature_modif = "Imputation de certaines lignes avec un pattern exclu"
                                modif = "Imputation de "+str(nb_lignes)+" lignes où la variable "+var_name+" a pour valeur le pattern exclu "+pattern_name+". Valeur imputée: "+str(val)
                                state.dataset_clean[var_name] = np.where(state.dataset_clean.index.isin(df_filtered['row_breaking_rule'].to_list()),
                                                                             val, state.dataset_clean[var_name])
                                state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
 
        
        
    if state.display_df == True:
        gf.display_data(state.dataset_clean)

    if state.reorder_cols == True:
        state.dataset_clean, state.reorder_var, state.reorder_how = gf.reorder_columns(state.dataset_clean, state.reorder_var, state.reorder_how)

    if state.change_value == True:
        state.dataset_clean = gf.change_cell_value(state.dataset_clean)
        
    if state.display_state_df == True:
        state.dataset_clean, state.DF_STATE = gf.display_df_states(state.dataset_clean, state.DF_STATE)