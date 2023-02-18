import streamlit as st
import fonctions.general_fun as gf
import fonctions.diagnosis_fun as df
import pandas as pd

def treatment_temp(state):
    st.header("Traitement des erreurs de séquence temporelle")
    gf.check_data_init(state.dataset_clean)
    if state.DF_TEMP.empty == True:
        st.warning("Aucune règles n'a été renseignée")
    else :
        for rule_temp in state.DF_TEMP.index:
            if st.checkbox("Règle "+str(rule_temp+1), key='checkloop3_'+str(rule_temp)) == True:
                df_flag = df.find_temporal_errors(state.dataset_clean, state.DF_TEMP, rule_temp)
                st.markdown("<h3 style='text-align: center; '>Liste des enregistrements ne respectant pas la contrainte</h1>", unsafe_allow_html=True)
                grid_response = gf.create_grid(df_flag, key='agloop2_'+str(rule_temp))
                data = grid_response['data'] #Ne pas commenter
                selected = grid_response['selected_rows']
                df_filtered = pd.DataFrame(selected)

                col1, col2, col3 = st.beta_columns([1, 1, 3])
                with col1:
                    if st.button("Supprimer toutes les lignes concernées", key='butloop2_'+str(rule_temp)):
                        #state.df_precedent_state = state.dataset_clean.copy()
                        nb_lignes = len(df_flag['row_breaking_rule'].to_list())
                        nature_modif = "Suppression des lignes violant une contrainte temporelle"
                        modif = "Suppression de "+str(nb_lignes)+" lignes violant la contrainte temporelle "+str(rule_temp+1)
                        state.dataset_clean = state.dataset_clean.drop(df_flag['row_breaking_rule'].to_list())
                        state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                with col2:
                    if st.button("Supprimer les lignes sélectionnées", key='butloop3_'+str(rule_temp)):
                        #state.df_precedent_state = state.dataset_clean.copy()
                        nb_lignes = len(df_filtered['row_breaking_rule'].to_list())
                        nature_modif = "Suppression de certaines lignes violant une contrainte temporelle"
                        modif = "Suppression de "+str(nb_lignes)+" lignes violant la contrainte temporelle "+str(rule_temp+1)
                        state.dataset_clean = state.dataset_clean.drop(df_filtered['row_breaking_rule'].values)
                        state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                        df_flag.reset_index(inplace=True)
                        df_filtered.reset_index(inplace=True)
                        ds1 = set([tuple(line) for line in df_flag.values])
                        ds2 = set([tuple(line) for line in df_filtered.values])
                        res = pd.DataFrame(list(ds1.difference(ds2)))
                        res.columns= df_flag.columns
                        df_flag = df_flag.set_index('index')
                with col3:
                    if st.checkbox("Souhaitez vous modifier la valeur des cellules concernées ?", key='checkloop4_'+str(rule_temp)):
                        selected_var = state.DF_TEMP.at[rule_temp, 'variable_temporelle']
                        st.info("Modification de la variable "+selected_var)
                        val = st.date_input('Entrer la valeur souhaitée', key='diloop2_'+str(rule_temp))
                        if st.button("Modifier la valeur", key='butloop4_'+str(rule_temp)):
                            nb_lignes = len(df_filtered['row_breaking_rule'].to_list())
                            nature_modif = "Imputation de certaines lignes violant une contrainte temporelle"
                            modif = "Imputation de "+str(nb_lignes)+" lignes violant la contrainte temporelle "+str(rule_temp+1)+". Valeur imputée: "+str(val)
                            for i in df_filtered['row_breaking_rule'].to_list():
                                state.dataset_clean.at[i,selected_var] = pd.to_datetime(val)
                            state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                            
    if state.display_df == True:
        gf.display_data(state.dataset_clean)

    if state.reorder_cols == True:
        state.dataset_clean, state.reorder_var, state.reorder_how = gf.reorder_columns(state.dataset_clean, state.reorder_var, state.reorder_how)

    if state.change_value == True:
        state.dataset_clean = gf.change_cell_value(state.dataset_clean)
        
    if state.display_state_df == True:
        state.dataset_clean, state.DF_STATE = gf.display_df_states(state.dataset_clean, state.DF_STATE)