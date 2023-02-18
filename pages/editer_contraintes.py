import streamlit as st
import fonctions.general_fun as gf
import numpy as np

def screening_constraints(state):
    st.header("Renseigner les contraintes")
    gf.check_data_init(state.dataset_clean)
    with st.beta_expander("Edition des règles métiers", expanded=True):
        if state.etat_contrainte == 'IF': #Création de l'interface IF
            #st.subheader("Déclaration SI pour la règle "+str(state.ruleID))
            st.markdown("<h3 style='text-align: center; color: blue'>Déclaration SI pour la règle "+str(state.ruleID)+"</h1>", unsafe_allow_html=True)
            selected_var = st.selectbox("Sélectionner une variable", state.var_contrainte, key='sb1')
            var_type = gf.return_type(state.dataset_clean, selected_var)
            
            if var_type != 'object': #Options pour variables non textuelles
                #con_list = ['egal', 'différent', 'supérieur', 'inférieur', 'supérieur ou égal', 'inférieur ou égal']
                con_list = ['egal', 'différent', 'supérieur', 'inférieur', 'supérieur ou égal', 'inférieur ou égal', 'is missing']
                selected_con = st.selectbox("Sélectionner une opération logique", con_list, key='sb2')
                if selected_con == 'is missing':
                    val = np.nan
                else:
                    if var_type == 'datetime64[ns]': #Options pour variable date
                        val = st.date_input("Sélectionner une date à comparer", key='di1')
                    else :
                        val = st.number_input("Sélectionner une valeur à comparer", value=float(), key='ni2')
                        
            else: #Options pour variables textuelles
                #con_list = ['egal', 'différent', 'commence par', 'fini par']
                con_list = ['egal', 'différent', 'commence par', 'fini par', 'contient', 'is missing']
                selected_con = st.selectbox("Sélectionner une opération logique", con_list, key='sb3')
                if selected_con == 'is missing':
                    val = np.nan
                else:
                    val = st.text_input("Entrer un pattern à comparer", key='ti3')
            
            b_if1, b_if2 = st.beta_columns(2)
            with b_if1:
                if st.button("Ajouter une autre condition (AND)", key='bu3'):
                    state.DF_IF = state.DF_IF.append({'ruleID': state.ruleID, 'variable':selected_var, 'connecteur':selected_con, 'comparison_value':val, 'logique':'AND'}, ignore_index=True)
                    state.var_contrainte.remove(selected_var) #On enlève la variable de la liste

            with b_if2:
                if st.button("Ajouter la condition puis passer à la déclaration THEN", key='bu4'):
                    state.DF_IF = state.DF_IF.append({'ruleID': state.ruleID, 'variable':selected_var, 'connecteur':selected_con, 'comparison_value':val, 'logique':'THEN'}, ignore_index=True)
                    state.var_contrainte.remove(selected_var) #On enlève la variable de la liste
                    state.etat_contrainte = 'THEN'
                
            show_if1, show_if2 = st.beta_columns(2)
            with show_if1:
                st.markdown("<h3 style='text-align: center;color: blue '>Déclarations SI</h1>", unsafe_allow_html=True)
                st.dataframe(state.DF_IF)

            with show_if2:
                st.markdown("<h3 style='text-align: center;color: red '>Déclarations ALORS</h1>", unsafe_allow_html=True)
                st.dataframe(state.DF_THEN)
            
            #Suppression de règles
            ruleID_to_del = st.number_input("Entrer le numéro de la règle à supprimer", value=int(1), key='ni3')
            if st.button('Supprimer la règle', key='bu5'):
                state.DF_IF = state.DF_IF.drop(state.DF_IF[state.DF_IF['ruleID']==ruleID_to_del].index)
                state.DF_THEN = state.DF_THEN.drop(state.DF_THEN[state.DF_THEN['ruleID']==ruleID_to_del].index)

                                                ##### INTERFACE THEN #####
                                                
        if state.etat_contrainte == 'THEN': #Création de l'interface THEN
            #st.subheader("Déclaration ALORS pour la règle "+str(state.ruleID))
            st.markdown("<h3 style='text-align: center; color: red'>Déclaration ALORS pour la règle "+str(state.ruleID)+"</h1>", unsafe_allow_html=True)
            selected_var = st.selectbox("Sélectionner une variable", state.var_contrainte, key='sb4')
            var_type = gf.return_type(state.dataset_clean, selected_var)
            
            if var_type != 'object':
                #con_list = ['egal', 'différent', 'supérieur', 'inférieur', 'supérieur ou égal', 'inférieur ou égal']
                con_list = ['egal', 'différent', 'supérieur', 'inférieur', 'supérieur ou égal', 'inférieur ou égal', 'is missing']
                selected_con = st.selectbox("Sélectionner une opération logique", con_list, key='sb5')
                if selected_con == 'is missing':
                    val = np.nan
                else:
                    if var_type == 'datetime64[ns]':
                        val = st.date_input("Sélectionner une date à comparer", key='di2')
                    else :
                        val = st.number_input("Sélectionner une valeur à comparer", value=float(), key='ni4')
            else:
                #con_list = ['egal', 'différent', 'commence par', 'fini par']
                con_list = ['egal', 'différent', 'commence par', 'fini par', 'contient', 'is missing']
                selected_con = st.selectbox("Sélectionner une opération logique", con_list, key='sb6')
                if selected_con == 'is missing':
                    val = np.nan
                else:
                    val = st.text_input("Entrer un pattern à comparer", key='ti4')
            
            b_then1, b_then2 = st.beta_columns(2)
            with b_then1:
                if st.button("Ajouter une autre condition (AND)", key='but6'):
                    state.DF_THEN = state.DF_THEN.append({'ruleID': state.ruleID, 'variable':selected_var, 'connecteur':selected_con, 'comparison_value':val, 'logique':'AND'}, ignore_index=True)
                    state.var_contrainte.remove(selected_var) #On enlève la variable de la liste

            with b_then2:
                if st.button("Ajouter la condition puis passer à la règle suivante", key='but7'):
                    state.DF_THEN = state.DF_THEN.append({'ruleID': state.ruleID, 'variable':selected_var, 'connecteur':selected_con, 'comparison_value':val, 'logique':'END'}, ignore_index=True)
                    state.var_contrainte = state.dataset_clean.columns.to_list()
                    state.ruleID = state.ruleID +1
                    state.etat_contrainte = "IF"
                
            show_then1, show_then2 = st.beta_columns(2)
            with show_then1:
                st.markdown("<h3 style='text-align: center;color: blue '>Déclarations SI</h1>", unsafe_allow_html=True)
                st.dataframe(state.DF_IF)

            with show_then2:
                st.markdown("<h3 style='text-align: center;color: red '>Déclarations ALORS</h1>", unsafe_allow_html=True)
                st.dataframe(state.DF_THEN)
            
            #Suppression de règles
            ruleID_to_del = st.number_input("Entrer le numéro de la règle à supprimer", value=int(1), key='ni5')
            if st.button('Supprimer la règle', key='but8'):
                state.DF_IF = state.DF_IF.drop(state.DF_IF[state.DF_IF['ruleID']==ruleID_to_del].index)
                state.DF_THEN = state.DF_THEN.drop(state.DF_THEN[state.DF_THEN['ruleID']==ruleID_to_del].index)

                                            ##### EDITION DES CONTRAINTES TEMPORELLES #####
                                            
    with st.beta_expander("Edition des contraintes temporelles"):    
        selected_col = st.selectbox("Choisir une variable à ajouter à la clé", state.var_temp, key='sb7')
        if st.button("Ajouter la variable à la clé", key='but9'):
            state.temp_keys.append(selected_col)
            state.var_temp.remove(selected_col)
            
        st.warning("Clé : "+ str([key for key in state.temp_keys]) + " ")
        
        cols_temp=[]
        for v in state.dataset_clean.columns:
            data_type = gf.return_type(state.dataset_clean, v)
            if (data_type == 'datetime64[ns]') & (v not in state.DF_TEMP['variable_temporelle'].values):
                cols_temp.append(v)
        selected_col_temp = st.selectbox("Choisir la variable temporelle à ajouter en contrainte", cols_temp, key='sb8')
        if st.button("Ajouter la contrainte temporelle", key='but10'):
            if not selected_col_temp:
                st.error("Aucune variable temporelle sélectionnée")
            elif not state.temp_keys:
                st.error("La clé est vide, veuillez sélectionner au moins une variable à ajouter à la clé")
            else:
                state.DF_TEMP = state.DF_TEMP.append({'clé':state.temp_keys, 'variable_temporelle':selected_col_temp}, ignore_index=True)
                state.temp_keys = []
                state.var_temp = state.dataset_clean.columns.to_list()

        st.markdown("<h3 style='text-align: center; '>Liste des règles temporelles</h1>", unsafe_allow_html=True)
        colt1, colt2, colt3 = st.beta_columns([1,10,1])
        with colt2:
            st.dataframe(state.DF_TEMP)
        
        #Suppression de règles
        if state.DF_TEMP.empty == False:
            ruleTEMP_to_del = st.selectbox("Choisir la variable temporelle à supprimer des contraintes", state.DF_TEMP['variable_temporelle'].values, key='sb9')
            if st.button('Supprimer la règle', key='but11'):
                state.DF_TEMP = state.DF_TEMP.drop(state.DF_TEMP[state.DF_TEMP['variable_temporelle']==ruleTEMP_to_del].index)
    
    if state.display_df == True:
        gf.display_data(state.dataset_clean)

    if state.reorder_cols == True:
        state.dataset_clean, state.reorder_var, state.reorder_how = gf.reorder_columns(state.dataset_clean, state.reorder_var, state.reorder_how)
        
    if state.change_value == True:
        state.dataset_clean = gf.change_cell_value(state.dataset_clean)
        
    if state.display_state_df == True:
        state.dataset_clean, state.DF_STATE = gf.display_df_states(state.dataset_clean, state.DF_STATE)