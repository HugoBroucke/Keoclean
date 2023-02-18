import streamlit as st
import fonctions.general_fun as gf
import fonctions.treatment_fun as tf
import numpy as np
import pandas as pd

def enrich_conditions(state):
    st.header("Enrichir à partir de conditions")
    with st.beta_expander("Créer des colonnes à partir de conditions"):
        gf.check_data_init(state.dataset_clean)
        st.subheader("Entrer les conditions")
        selected_var_cdt = st.selectbox('Sélectionner la variable', state.var_contrainte_cdt, key='100')
        var_type_cdt = gf.return_type(state.dataset_clean, selected_var_cdt)
    
        if var_type_cdt == 'float64': #Options pour variables numériques
            con_list = ['egal', 'différent', 'supérieur', 'inférieur', 'supérieur ou égal', 'inférieur ou égal']
            selected_con_cdt = st.selectbox("Sélectionner une opération logique", con_list, key='101')
            val_cdt = st.number_input("Sélectionner une valeur à comparer", value=float(), key='102')
        elif var_type_cdt == 'datetime64[ns]': #Options pour variable date
            con_list = ['egal', 'différent', 'supérieur', 'inférieur', 'supérieur ou égal', 'inférieur ou égal']
            selected_con_cdt = st.selectbox("Sélectionner une opération logique", con_list, key='103')
            val_cdt = st.date_input("Sélectionner une date à comparer", key='104')       
        else: #Options pour variables textuelles
            con_list = ['egal', 'différent', 'commence par', 'fini par', 'contient']
            selected_con_cdt = st.selectbox("Sélectionner une opération logique", con_list, key='105')
            val_cdt = st.text_input("Entrer un pattern à comparer", key='106')
    
        colA, colB, colC = st.beta_columns([2, 1, 2])
        with colA:
            if st.button("Ajouter la condition", key='107'):
                state.DF_IF_ENRICH = state.DF_IF_ENRICH.append({'variable':selected_var_cdt, 'connecteur':selected_con_cdt, 'comparison_value':val_cdt}, ignore_index=True)
                state.var_contrainte_cdt.remove(selected_var_cdt)
        with colC:
            if st.button('Supprimer les conditions', key='supcon'):
                state.DF_IF_ENRICH = state.DF_IF_ENRICH.drop(state.DF_IF_ENRICH.index)
                state.var_contrainte_cdt = state.dataset_clean.columns.to_list()
            
        st.markdown("<h3 style='text-align: center; '>Conditions</h1>", unsafe_allow_html=True)
        col1, col2, col3 = st.beta_columns([1,10,1]) 
        with col2:
            st.dataframe(state.DF_IF_ENRICH)
        
        st.subheader("Choisir la variable à créer ou à modifier")
        use_existing_var = st.radio("Sélectionner une option",("Utiliser une variable existante", "Créer une nouvelle variable héritant des valeurs d'une variable existante", "Créer une nouvelle variable"))
        #use_existing_var = st.checkbox("Utiliser une variable existante", key='108')
        #inherit_from_var = st.checkbox("Créer une nouvelle variable héritant des valeurs d'une variable existante", key='inherit')
        if use_existing_var == "Utiliser une variable existante" :
            selected_var = st.selectbox('Choisir la variable à modifier', state.dataset_clean.columns, key='109')
            var_type = gf.return_type(state.dataset_clean, selected_var)
            if var_type == 'float64':
                val = st.number_input("Choisir la valeur", value=float(), key='110')
            elif (var_type == 'datetime64[ns]'):
                val = st.date_input("Choisir la valeur", key='111')
            else: #Options pour variables textuelles
                val = st.text_input("Choisir la valeur", key='112')
        
        elif use_existing_var == "Créer une nouvelle variable héritant des valeurs d'une variable existante":
            selected_var = st.selectbox('Choisir la variable dont les valeurs seront héritées', state.dataset_clean.columns, key='inhe')
            new_var = st.text_input("Entrer le nom de la nouvelle variable")
            var_type = gf.return_type(state.dataset_clean, selected_var)
            if var_type == 'object':
                val = st.text_input("Choisir la valeur", key='mykey1')
            elif var_type == 'datetime64[ns]':
                val = st.date_input("Choisir la valeur", key='mykey2')
            else:
                val = st.number_input("Choisir la valeur", value=float(), key='mykey3')
                
        else:
            selected_var = st.text_input("Entrer le nom de la variable à créer (ATTENTION: si le nom de la variable existe déjà, elle sera remplacée)", key='113')
            type_liste = ['STRING', 'NUMBER', 'DATE']
            var_type = st.selectbox('Choisir le type de la variable', type_liste, key='114')      
            if var_type =='NUMBER': #Options pour variables numériques
                val = st.number_input("Choisir la valeur", value=float(), key='115')
            elif var_type == 'DATE': #Options pour variable date
                val = st.date_input("Choisir la valeur", key='116')       
            else: #Options pour variables textuelles
                val = st.text_input("Choisir la valeur", key='117')
        
        if st.button('Valider'):
            if state.DF_IF_ENRICH.empty == True:
                st.error('Veuillez ajouter au moins une condition')
            if use_existing_var == 'Créer une nouvelle variable':
                state.dataset_clean[selected_var] = np.nan
                nature_modif = "Création d'une colonne à partir d'une condition"
                modif = "Création de la variable "+selected_var+" = "+str(val)+" SI : "
                for l in state.DF_IF_ENRICH.index:
                    modif = modif+str(state.DF_IF_ENRICH.at[l,'variable'])+' '+str(state.DF_IF_ENRICH.at[l,'connecteur'])+' '+str(state.DF_IF_ENRICH.at[l,'comparison_value'])
                    if l != len(state.DF_IF_ENRICH)-1:
                        modif = modif+'AND '
                state.dataset_clean = tf.create_var_from_cdt(state.dataset_clean, state.DF_IF_ENRICH, selected_var, val)
                state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                state.DF_IF_ENRICH = state.DF_IF_ENRICH.drop(state.DF_IF_ENRICH.index)
                state.var_contrainte_cdt = state.dataset_clean.columns.to_list()
            elif use_existing_var == "Utiliser une variable existante":
                nature_modif = "Modification d'une colonne à partir d'une condition"
                modif = "Modification de la variable "+selected_var+" = "+str(val)+" SI : "
                for l in state.DF_IF_ENRICH.index:
                    modif = modif+str(state.DF_IF_ENRICH.at[l,'variable'])+' '+str(state.DF_IF_ENRICH.at[l,'connecteur'])+' '+str(state.DF_IF_ENRICH.at[l,'comparison_value'])
                    if l != len(state.DF_IF_ENRICH)-1:
                        modif = modif+' AND '
                #state.df_precedent_state = state.dataset_clean.copy()
                state.dataset_clean = tf.create_var_from_cdt(state.dataset_clean, state.DF_IF_ENRICH, selected_var, val)
                state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                state.DF_IF_ENRICH = state.DF_IF_ENRICH.drop(state.DF_IF_ENRICH.index)
                state.var_contrainte_cdt = state.dataset_clean.columns.to_list()
            else:
                nature_modif = "Création d'une colonne à partir d'une condition en héritant des valeurs d'une variable existante"
                modif = "Création de la variable "+new_var+" = "+str(val)+" SI : "
                for l in state.DF_IF_ENRICH.index:
                    modif = modif+str(state.DF_IF_ENRICH.at[l,'variable'])+' '+str(state.DF_IF_ENRICH.at[l,'connecteur'])+' '+str(state.DF_IF_ENRICH.at[l,'comparison_value'])
                    if l != len(state.DF_IF_ENRICH)-1:
                        modif = modif+' AND '
                modif = modif + ' SINON, hérite des valeurs de la variable '+selected_var
                #state.df_precedent_state = state.dataset_clean.copy()
                state.dataset_clean = tf.create_var_from_cdt(state.dataset_clean, state.DF_IF_ENRICH, new_var, val, inherit=True, old_var=selected_var)
                state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
                state.DF_IF_ENRICH = state.DF_IF_ENRICH.drop(state.DF_IF_ENRICH.index)
                state.var_contrainte_cdt = state.dataset_clean.columns.to_list()

    with st.beta_expander("Ajouter une nouvelle colonne"):
        selected_var = st.text_input("Entrer le nom de la variable à créer", key='lalla113')
        type_liste = ['STRING', 'NUMBER', 'DATE']
        var_type = st.selectbox('Choisir le type de la variable', type_liste, key='lalla')      
        if var_type =='NUMBER': #Options pour variables numériques
            val = st.number_input("Choisir la valeur", value=float(), key='lalla1')
        elif var_type == 'DATE': #Options pour variable date
            val = st.date_input("Choisir la valeur", key='lalla116')       
        else: #Options pour variables textuelles
            val = st.text_input("Choisir la valeur", key='lalla117')
            
        if st.button('Valider', key='reee'):
            state.dataset_clean[selected_var] = val
            nature_modif = "Création d'une nouvelle colonne"
            modif = "Création de la colonne "+selected_var+" ("+var_type+"), initialisation par la valeur "+str(val)
            state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
            
    if state.display_df == True:
        gf.display_data(state.dataset_clean)

    if state.reorder_cols == True:
        state.dataset_clean, state.reorder_var, state.reorder_how = gf.reorder_columns(state.dataset_clean, state.reorder_var, state.reorder_how)
        
    if state.change_value == True:
        state.dataset_clean = gf.change_cell_value(state.dataset_clean)
        
    if state.display_state_df == True:
        state.dataset_clean, state.DF_STATE = gf.display_df_states(state.dataset_clean, state.DF_STATE)