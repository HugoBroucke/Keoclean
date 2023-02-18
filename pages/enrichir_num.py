import streamlit as st
import fonctions.general_fun as gf
import fonctions.treatment_fun as tf
from streamlit_ace import st_ace

def enrich_num(state):
    st.header("Enrichir à partir des variables numériques")
    gf.check_data_init(state.dataset_clean)
    cols_num = []
    for col in state.dataset_clean.columns:
        data_type = gf.return_type(state.dataset_clean, col)
        if data_type =='float64':
            cols_num.append(col)

    with st.beta_expander("Discrétiser une variable"):
        selected_col_num = st.selectbox("Choisir la variable à discrétiser", cols_num, key='sb15')        
        bins = st.number_input("Définir le nombre de classes", value=int(2), min_value=2, key='ni7')
        if st.button("Appliquer", key='but15'):
            nature_modif = "Discrétisation d'une variable numérique"
            modif = "Discrétisation de la variable "+selected_col_num+" en "+str(bins)+" classes"
            #state.df_precedent_state = state.dataset_clean.copy()
            state.dataset_clean = tf.discretiser(state.dataset_clean, selected_col_num, bins, method='auto', labels=False)
            state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
            
    with st.beta_expander("Générer des percentiles"):
        selected_col_num = st.selectbox("Choisir la variable", cols_num, key='sb16')   
        if st.button("Appliquer", key='but16'):
            nature_modif = "Génération des percentiles d'une variable numérique"
            modif = "Génération des percentiles pour la variable "+selected_col_num
            #state.df_precedent_state = state.dataset_clean.copy()
            state.dataset_clean = tf.calculate_percentil(state.dataset_clean, selected_col_num)
            state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
            st.success('Modification effectuée')
    
                
    with st.beta_expander("Créer une fonction custom"):
        content = st_ace(value="#Ajouter à la ligne 4 la fonction souhaitée\n#x représente la valeur de la colonne considérée\ndef custom(x):\n\treturn x", language = 'python')
        col_used = st.selectbox("Sélectionner la colonne sur laquelle appliquer la fonction", cols_num)
        new_var = st.text_input("Entrer le nom de la variable à créer")
        if st.button("Créer la variable à partir de la fonction custom"):
            try:
                with open("tempDir/custom_fn.py", "w") as file:
                    file.write(content)
                from tempDir.custom_fn import custom
                state.dataset_clean[new_var] = state.dataset_clean[col_used].apply(custom)
            except:
                st.error("Une erreur est survenue")
            
    if state.display_df == True:
        gf.display_data(state.dataset_clean)

    if state.reorder_cols == True:
        state.dataset_clean, state.reorder_var, state.reorder_how = gf.reorder_columns(state.dataset_clean, state.reorder_var, state.reorder_how)
        
    if state.change_value == True:
        state.dataset_clean = gf.change_cell_value(state.dataset_clean)
        
    if state.display_state_df == True:
        state.dataset_clean, state.DF_STATE = gf.display_df_states(state.dataset_clean, state.DF_STATE)