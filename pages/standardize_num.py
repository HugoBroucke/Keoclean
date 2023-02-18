import streamlit as st
import fonctions.general_fun as gf
import math
from scipy.stats import shapiro
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def standard_num(state):
    st.header("Standardisation des variables numériques")
    gf.check_data_init(state.dataset_clean)
    cols_num = []
    for col in state.dataset_clean.columns:
        data_type = gf.return_type(state.dataset_clean, col)
        #if (data_type == 'Int64') | (data_type =='float64'):
        if data_type =='float64':
            cols_num.append(col)
            
    with st.beta_expander("Standardisation"):
        selected_col_num = st.selectbox("Choisir une variable à modifier", cols_num, key='sb26')
        fn_num = ['MinMaxScaling', 'Standardiser', "Mettre à l'échelle logarithmique"]
        selected_fn_num = st.selectbox("Choisir une fonction", fn_num, key='sb27')
        if selected_fn_num == 'MinMaxScaling':
            st.write("[Lien vers la librairie utilisée](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)")
        elif selected_fn_num == 'Standardiser':
            st.write("[Lien vers la librairie utilisée](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)")
        elif selected_fn_num == "Mettre à l'échelle logarithmique":
            base_set = st.checkbox("Fixer une base ? (Le logarithme néperien est utilisé par défaut)", key='check13')
            if base_set == True:
                base = st.number_input("Selectionner la base (défaut = logarithme néperien)", value=float(math.e), key='ni10')
            else :
                base = math.e

        if st.button('Appliquer', key='but28'):
            nature_modif = "Standardisation d'une variable numérique"
            modif = "Standardisation de la variable "+selected_col_num+" avec la méthode: "+selected_fn_num
            if selected_fn_num == "Standardiser":
                p_val = shapiro(state.dataset_clean[selected_col_num].sample(5000).dropna())[1] #Vérifie la normalité
                if p_val < 0.05:
                    st.warning("Test Shapiro -  p-val = "+str(round(p_val, 5))+" : la standardisation est déconseillé")
                else :
                    st.success("Test Shapiro -  p-val = "+str(round(p_val, 5)))
                #state.df_precedent_state = state.dataset_clean.copy()

                state.dataset_clean[selected_col_num] = StandardScaler().fit_transform(state.dataset_clean[[selected_col_num]])
            elif selected_fn_num == "MinMaxScaling":
                #state.df_precedent_state = state.dataset_clean.copy()
                state.dataset_clean[selected_col_num] = MinMaxScaler().fit_transform(state.dataset_clean[[selected_col_num]])
            elif selected_fn_num == "Mettre à l'échelle logarithmique":
                #state.df_precedent_state = state.dataset_clean.copy()
                state.dataset_clean[selected_col_num] =state.dataset_clean[selected_col_num].apply(lambda x: math.log(x, base))
                
            state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
            st.success('Modification effectuée')
                
    with st.beta_expander("Arrondir la variable"):
        selected_col_num_arr = st.selectbox("Choisir une variable à modifier", cols_num, key='sb28')
        arrondi = st.number_input("Choisir l'arrondi", min_value=1, value=int(2), key='ni11')
        if st.button('Appliquer', key='but29'):
            nature_modif = "Arrondi d'une variable numérique"
            modif = "Arrondi de la variable "+selected_col_num_arr+" ("+str(arrondi)+" chiffre(s))" 
            #state.df_precedent_state = state.dataset_clean.copy()
            state.dataset_clean = state.dataset_clean.round({selected_col_num_arr:arrondi})
            state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
            st.success('Modification effectuée')
            
    if state.display_df == True:
        gf.display_data(state.dataset_clean)

    if state.reorder_cols == True:
        state.dataset_clean, state.reorder_var, state.reorder_how = gf.reorder_columns(state.dataset_clean, state.reorder_var, state.reorder_how)

    if state.change_value == True:
        state.dataset_clean = gf.change_cell_value(state.dataset_clean)
        
    if state.display_state_df == True:
        state.dataset_clean, state.DF_STATE = state.dataset_clean, state.DF_STATE = gf.display_df_states(state.dataset_clean, state.DF_STATE)