import streamlit as st
import fonctions.general_fun as gf
import fonctions.treatment_fun as tf

def enrich_date(state):
    st.header("Enrichir à partir des variables temporelles")
    gf.check_data_init(state.dataset_clean)
    cols_date = []
    for col in state.dataset_clean.columns:
        data_type = gf.return_type(state.dataset_clean, col)
        if data_type == 'datetime64[ns]':
            cols_date.append(col)

    for col in state.dataset_clean.columns:
        print(type(col))
        
    with st.beta_expander("Extraction d'un pattern"):
        selected_col_date = st.selectbox("Choisir la variable à utiliser", cols_date, key='sb10')
        date_pattern = ['Weekday abbr (lun.)', 'Weekday complet (lundi)', 'Weekday chiffre (01)', 'Jour (18)', 'Mois abbr (jan.)', 'Mois complet (janvier)',
                        'Mois chiffre (01)', 'Année derniers chiffres (21)', 'Année complète (2021)']
        date_pattern_selected = st.selectbox('Choisir le pattern à extraire (Exemple : lundi 18 janvier 2021)', date_pattern, key='sb11')
        if st.button('Appliquer', key='but12'):
            #state.df_precedent_state = state.dataset_clean.copy()
            nature_modif = "Extraction d'un pattern d'une variable temporelle"
            modif = "Extraction du pattern "+date_pattern_selected+" pour la variable "+selected_col_date
            state.dataset_clean = tf.extract_date_pattern(state.dataset_clean, selected_col_date, date_pattern_selected)
            state.DF_STATE = gf.df_state_changed(state.DF_STATE, nature_modif, modif, state.dataset_clean)
            st.success('Modification effectuée')
    
    with st.beta_expander("Modification du format de la date"):
        selected_col_date = st.selectbox("Choisir la variable à utiliser", cols_date, key='sb43')
        date_format = ['18 Janvier 2021', 'lundi 18 Janvier 2021', '18/01/2021', '18-01-2021', '01/18/2021', '01-18-2021']
        date_format_manuel = st.checkbox("Définir le format manuellement ?", key="check51")
        if date_format_manuel == True:
            date_format_selected = st.text_input("Entrer le format désiré (ex: %d/%m)", key='ti12')
            st.write("[Lien vers la documentation](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior)")
        else:
            date_format_selected = st.selectbox('Choisir le format à utiliser (Exemple : lundi 18 janvier 2021)', date_format, key='sb44')
        if st.button('Appliquer', key='but62'):
            nature_modif = "Modification du format d'une variable temporelle" 
            if date_format_manuel == True:
                #state.df_precedent_state = state.dataset_clean.copy()
                state.dataset_clean[selected_col_date+'_formatted'] = state.dataset_clean[selected_col_date].dt.strftime(date_format_selected)
            else:
                #state.df_precedent_state = state.dataset_clean.copy()
                state.dataset_clean = tf.format_date(state.dataset_clean, selected_col_date, date_format_selected)
            modif = "Application du format "+date_format_selected+" pour la variable "+selected_col_date
            st.success('Modification effectuée')
            
    with st.beta_expander("Calculer une différence entre deux dates"):
        date1 = st.selectbox("Sélectionner la colonne correspondant à la première date", cols_date)
        date2 = st.selectbox("Sélectionner la colonne correspondant à la seconde date", cols_date)
        returned_unit = ['jours', 'minutes', 'secondes']
        unit = st.selectbox("Sélectionner l'unité désirée", returned_unit)
        if st.button("Calculer la différence entre les deux dates"):
            state.dataset_clean = tf.calculate_diff_between_dates(state.dataset_clean, date1, date2, returned_unit=unit)
    
    if state.display_df == True:
        gf.display_data(state.dataset_clean)

    if state.reorder_cols == True:
        state.dataset_clean, state.reorder_var, state.reorder_how = gf.reorder_columns(state.dataset_clean, state.reorder_var, state.reorder_how)
        
    if state.change_value == True:
        state.dataset_clean = gf.change_cell_value(state.dataset_clean)
        
    if state.display_state_df == True:
        state.dataset_clean, state.DF_STATE = gf.display_df_states(state.dataset_clean, state.DF_STATE)
