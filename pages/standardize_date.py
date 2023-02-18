import streamlit as st
import fonctions.general_fun as gf

def standard_date(state):
    st.header("Standardisation des variables temporelles")
    gf.check_data_init(state.dataset_clean)
    cols_date = []
    for col in state.dataset_clean.columns:
        data_type = gf.return_type(state.dataset_clean, col)
        if data_type == 'datetime64[ns]':
            cols_date.append(col)
            
    with st.beta_expander("Standardisation"):
        st.write("Work in Progress")
        
    if state.display_df == True:
        gf.display_data(state.dataset_clean)

    if state.reorder_cols == True:
        state.dataset_clean, state.reorder_var, state.reorder_how = gf.reorder_columns(state.dataset_clean, state.reorder_var, state.reorder_how)
        
    if state.change_value == True:
        state.dataset_clean = gf.change_cell_value(state.dataset_clean)
        
    if state.display_state_df == True:
        state.dataset_clean, state.DF_STATE = state.dataset_clean, state.DF_STATE = gf.display_df_states(state.dataset_clean, state.DF_STATE)