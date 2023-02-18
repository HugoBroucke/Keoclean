import streamlit as st
import os
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode
import base64
import sqlite3 as sql
import pandas as pd
from fpdf import FPDF

class PDF(FPDF):
    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

def df_state_changed(dataset_state, nature_modif, modif, dataset):
    id_modif = len(dataset_state)
    dataset_state = dataset_state.append({'ID modification':id_modif, 'Nature de la modification':nature_modif, 'Modification effectuée':modif},
                                         ignore_index=True)
    conn = sql.connect('tempDir/tempBDD/keoclean_versioning.db')
    dataset.to_sql('keoclean_'+str(id_modif), conn, index_label='SQL_INDEX')
    conn.close()

    return dataset_state

def display_df_states(dataset, dataset_state):
    if dataset_state.empty:
        st.info("Aucune modification n'a été effectuée")
    else:
        st.markdown("<h3 style='text-align: center; '>Modifications effectuées</h1>", unsafe_allow_html=True)
        st.dataframe(dataset_state)
        
        col1, col2 = st.beta_columns(2)
        with col1:
            id_table = st.number_input("Choisir la version à charger", min_value=dataset_state['ID modification'].min(), max_value=dataset_state['ID modification'].max(),value=int())
            st.warning("Attention, les modification postérieures à la version sélectionnée seront supprimées")
            if st.button("Charger à la version sélectionnée"):
                conn = sql.connect('tempDir/tempBDD/keoclean_versioning.db')
                #Chargement de la version désirée
                dataset = pd.read_sql('SELECT * FROM keoclean_'+str(id_table), conn, index_col='SQL_INDEX')
                dataset.index.name = None
                #Suppression des tables postérieures
                index_to_drop = range(id_table+1, dataset_state["ID modification"].max()+1)
                cursor = conn.cursor()
                for i in index_to_drop:
                    cursor.execute("DROP TABLE keoclean_"+str(i))
                conn.commit()
                conn.close()
                #Suppression des modifications postérieures du rapport
                dataset_state = dataset_state.loc[:id_table]
                
                #return dataset, dataset_state
            
            with col2:
                 if st.button("Générer le rapport des modifications effectuées"):
                    pdf = PDF()
                    pdf.alias_nb_pages()
                    pdf.add_page()
                    pdf.set_font('arial', 'B', 20)
                    pdf.cell(0, 0, "Rapport des modifications effectuées sur le dataset", 0,1, align='C')
                    pdf.set_xy(10,20)
                    for i in dataset_state.index:
                        pdf.set_font('arial', 'B', 14)
                        pdf.multi_cell(0, 10, "Step "+str(i)+": "+str(dataset_state.at[i,'Nature de la modification']),0,1)
                       # pdf.ln()
                        pdf.set_font('arial', '', 10)
                        pdf.multi_cell(0, 10, str(dataset_state.at[i,'Modification effectuée']),0,1)
                    
                    
                    pdf.output("output/Cleaning_steps.pdf", 'F')
                    st.markdown(get_binary_file_downloader_html('output/Cleaning_steps.pdf', 'au format PDF'), unsafe_allow_html=True)
            
        return dataset, dataset_state

# def get_precedent_state(dataset, df_precedent_state):
#     dataset = df_precedent_state.copy()
#     df_precedent_state = df_precedent_state.drop(df_precedent_state.index)
    
#     return dataset, df_precedent_state

def clean_folder(PATH):
    for root, dirs, files in os.walk(PATH):
        for f in files:
            os.unlink(os.path.join(root, f))
           
def display_data(dataset):
    #st.markdown("Overview du dataset")
    st.markdown("<h3 style='text-align: center; '>Overview du dataset</h1>", unsafe_allow_html=True)
    st.dataframe(dataset.head(50))
    view_unique = st.checkbox("Visualiser les modalités d'une variable ?")
    if view_unique == True:   
        col = st.selectbox("Choisir une variable à analyser", dataset.columns)
        df = pd.DataFrame(dataset[col].value_counts(), columns=[col]).reset_index().sort_values(col, ascending=False)
        df = df.rename(columns={'non':"Nombre d'occurences", 'index':col})
        st.markdown("<h3 style='text-align: center; '>Modalités de la variable "+col+"</h1>", unsafe_allow_html=True)
        col1, col2, col3 = st.beta_columns([1,2,1])
        with col2:
            st.dataframe(df)
        


def change_cell_value(dataset):
    grid_response = create_grid_value_changed(dataset)
    data = grid_response['data'] #Ne pas commenter
    selected = grid_response['selected_rows']
    df_filtered = pd.DataFrame(selected)
    
    return data
    
def reorder_columns(dataset, reorder_var, reorder_how):
    with st.form("form_reorder"):
        var_to_reorder = st.selectbox("Choisir la variable", dataset.columns)
        how_to_reorder = st.selectbox("Choisir la méthode", ['ASCENDING', 'DESCENDING'])
       
        submit = st.form_submit_button("Valider")
    
    if submit:
        reorder_var.append(var_to_reorder)
        reorder_how.append(how_to_reorder)
    
    st.info("Variables concernées: "+str(reorder_var)+"\n Méthodes appliquées: "+str(reorder_how))
        
    col1, col2, col3 = st.beta_columns(3)
    with col1:
        if st.button('Réordonner le dataset', key='uhfefn'):
            how_to_reorder_bool = []
            for c in reorder_how:
                if c == 'ASCENDING':
                    how_to_reorder_bool.append(True)
                else:
                    how_to_reorder_bool.append(False)
                    
            dataset = dataset.sort_values(by=reorder_var, ascending=how_to_reorder_bool)
            st.write(dataset)
            reorder_var = []
            reorder_how = []
    with col2:
        if st.button("Supprimer les paramètres"):
            reorder_var = []
            reorder_how = []
    with col3:
        if st.button("Revenir au dataset par défaut"):
            reorder_var = []
            reorder_how = []
            dataset = dataset.sort_index()
        
    return dataset, reorder_var, reorder_how            
            
def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    else:
        return obj
    
def save_uploaded_file(uploadedfile):
  with open(os.path.join("tempDir",uploadedfile.name),"wb") as f:
     f.write(uploadedfile.getbuffer())

def add_value_to_dic(dic, key, value, i):
    """
    Permet d'ajouter une valeur à un dictionnaire sans avoir de duplicat.
    
    Input : nom du dictionnaire, clé, valeur à ajouté, type d'objet (ça peut être set(), list(), str, int ou float)
    Output : le dictionnaire créé/modifié
    """
    if key in dic: i = dic[key]
    if   isinstance(i, set):   i.add(value)
    elif isinstance(i, list):  i.append(value)
    elif isinstance(i, str):   i += str(value)
    elif isinstance(i, int):   i += int(value)
    elif isinstance(i, float): i += float(value)
    dic[key] = i
    
    return dic

def return_type(dataset, variable):
    """
    Permet de retourner le type de la variable
    
    Input : variable
    Output : type de la variable
    """
    data_type = dataset[variable].dtype
    
    return data_type

def create_grid(dataset, key=None):
    gb = GridOptionsBuilder.from_dataframe(dataset)
    gb.configure_selection(selection_mode='multiple', use_checkbox=True)
    gridOptions = gb.build()
    
    height = 50 + len(dataset) * 35
    if height > 500:
        height = 500
    elif height == 0:
        height = 70
        
    grid_response = AgGrid(dataset,
                       gridOptions=gridOptions,
                       data_return_mode=DataReturnMode.AS_INPUT,
                       update_mode=GridUpdateMode.MODEL_CHANGED,
                       height=height,
                       #reload_data=reload_data,
                       key=key)
    
    return grid_response

def create_grid_value_changed(dataset, key=None):
    gb = GridOptionsBuilder.from_dataframe(dataset)
    #gb.configure_selection(selection_mode='multiple', use_checkbox=True)
    gb.configure_default_column(editable=True)
    gridOptions = gb.build()
    
    height = 50 + len(dataset) * 35
    if height > 500:
        height = 500
    elif height == 0:
        height = 70
        
    grid_response = AgGrid(dataset,
                       gridOptions=gridOptions,
                       data_return_mode=DataReturnMode.AS_INPUT,
                       update_mode=GridUpdateMode.VALUE_CHANGED,
                       height=height,
                       #reload_data=reload_data,
                       key=key)
    
    return grid_response

def create_grid_return_filtered_and_sorted(dataset, key=None):
    gb = GridOptionsBuilder.from_dataframe(dataset)
    gb.configure_selection(selection_mode='multiple', use_checkbox=True)
    gridOptions = gb.build()
    
    height = 50 + len(dataset) * 35
    if height > 500:
        height = 500
    elif height == 0:
        height = 70
        
    grid_response = AgGrid(dataset,
                       gridOptions=gridOptions,
                       data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                       update_mode=GridUpdateMode.MODEL_CHANGED,
                       height=height,
                       #reload_data=reload_data,
                       key=key)
    
    return grid_response

def create_grid_return_filtered_and_sorted_no_selection_allowed(dataset, key=None):
    gb = GridOptionsBuilder.from_dataframe(dataset)
    gridOptions = gb.build()
    
    height = 50 + len(dataset) * 35
    if height > 500:
        height = 500
    elif height == 0:
        height = 70
        
    grid_response = AgGrid(dataset,
                       gridOptions=gridOptions,
                       data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                       update_mode=GridUpdateMode.MODEL_CHANGED,
                       height=height,
                       #reload_data=reload_data,
                       key=key)
    
    return grid_response

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Télécharger {file_label}</a>'
    return href

def check_data_init(dataset):
    if dataset is None:
        st.error("Le dataset n'a pas été chargé ! (Appuyer sur 'Valider' après import des données pour charger le dataset)")
        st.stop()

# def display_df_streamlit(dataset):
#     #st.dataframe ne marche pas pour les <NA> : pour contrer cela, on transforme les variables int en str puis on retransforme ensuite dans le bon type
#     col_int64 = []
#     for col in dataset.columns:
#         if dataset[col].dtype == 'Int64':
#             col_int64.append(col)
    
#     for colint in col_int64:
#         try:
#             dataset[colint] = dataset[colint].astype('Int64').astype('str').replace('<NA>', np.nan)
#         except : 
#             dataset[colint] = dataset[colint].astype(object)
    
#     DISPLAYED_DF = st.dataframe(dataset)
    
#     for colint in col_int64:
#         try:
#             dataset[colint] = dataset[colint].astype('float64').astype('Int64')
#         except :
#             dataset[colint] = dataset[colint].astype('float64')
            
#     return DISPLAYED_DF
