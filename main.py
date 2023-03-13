import streamlit as st
from streamlit.hashing import _CodeHasher
import pandas as pd
import chardet
import os
import os.path
import csv
import recordlinkage
import pickle as pkle
import sqlite3 as sql
import locale
locale.setlocale(locale.LC_TIME,'')
#import plotly.graph_objects as go

## Import des fonctions
import fonctions.general_fun as gf
import fonctions.screening_fun as sf

## Import des pages
import pages.renseigner_meta as s1
import pages.editer_contraintes as s2
import pages.profiling as d1
import pages.traitement_doublons as t1
import pages.traitement_missing as t2
import pages.traitement_hors_limite as t3
import pages.traitement_incompatible as t4
import pages.traitement_tempo as t5
import pages.standardize_num as t6
import pages.standardize_date as t7
import pages.standardize_text as t8
import pages.enrichir_num as t9
import pages.enrichir_date as t10
import pages.enrichir_txt as t11
import pages.enrichir_df as t12
import pages.enrichir_cdts as t13

## Gestion des informations de session
try:
    # Before Streamlit 0.65
    from streamlit.ReportThread import get_report_ctx
    from streamlit.server.Server import Server
except ModuleNotFoundError:
    # After Streamlit 0.65
    from streamlit.report_thread import get_report_ctx
    from streamlit.server.server import Server
  
# Trouver une façon d'éxécuter les fonctions seulement la première fois au démarrage de l'app 
# gf.clean_folder('output/')
# gf.clean_folder('tempDir/')
# gf.clean_folder('tempDir/tempProfiling/')
    
def main():
#https://discuss.streamlit.io/t/navigation-panel-can-we-have-a-button-on-the-right-hand-side-like-next-which-automatically-moves-to-the-next-page-in-the-navigation-panel/6588
    state = _get_state()
    st.sidebar.write("[Lien vers la documentation Keoclean](https://mongit.kaduceo.com/hb/keoclean/-/wikis/ACCUEIL-DOCUMENTATION-KEOCLEAN)")
    state.display_df = st.sidebar.checkbox("Afficher le dataset", key='check1')
    state.reorder_cols = st.sidebar.checkbox("Réordonner les colonnes", key='check19898')
    state.change_value =st.sidebar.checkbox("Modifier la valeur des cellules")
    state.display_state_df = st.sidebar.checkbox("Afficher le rapport des modifications effectuées", key='118')
    download_csv = st.sidebar.button("Exporter le dataset", key='but1')
    if download_csv :
        try:
            state.dataset_clean.to_csv("output/keoclean.csv", index=False)
            st.sidebar.markdown(gf.get_binary_file_downloader_html('output/keoclean.csv', 'le dataset néttoyé'), unsafe_allow_html=True)
        except:
            st.sidebar.error("Le dataset n'existe pas")
        
    st.sidebar.title("📌 Index")
    col1, mid, col2 = st.beta_columns([1,1,20])
    #with col1:
        #st.image('static/logoKaduceo.png', width=60)
    with col2:
        st.title("KeoClean")

    next = st.sidebar.button("Passer à l'étape suivante", key='but2')
        
    new_choice = [  "Importer les données",
                    "Renseigner les métadonnées", 
                    "Editer les contraintes",
                    "Profiling",
                    "Traitement des données dupliquées",
                    "Traitement des données manquantes",
                    "Traitement des valeurs hors limite",
                    "Traitement des valeurs incompatibles",
                    "Traitement des erreurs de séquence temporelle",
                    "Standardisation des variables numériques",
                    "Standardisation des variables temporelles",
                    "Standardisation des variables textuelles",
                    "Enrichir à partir des variables numériques" ,
                    "Enrichir à partir des variables temporelles",
                    "Enrichir à partir des variables textuelles",
                    "Enrichir à partir d'un autre dataset",
                    "Enrichir à partir de conditions"]

    if os.path.isfile('next.p'):
        next_clicked = pkle.load(open('next.p', 'rb'))
        if next_clicked == len(new_choice):
            next_clicked = 0 #Changer pour la derniere page
    else:
        next_clicked = 0 #the start
    
    if next:
        next_clicked = next_clicked +1

        if next_clicked == len(new_choice):
            next_clicked = 0
    
    # create your radio button with the index that we loaded
    choice = st.sidebar.radio("Sélectionner l'étape",
                   ("Importer les données",
                    "Renseigner les métadonnées", 
                    "Editer les contraintes",
                    "Profiling",
                    "Traitement des données dupliquées",
                    "Traitement des données manquantes",
                    "Traitement des valeurs hors limite",
                    "Traitement des valeurs incompatibles",
                    "Traitement des erreurs de séquence temporelle",
                    "Standardisation des variables numériques",
                    "Standardisation des variables temporelles",
                    "Standardisation des variables textuelles",
                    "Enrichir à partir des variables numériques" ,
                    "Enrichir à partir des variables temporelles",
                    "Enrichir à partir des variables textuelles",
                    "Enrichir à partir d'un autre dataset",
                    "Enrichir à partir de conditions"), index=next_clicked)
    
    # pickle the index associated with the value, to keep track if the radio button has been used
    pkle.dump(new_choice.index(choice), open('next.p', 'wb'))
    
    # finally get to whats on each page
    if choice == 'Importer les données':
        screening_import(state)
    if choice == 'Renseigner les métadonnées':
        s1.screening_meta(state)
    elif choice == 'Editer les contraintes':
        s2.screening_constraints(state)
    elif choice == 'Profiling':
        d1.profiling(state)
    elif choice == 'Traitement des données dupliquées':
        t1.treatment_doublons(state)
    elif choice == 'Traitement des données manquantes':
        t2.treatment_missing(state)
    elif choice == 'Traitement des valeurs hors limite':
        t3.treatment_hl(state)
    elif choice == 'Traitement des valeurs incompatibles':
        t4.treatment_incomp(state)
    elif choice == 'Traitement des erreurs de séquence temporelle':
        t5.treatment_temp(state)
    elif choice == 'Standardisation des variables numériques':
        t6.standard_num(state)
    elif choice == 'Standardisation des variables temporelles':
        t7.standard_date(state)
    elif choice == 'Standardisation des variables textuelles':
        t8.standard_txt(state)
    elif choice == 'Enrichir à partir des variables numériques':
        t9.enrich_num(state)
    elif choice == 'Enrichir à partir des variables temporelles':
        t10.enrich_date(state)
    elif choice == 'Enrichir à partir des variables textuelles':
        t11.enrich_txt(state)
    elif choice == "Enrichir à partir d'un autre dataset":
        t12.enrich_df(state)
    elif choice == "Enrichir à partir de conditions":
        t13.enrich_conditions(state)

    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()

def screening_import(state):
    
    uploaded_file = st.file_uploader("Choisir un fichier CSV", key='fu1')
    if st.checkbox("Mon fichier ne contient pas d'en tête"):
        header = None
    else:
        header = st.number_input("Entrer la ligne d'en tête du fichier", 0, key='ni1')
    encod_bool = st.checkbox("Définir manuellement l'encodage ?", key='check2')
    delim_bool = st.checkbox("Définir manuellement le délimiteur ?", key='check3')
    
    if uploaded_file is not None:
        if os.path.isfile("tempDir/"+uploaded_file.name):
            print('File exists')
        else:
            with st.spinner('Veuillez patienter, le fichier sera bientôt chargé !'):
                gf.save_uploaded_file(uploaded_file) 
        
    if encod_bool == True:
        encoding = st.text_input("Entrer l'encodage'du fichier CSV", key='ti1')
    else:
        if uploaded_file is not None:
            with open("tempDir/"+uploaded_file.name, 'rb') as rawdata:
                result = chardet.detect(rawdata.read(1000))
                encoding = result['encoding']
    
    if delim_bool == True:
        sep = st.text_input("Entrer le délimiteur du fichier csv", key='ti2')
    else:
        if uploaded_file is not None:
            with open("tempDir/"+uploaded_file.name, 'r') as csvfile:
                dialect = csv.Sniffer().sniff(csvfile.readline(1000))
                sep = dialect.delimiter
    
    if uploaded_file is not None:
        st.info("Encodage détecté : "+encoding+" --- Séparateur détecté : "+'"'+sep+'"')
        try:
            with st.spinner("Chargement du dataset, veuillez patienter ..."):
                state.dataset = pd.read_csv(uploaded_file, header=header, encoding=encoding, na_values='', sep=sep)
                if header == None:
                    for col in state.dataset.columns:
                        state.dataset = state.dataset.rename(columns={col:str(col)})
                state.var_to_reorder_l = []
                state.how_to_reorder_l = []
                ### Mise sous forme de formulaire pour éviter que la page se recharge à chaque fois
                form = st.form("my_form")
                filtered = form.multiselect("Sélectionner les colonnes", options=list(state.dataset.columns), default=list(state.dataset.columns))
                form.form_submit_button("Valider")
                #filtered = st.multiselect("Filter columns", options=list(state.dataset.columns), default=list(state.dataset.columns))
                
                authorize_manual_filter = st.checkbox("Filtrer manuellement sur certaines lignes ?")

                with st.form("my_form2"):
                    st.markdown("<h3 style='text-align: center; '>Sélectionner les lignes</h1>", unsafe_allow_html=True)
                    if authorize_manual_filter == True:                           
                        grid_response = gf.create_grid_return_filtered_and_sorted(state.dataset[filtered])
                        data = grid_response['data'] #Ne pas commenter
                        selected = grid_response['selected_rows']
                        df_filtered = pd.DataFrame(selected)
                    else:
                        grid_response = gf.create_grid_return_filtered_and_sorted_no_selection_allowed(state.dataset[filtered])
                        data = grid_response['data'] #Ne pas commenter
                        #selected = grid_response['selected_rows']
                        df_filtered = pd.DataFrame(data)
                        
                    
                    VAL_BUTTON = st.form_submit_button("Valider l'import des données")
                
                # with st.form("my_form3"):
                #     st.markdown("<h3 style='text-align: center; '>Réordonner les observations</h1>", unsafe_allow_html=True)

                #     var_to_reorder = st.selectbox("Sélectionner la variable sur laquelle ordonnée", df_filtered.columns)
                #     how_to_reorder = st.selectbox("Sélectionner l'ordre", ['ASCENDING', 'DESCENDING'])
                    
                #     if "var_to_reorder_l" in locals():
                #         print('oui')
                #         var_to_reorder_l.append(var_to_reorder)
                #     else:
                #         print('non')
                #         var_to_reorder_l = []
                #         var_to_reorder_l.append(var_to_reorder)
                        
                #     if "state.how_to_reorder_l" in locals():
                #             state.how_to_reorder_l.append(how_to_reorder)
                #     else:
                #         state.how_to_reorder_l = []
                #         state.how_to_reorder_l.append(how_to_reorder)
                #     #st.info("Liste des réordonnancements: "+state.var_to_reorder_l+"\n"+state.how_to_reorder_l)
                    
                #     OK_BUTTON = st.form_submit_button("Ajouter la variable à la liste")

                # st.write(var_to_reorder_l)
                #st.dataframe(state.dataset[filtered].head(100))
                
                #VAL_BUTTON = st.button("Valider l'import des données", key='but3')
                
                if st.checkbox("Afficher une courte analyse ?"):
                    returned_df, len_df, nb_cols, int_df, obj_df = sf.rapid_analysis(df_filtered)
                    st.markdown("<h3 style='text-align: center; '>Simple description du dataset</h1>", unsafe_allow_html=True)
                    st.write("Nombre de lignes: "+str(len_df))
                    st.write("Nombre de colonnes: "+str(nb_cols))
                    st.dataframe(returned_df)
                    st.markdown("<h3 style='text-align: center; '>Analyse des variables numériques</h1>", unsafe_allow_html=True)
                    st.dataframe(int_df)
                    st.markdown("<h3 style='text-align: center; '>Modalités les plus représentées des variables textuelles</h1>", unsafe_allow_html=True)
                    st.dataframe(obj_df)
            
            if VAL_BUTTON:
                with st.spinner('Chargement du dataset, création des variables globales ...'):
                    #Supprimme les tables existantes dans la db des versions
                    conn = sql.connect('tempDir/tempBDD/keoclean_versioning.db')
                    cursor = conn.cursor()
                    for i in range(99999): 
                        try:
                            dropTableStatement = "DROP TABLE keoclean_"+str(i)
                            cursor.execute(dropTableStatement)
                        except:
                            break
                    conn.commit()
                    conn.close()
                    
                    state.dataset_clean = df_filtered #init dataset_clean
                    state.SHOW_DF_CLEAN = False
                    state.reorder_var = []
                    state.reorder_how = []
                    state.df_precedent_state = pd.DataFrame()
                    state.DF_STATE = pd.DataFrame(columns=['ID modification','Nature de la modification', 'Modification effectuée'])
                    state.DF_STATE = gf.df_state_changed(state.DF_STATE, 'Initialisation du dataset', 'Initialisation du dataset', state.dataset_clean)

                    ###Création des variables globales###
                    # Renseigner métadonnées
                    state.DF_META = pd.DataFrame(columns=['variable', 'type', 'master', 'nullable', 'boolean', 'unique', 'min_value', 'max_value', 'exclude_values'])
                    state.dataset_clean, state.DF_META = sf.init_META(state.dataset_clean, state.DF_META)#Initialisation des métadonnées
                    state.dic_excluded_values = {}
                    # Editer contraintes
                    state.DF_IF = pd.DataFrame(columns=['ruleID','variable', 'connecteur', 'comparison_value', 'logique'])
                    state.DF_THEN = pd.DataFrame(columns=['ruleID', 'variable', 'connecteur', 'comparison_value', 'logique'])
                    state.ruleID = 1
                    state.etat_contrainte = "IF"
                    state.var_contrainte = state.dataset_clean.columns.to_list()
                    state.var_temp = state.dataset_clean.columns.to_list()
                    state.temp_keys = [] 
                    state.DF_TEMP = pd.DataFrame(columns=['clé', 'variable_temporelle'])
                    # Traitement dupliquées
                        #Lignes
                    state.b1onclick = False
                    state.duplicate_lignes = pd.DataFrame()
                        #Colonnes
                    state.sig_matrix = None
                    state.b2onclick = False
                        #Enregistrements similaires
                    state.compare_cl = recordlinkage.Compare()
                    state.nb_keys_rl = 0
                    state.df_rl_keys = pd.DataFrame(columns=['variable', 'distance', 'seuil'])
                    state.df_rl_groups = pd.DataFrame()
                    state.block = []
                    state.rl_executed = False
                    state.fill_color_rl = ['white']
                    # Traitement missing
                    state.b3onclick = False
                    state.df_lig_mis=pd.DataFrame()
                    state.b4onclick = False
                    state.df_col_mis=pd.DataFrame()
                    # Traitement out of range
                        #OOR
                    state.b5onclick = False
                    state.df_out_of_range_val = pd.DataFrame()
                        #Outliers
                    state.b6onclick = False
                    state.df_outliers = pd.DataFrame()
                    # Standardisation
                        # Variable text
                    state.clusters = pd.DataFrame()
                    state.df_transitoire = pd.DataFrame()
                    state.df_tomerge = pd.DataFrame()
                    state.show_grid = False
                    # Enrichissement
                        # variable text
                    state.df_dummy = pd.DataFrame()
                    state.df_dummy2 = pd.DataFrame()
                    state.b_ner2_onclick = False
                    state.b_ner1_onclick = False
                    state.group_cols = []
                        # dataset
                    state.dataset_added = pd.DataFrame()
                    state.df_cles = pd.DataFrame(columns=['Mapping_df_origine', 'Mapping_nouveau_df'])
                        # conditions
                    state.DF_IF_ENRICH = pd.DataFrame(columns=['variable', 'connecteur', 'comparison_value'])
                    state.var_contrainte_cdt = state.dataset_clean.columns.to_list()
                
        except:
            st.error("Le dataset n'a pas pu être chargé, veuillez modifier manuellement l'encodage et/ou le délimiteur")

class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)
        
    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value
    
    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()
    
    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False
        
        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    
    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state




if __name__ == "__main__":
    main()
