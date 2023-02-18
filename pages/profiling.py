import streamlit as st
import fonctions.general_fun as gf
import fonctions.profiling_fun as pf
import plotly.graph_objects as go
from fpdf import FPDF
import matplotlib.pyplot as plt

class PDF(FPDF):
    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')
        
def profiling(state):
    st.header("Profiling")
    gf.check_data_init(state.dataset_clean)
    big_df = st.checkbox("Réaliser une analyse rapide ?")
    st.info("l'analyse rapide est recommandé pour des daatsets volumineux. L'analyse rapide n'exécutera pas les fonctions suivantes : \n - Analyse des corrélations entre les variables")
    if st.button("Démarrer l'analyse", key='but25'):
        pdf = PDF()
        pdf.alias_nb_pages()
        pdf.add_page()
        pdf.set_xy(0, 10)
        pdf.set_font('arial', 'B', 30)
        with st.spinner("Analyse en cours, cette opération peut prendre du temps ..."):
            st.subheader("Description du dataset")
            pdf.cell(0, 0, "Description du dataset", align='C')
            
            nb_var, nb_lignes, histo, camembert = pf.describe_df(state.dataset_clean, state.DF_META)
            
            st.markdown("**Nombre de lignes : "+str(nb_lignes)+"**")
            pdf.set_font('arial', 'B', 12)
            pdf.set_xy(0, 30)
            pdf.cell(0, 0, "Nombre de lignes : "+str(nb_lignes), align='C')
            
            st.markdown("**Nombre de variables : "+str(nb_var)+"**")
            pdf.set_xy(0, 40)
            pdf.cell(0, 0, "Nombre de variables : "+str(nb_var), align='C')
            
            st.markdown("**Analyse des incohérences dans les métadonnées**")
            liste_unique_false = []
            liste_bool_false = []
            for i in state.DF_META.index:
                var = state.DF_META.at[i, 'variable']
                if state.DF_META.at[i, 'unique'] == True:
                    if len(state.dataset_clean[var].unique()) != len(state.dataset_clean):
                        liste_unique_false.append(var)
                if state.DF_META.at[i, 'boolean'] == True:
                    if len(state.dataset_clean[var].unique()) != 2:
                        liste_bool_false.append(var)
            pdf.set_xy(0, 50)
            pdf.cell(0, 0, "Analyse des incohérences dans les métadonnées", align='C')
            pdf.set_font('arial', '', 12)
            if not liste_unique_false:
                st.markdown("Pas d'incohérences détectée pour les variables uniques")
                pdf.set_xy(0, 60)
                pdf.cell(0, 0, "Pas d'incohérences détectée pour les variables uniques", align='C')
            else:
                modif = ''
                for v in liste_unique_false:
                    add_var = v+' '
                    modif = modif + add_var
                st.markdown("Les variables: "+modif+" ne sont pas des clés primaires")
                pdf.set_xy(0, 60)
                pdf.cell(0, 0, "Les variables: "+modif+" ne sont pas des clés primaires", align='C')
                
            if not liste_bool_false:
                st.markdown("Pas d'incohérence détectée pour les variables uniques")
                pdf.set_xy(0, 70)
                pdf.cell(0, 0, "Pas d'incohérences détectée pour les variables uniques", align='C')
            else:
                modif = ''
                for v in liste_bool_false:
                    add_var = v+' '
                    modif = modif + add_var
                st.markdown("Les variables: "+modif+" ne sont pas booléennes")
                pdf.set_xy(0, 70)
                pdf.cell(0, 0, "Les variables: "+modif+" ne sont pas des clés primaires", align='C')            
              
            pdf.set_xy(0, 80)
            if state.DF_META.empty == True:
                st.markdown("**Les métadonnées n'ont pas été renseignées !**")
                pdf.cell(0, 0, "Les métadonnées n'ont pas été renseignées !", align='C')
                pdf.set_xy(0, 100)
            else:
                st.pyplot(histo)             
                pdf.image('tempDir/tempProfiling/anaylse_meta1.png', x=60, w = 100, h = 100)
            
            st.pyplot(camembert) 
            pdf.image('tempDir/tempProfiling/anaylse_meta2.png', x=65, w = 0, h = 60)     

            ##### ANALYSE INCOHERENCES #####
            pdf.add_page()
            pdf.set_xy(0, 10)
            pdf.set_font('arial', 'B', 30)
            st.subheader("Analyse des incohérences")
            pdf.cell(0, 0, "Analyse des incohérences", align='C')
                ##### DUP #####
            st.markdown("**Données dupliquées**")
            pdf.set_xy(0, 20)
            pdf.set_font('arial', 'B', 20)
            pdf.cell(0, 0, "Données dupliquées", align='C')
            
            if big_df == False:
                lignes_dup, hm = pf.doublons(state.dataset_clean, big_df=big_df)
                st.write("Nombre de lignes dupliquées: "+str(lignes_dup))
                st.pyplot(hm)
                pdf.set_xy(0, 40)
                pdf.set_font('arial', '', 12)
                pdf.cell("0, 0,Nombre de lignes dupliquées: "+str(lignes_dup), align='C')
                pdf.set_xy(0, 50)
                pdf.image('tempDir/tempProfiling/doublons1.png', x=25, w = 150, h = 150)
                
            else:
                lignes_dup = pf.doublons(state.dataset_clean, big_df=big_df)
                st.write("Nombre de lignes dupliquées: "+str(lignes_dup))
                pdf.set_xy(0, 40)
                pdf.set_font('arial', '', 12)
                pdf.cell(0, 0,"Nombre de lignes dupliquées: "+str(lignes_dup), align='C')
                
            
                ##### MISSING #####
            pdf.add_page()
            pdf.set_xy(0, 10)
            pdf.set_font('arial', 'B', 30)
            pdf.cell(0, 0, "Analyse des incohérences", align='C')
            pdf.set_xy(0, 20)
            pdf.set_font('arial', 'B', 20)
            pdf.cell(0, 0, "Données manquantes", align='C')
            
            st.markdown("**Données manquantes**")
            histo, camembert = pf.missing(state.dataset_clean, state.DF_META)
            st.pyplot(histo)
            st.pyplot(camembert) 
            
            pdf.set_xy(0, 30)
            pdf.image('tempDir/tempProfiling/missing1.png', x=40, w = 150, h = 100)
            pdf.image('tempDir/tempProfiling/missing2.png', x=50, w = 0, h = 60)           
    
                    ##### HORS LIMITE #####
            pdf.add_page()
            pdf.set_xy(0, 10)
            pdf.set_font('arial', 'B', 30)
            pdf.cell(0, 0, "Analyse des incohérences", align='C')
            pdf.set_xy(0, 20)
            pdf.set_font('arial', 'B', 20)
            pdf.cell(0, 0, "Données hors limites", align='C')
            
            st.markdown("**Données hors limites**")
            if 'NUMBER' in state.DF_META['type'].tolist():
                histo_hl, camembert_hl = pf.hors_limite(state.dataset_clean, state.DF_META)
                st.pyplot(histo_hl)
                st.pyplot(camembert_hl)
                
                pdf.set_xy(0, 30)
                pdf.image('tempDir/tempProfiling/hl1.png', x=30, w = 150, h = 100)
                pdf.image('tempDir/tempProfiling/hl2.png', x=20, w = 0, h = 60)  
            else:
                st.write("Aucune donnée numérique dans le dataset")
                pdf.set_font('arial', 'B', 12)
                pdf.set_xy(0, 40)
                pdf.cell(0, 0, "Aucune données numériques dans le dataset", align='C')
            
                                ##### INCOMPATIBLES #####
            pdf.add_page()
            pdf.set_xy(0, 10)
            pdf.set_font('arial', 'B', 30)
            pdf.cell(0, 0, "Analyse des incohérences", align='C')
            pdf.set_xy(0, 20)
            pdf.set_font('arial', 'B', 20)
            pdf.cell(0, 0, "Données incompatibles", align='C')
            pdf.set_font('arial', 'B', 12)
            
            st.markdown("**Données incompatibles**")
            if state.DF_THEN.empty == False:
                regle_def = True
                df_incomp = pf.incompatible(state.dataset_clean, state.DF_THEN, state.DF_IF)
                plotly_height = 50*len(df_incomp)+40
                if plotly_height > 500:
                    plotly_height = 500
                fig_incomp = go.Figure(data=[go.Table(header=dict(values=list(df_incomp.columns),
                                                                 fill_color='lightgrey', align='center', height=40), 
                                                     cells=dict(values=[df_incomp[col] for col in df_incomp],
                                                                fill_color='white',align='center', height=50))])
                fig_incomp.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=plotly_height)
                fig_height = 50*len(df_incomp)+40
                fig_incomp.write_image("tempDir/tempProfiling/incomp1.png", engine='orca', height=fig_height)
                st.write("Tableau des enregistrements violants les contraintes")
                st.write(fig_incomp)
                
            else:
                regle_def = False
                st.write("Aucune règle n'a été définie !")
                

            if (all(element == None for element in state.DF_META['exclude_values'].to_list())) or (state.DF_META.empty == True):
                pattern_def = False
                st.write("Aucun pattern à exclure n'a été défini !")
            else:
                pattern_def = True
                df_excluded = pf.excluded(state.dataset_clean, state.DF_META)
                plotly_height = 50*len(df_excluded)+40
                if plotly_height > 500:
                    plotly_height = 500
                fig_excluded = go.Figure(data=[go.Table(header=dict(values=list(df_excluded.columns),
                                                                 fill_color='lightgrey', align='center', height=40), 
                                                     cells=dict(values=[df_excluded[col] for col in df_excluded],
                                                                fill_color='white',align='center', height=50))])
                fig_excluded.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=plotly_height)            
                fig_height = 50*len(df_excluded)+40
                fig_excluded.write_image("tempDir/tempProfiling/incomp2.png", engine='orca', height=fig_height)
                st.write("Tableau des données violants les pattern exclus")
                st.write(fig_excluded)

            if (regle_def == True) & (pattern_def == True):
                pdf.set_xy(0, 40)
                pdf.image("tempDir/tempProfiling/incomp1.png", x=15, w = 180, h = 26)
                pdf.image("tempDir/tempProfiling/incomp2.png", x=15, w = 180, h = 26)
                
            elif (regle_def == True) & (pattern_def == False):
                pdf.set_xy(0, 40)
                pdf.cell(0, 0, "Aucun pattern à exclure n'a été défini !", align='C')
                pdf.set_xy(0, 50)
                pdf.image("tempDir/tempProfiling/incomp1.png", x=15, w = 180, h = 26)
                
            elif (regle_def == False) & (pattern_def == True):
                pdf.set_xy(0, 40)
                pdf.cell(0, 0, "Aucune règle n'a été définie !", align='C')
                pdf.set_xy(0, 50)                
                pdf.image("tempDir/tempProfiling/incomp2.png", x=15, w = 180, h = 26)

            elif (regle_def == False) & (pattern_def == False):
                pdf.set_xy(0, 40)
                pdf.cell(0, 0, "Aucune règle n'a été définie !", align='C')
                pdf.set_xy(0, 50)                
                pdf.cell(0, 0, "Aucun pattern à exclure n'a été défini !", align='C')
                        
                                ##### ERREURS TEMPO #####            
            pdf.add_page()
            pdf.set_xy(0, 10)
            pdf.set_font('arial', 'B', 30)
            pdf.cell(0, 0, "Analyse des incohérences", align='C')
            pdf.set_xy(0, 20)
            pdf.set_font('arial', 'B', 20)
            pdf.cell(0, 0, "Erreurs de séquence temporelle", align='C')
                
            st.markdown("**Erreurs de séquence temporelle**")
            if state.DF_TEMP.empty == False:
                df_tempo = pf.erreur_tempo(state.dataset_clean, state.DF_TEMP)
                plotly_height = 50*len(df_tempo)+40
                if plotly_height > 500:
                    plotly_height = 500
                fig_tempo = go.Figure(data=[go.Table(header=dict(values=list(df_tempo.columns),
                                                 fill_color='lightgrey', align='center', height=40), 
                                     cells=dict(values=[df_tempo[col] for col in df_tempo],
                                                fill_color='white',align='center', height=50))])
                fig_tempo.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=plotly_height)            
                fig_height = 50*len(df_tempo)+40
                fig_tempo.write_image("tempDir/tempProfiling/temp1.png", engine='orca', height=fig_height)                
                st.write("Tableau des données violants les contraintes temporelles")
                st.write(fig_tempo)
                
                pdf.set_xy(0, 30)
                pdf.image("tempDir/tempProfiling/temp1.png", x=30, w = 150, h = 19)
            else:
                st.write("Aucune règle temporelle n'a été définie !")
                pdf.set_xy(0, 50)
                pdf.set_font('arial', 'B', 12)
                pdf.cell(0, 0, "Aucune règle temporelle n'a été définie !", align='C')

            ##### ANALYSE VARIABLES #####
            st.subheader("Analyse des variables")
            for col in state.dataset_clean.columns:
                pdf.add_page()
                pdf.set_xy(0, 10)
                pdf.set_font('arial', 'B', 30)
                pdf.cell(0, 0, "Analyse des variables", align='C')
                pdf.set_xy(0, 20)
                pdf.set_font('arial', 'B', 20)
                pdf.cell(0, 0, "Analyse de la variable : "+str(col), align='C')
                pdf.set_font('arial', '', 12)
                pdf.set_xy(0, 40)
                
                data_type = gf.return_type(state.dataset_clean, col)
                df_copy = state.dataset_clean.loc[:,[col]]
                
                if len(df_copy.dropna())== 0:
                    st.markdown("**"+col+" : MISSING**")
                    st.write("Cette variable ne possède que des valeurs manquantes !")
                    pdf.set_font('arial', 'B', 12)
                    pdf.cell(0, 0, "Cette variable ne possède que des valeurs manquantes !", align='C')
                    continue
                    
                elif data_type == 'object':
                    st.markdown("**"+col+" : STRING**")
                    nb_unique, fig0, fig1 = pf.var_object(df_copy, col)
                    fig2, fig3 = pf.var_object_pattern(df_copy, col)
                    st.write("Nombre de valeurs uniques : "+str(nb_unique))
                    col0, col1 = st.beta_columns(2)
                    with col0:
                        st.pyplot(fig0)     
                    with col1:
                        st.pyplot(fig1)
                    col2, col3 = st.beta_columns(2)
                    with col2:
                        st.pyplot(fig2)
                    with col3:
                        st.pyplot(fig3)
                        
                    pdf.cell(0, 0, "Nombre de valeurs uniques : "+str(nb_unique), align='C')
                    pdf.set_xy(0, 50)
                    pdf.image("tempDir/tempProfiling/"+col+"_head.png", x=20, w = 80, h = 80)
                    pdf.image("tempDir/tempProfiling/"+col+"_pattern_head.png", x=20, w = 80, h = 80)
                    pdf.set_xy(0, 50)
                    pdf.image("tempDir/tempProfiling/"+col+"_tail.png", x=110, w = 80, h = 80)                  
                    pdf.image("tempDir/tempProfiling/"+col+"_pattern_tail.png", x=110, w = 80, h = 80)
                
                elif data_type == 'datetime64[ns]':
                    st.markdown("**"+col+" : DATE**")
                    nb_unique, min_val, max_val, fig0, fig1 = pf.var_date(df_copy, col)
                    col0, col1, col2 = st.beta_columns([1,2,2])
                    with col0:
                        st.write("Nombre de valeurs uniques : "+str(nb_unique))
                        st.write("Plus ancienne date : "+str(min_val))
                        st.write("Plus récente date : "+str(max_val))
                    with col1:
                        st.pyplot(fig0)
                    with col2:
                        st.pyplot(fig1)
                    
                    pdf.cell(0, 0, "Nombre de valeurs uniques : "+str(nb_unique), align='C')
                    pdf.set_xy(0, 50)
                    pdf.cell(0, 0, "Plus ancienne date : "+str(min_val), align='C')
                    pdf.set_xy(0, 60)
                    pdf.cell(0, 0, "Plus récente date : "+str(max_val), align='C')
                    pdf.set_xy(0, 70)
                    pdf.image("tempDir/tempProfiling/"+col+"_head.png", x=60, w = 80, h = 80)
                    pdf.image("tempDir/tempProfiling/"+col+"_tail.png", x=60, w = 80, h = 80)
                
                else :
                    st.markdown("**"+col+" : NUMBER**")
                    nb_unique, min_val, max_val, mean_val, median_val, nb_zero, fig0, fig1 = pf.var_number(df_copy, col)
                    col0, col1, col2 = st.beta_columns([1,2,2])
                    with col0:
                        st.write("Nombre de valeurs uniques : "+str(nb_unique))
                        st.write("Valeur minimale : "+str(min_val))
                        st.write("Valeur maximale : "+str(max_val))
                        st.write("Valeur moyenne : "+str(mean_val))
                        st.write("Valeur médiane : "+str(median_val))
                        st.write("Nombre de zéros : "+str(nb_zero))                        
                    with col1:
                        st.pyplot(fig0)
                    with col2:
                        st.pyplot(fig1)
                        
                    pdf.cell(0, 0, "Nombre de valeurs uniques : "+str(nb_unique), align='C')
                    pdf.set_xy(0, 50)
                    pdf.cell(0, 0, "Valeur minimale : "+str(min_val), align='C')
                    pdf.set_xy(0, 60)
                    pdf.cell(0, 0, "Valeur maximale : "+str(max_val), align='C')
                    pdf.set_xy(0, 70)
                    pdf.cell(0, 0, "Valeur moyenne : "+str(mean_val), align='C')
                    pdf.set_xy(0, 80)
                    pdf.cell(0, 0, "Valeur médiane : "+str(median_val), align='C')
                    pdf.set_xy(0, 90)
                    pdf.cell(0, 0, "Nombre de zéros : "+str(nb_zero), align='C')
                    pdf.set_xy(0, 100)
                    pdf.image("tempDir/tempProfiling/"+col+"_head.png", x=60, w = 80, h = 80)
                    pdf.image("tempDir/tempProfiling/"+col+"_tail.png", x=60, w = 80, h = 80)

        
        ### ECRITURE PDF ###
        pdf.output("output/Profiling.pdf", 'F')
        plt.close('all') #Close all pyplot figures
        st.markdown(gf.get_binary_file_downloader_html('output/Profiling.pdf', 'au format PDF'), unsafe_allow_html=True)
            
    if state.display_df == True:
        gf.display_data(state.dataset_clean)

    if state.reorder_cols == True:
        state.dataset_clean, state.reorder_var, state.reorder_how = gf.reorder_columns(state.dataset_clean, state.reorder_var, state.reorder_how)
        
    if state.change_value == True:
        state.dataset_clean = gf.change_cell_value(state.dataset_clean)
        
    if state.display_state_df == True:
        state.dataset_clean, state.DF_STATE = gf.display_df_states(state.dataset_clean, state.DF_STATE)