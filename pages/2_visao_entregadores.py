# Libraries 
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import folium
import re
import streamlit as st
from PIL import Image
from streamlit_folium import folium_static

st.set_page_config(page_title='Visão Entregadores', page_icon='🚐', layout='wide')

# -------------------------------------------
# Functions
# -------------------------------------------

def top_delivers(df, top_asc):
    df_slow_delivery = (
                    df.loc[:, ['Delivery_person_ID', 'City', 'Time_taken(min)']]
                    .groupby(['City', 'Delivery_person_ID'])
                    .max()
                    .sort_values(['City', 'Time_taken(min)'], ascending=top_asc)
                    .reset_index()
                    )

    df_aux1 = df_slow_delivery.loc[df_slow_delivery['City'] == 'Metropolitian', :].head(10)
    df_aux2 = df_slow_delivery.loc[df_slow_delivery['City'] == 'Urban', :].head(10)
    df_aux3 = df_slow_delivery.loc[df_slow_delivery['City'] == 'Semi-Urban', :].head(10)

    df_slow_delivery_final = pd.concat([df_aux1, df_aux2, df_aux3]).reset_index(drop=True)

    return(df_slow_delivery_final)


def clean_code(df):
    """ This function cleans the DataFrame
        
        Types of cleaning:
        1. Removal of all NaM
        2. Chage of the data type on some columns
        3. Removal of all empty spaces from the strings
        4. Change on the date format on a column
        5. Cleaning on the time column ( removal of any text from the numeric variable)
        
        Iput: DataFrame
        Output: DataFrame
    """
    
    # Removing empty spaces from strings
    df.loc[:, "ID"] = df.loc[:, "ID"].str.strip()
    df.loc[:, "Delivery_person_ID"] = df.loc[:, "Delivery_person_ID"].str.strip()
    df.loc[:, "Delivery_person_Age"] = df.loc[:, "Delivery_person_Age"].str.strip()
    df.loc[:, "Road_traffic_density"] = df.loc[:, "Road_traffic_density"].str.strip()
    df.loc[:, "Type_of_order"] = df.loc[:, "Type_of_order"].str.strip()
    df.loc[:, "Type_of_vehicle"] = df.loc[:, "Type_of_vehicle"].str.strip()
    df.loc[:, "multiple_deliveries"] = df.loc[:, "multiple_deliveries"].str.strip()
    df.loc[:, "Festival"] = df.loc[:, "Festival"].str.strip()
    df.loc[:, "City"] = df.loc[:, "City"].str.strip()

    # Delete all rows that contain NaN from the "Delivery_person_Age" column
    age_empty_lines = df['Delivery_person_Age'] != "NaN"
    df = df.loc[age_empty_lines, :]

    # Delete all rows that contain NaN from the "Road_traffic_density" column
    density_empty_lines = df["Road_traffic_density"] != "NaN"
    df = df.loc[density_empty_lines, :]

    # Delete all rows that contain NaN from the "City" column
    city_empty_lines = df["City"] != "NaN"
    df = df.loc[city_empty_lines, :]

    # Delete all rows that contain NaN from the "Festival" column
    festival_empty_lines = df["Festival"] != "NaN"
    df = df.loc[festival_empty_lines, :]
    
    # Delete all rows from "multiple_deliveries" column that contains NaN
    delivery_empty_lines = df["multiple_deliveries"] != 'NaN'

    # Transforming Delivery_person_Age from object to int
    df["Delivery_person_Age"] = df["Delivery_person_Age"].astype(int)

    # Transforming Delivery_person_Ratings from object to float
    df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].astype(float)

    # Transforming the Order_Date from Object to date format = '%d-%m-%Y'
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], format='%d-%m-%Y')

    # Transforming all data under the "multiple_deliveries" column into int
    df = df.loc[delivery_empty_lines, :]
    df["multiple_deliveries"] = df["multiple_deliveries"].astype(int)

    # Reseting the DataFrame index
    df = df.reset_index(drop = True)

    # Removing all text from numbers
    df['Time_taken(min)'] = df['Time_taken(min)'].apply(lambda x: x.split('(min) ')[1])
    df['Time_taken(min)'] = df['Time_taken(min)'].astype(int)  
    
    return df

# ------------------------------------------- Logic Structure of the Code ------------------------------
# -------------------------
# Import dataset
# -------------------------
df_raw = pd.read_csv('dataset/train.csv')

# Create a copy of the DataFrame
df = df_raw.copy()

# -------------------------
# Cleaning data
# -------------------------
df = clean_code(df)

    
# Visao - Entregadores
# ======================================
# Side Bar
# ======================================

st.header('Marketplace - Visão Entregadores')

# image_path = '/Users/brcwb/repos/ftc_programacao_python/logo.png'
image = Image.open('logo.png')
st.sidebar.image(image, width=120)

st.sidebar.markdown('# Cury Company')
st.sidebar.markdown('## Fastest Delivery in Town')
st.sidebar.markdown("""___""")

st.sidebar.markdown('## Selecione uma data limite')

date_slider = st.sidebar.slider(
    'Até qual valor?',
    value=pd.datetime(2022, 4, 13),
    min_value=pd.datetime(2022, 2, 11),
    max_value = pd.datetime(2022, 4, 6),
    format='DD-MM-YY')

st.sidebar.markdown("""___""")

traffic_options = st.sidebar.multiselect(
    'Quais as condições do trânsito',
    ['Low', 'Medium', 'High', 'Jam'],
    default = ['Low', 'Medium', 'High', 'Jam'])

st.sidebar.markdown("""___""")
st.sidebar.markdown('### Powred by Comunidade DS')

# Date Filter
selected_lines = df['Order_Date'] < date_slider
df = df.loc[selected_lines, :]

# Traffic Filter
selected_lines = df['Road_traffic_density'].isin(traffic_options)
df = df.loc[selected_lines, :]

# ======================================
# Streamlit Layout
# ======================================

tab1, tab2, tab3 = st.tabs(['Visão Gerencial', '_', '_'])

with tab1:
    with st.container():
        st.title('Overall Metrics')
        col1, col2, col3, col4 = st.columns(4, gap='large')
        with col1:
            # Max Delivery person age
            max_age = df.loc[:, 'Delivery_person_Age'].max()
            col1.metric('Maior de idade', max_age)
        with col2:
            # Min Delivery person age
            min_age = df.loc[:, 'Delivery_person_Age'].min()
            col2.metric('Menor de idade', min_age)
        with col3:
            # Best vehicle condition
            max_condition = df.loc[:, 'Vehicle_condition'].max()
            col3.metric('Melhor condição', max_condition)
        with col4:
            # Worst vehicle condition
            min_condition = df.loc[:, 'Vehicle_condition'].min()
            col4.metric('Pior condição', min_condition)

    with st.container():
        st.markdown("""___""")
        st.title('Avaliações')
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('##### Avaliação média por Entregador')
            df_avg_rating_per_deliveryperson = (
                                                df.loc[:, ['Delivery_person_ID', 'Delivery_person_Ratings']]
                                                .groupby('Delivery_person_ID')
                                                .mean()
                                                .reset_index()
                                                )
            
            st.dataframe(df_avg_rating_per_deliveryperson)
      
        with col2:
            st.markdown('##### Avaliação média por Trânsito')
            df_avg_rating_by_traffic = (
                                        df.loc[: , ['Delivery_person_Ratings', 'Road_traffic_density']]
                                        .groupby('Road_traffic_density')
                                        .agg({'Delivery_person_Ratings': ['mean', 'std']})
                                       )
            
            # Column name change
            df_avg_rating_by_traffic.columns = ['delivery_mean', 'delivery_std']

            st.dataframe(df_avg_rating_by_traffic.reset_index())
            
            
            st.markdown('##### Avaliação média por Clima')
            df_mean_std = (
                            df.loc[:, ['Delivery_person_Ratings', 'Weatherconditions']]
                            .groupby('Weatherconditions')
                            .agg({'Delivery_person_Ratings': ['mean', 'std']})
                            )
            # Column name change
            df_mean_std.columns = ['delivery_rating_mean', 'delivery_rating_std']

            st.dataframe(df_mean_std.reset_index())
            
    with st.container():
        st.markdown("""___""")
        st.title('Velocidade de Entrega')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('##### Top Entregadores mais rápidos')
            df = top_delivers(df, top_asc=True)
            st.dataframe(df)

            
        with col2:
            st.markdown('##### Top Entregadores mais lentos')
            df = top_delivers(df, top_asc=False)
            st.dataframe(df)