# Libraries 
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import folium
import re
import streamlit as st
from PIL import Image
from streamlit_folium import folium_static
from haversine import haversine
import numpy as np
import plotly.graph_objs as go

st.set_page_config(page_title='Visão Restaurantes', page_icon='🥡', layout='wide')

# -------------------------------------------
# Functions
# -------------------------------------------

def avg_std_time_on_traffic(df):
    df_aux = (
              df.loc[:, ['City', 'Road_traffic_density', 'Time_taken(min)']]
              .groupby(['City', 'Road_traffic_density'])
              .agg({'Time_taken(min)': ['mean', 'std']})
              )

    df_aux.columns = ['avg_time', 'std_time']

    df_aux = df_aux.reset_index()

    fig = px.sunburst(df_aux, path=['City', 'Road_traffic_density'], values='avg_time',
                     color='std_time', color_continuous_scale='RdBu',
                     color_continuous_midpoint=np.average(df_aux['std_time']))
    return fig


def avg_std_time_graph(df):
    df_aux = (
              df.loc[:, ['City', 'Time_taken(min)']]
                .groupby('City')
                .agg({'Time_taken(min)': ['median', 'std']})
              )

    df_aux.columns = ['avg_time', 'std_time']

    df_aux = df_aux.reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Control',
                         x=df_aux['City'],
                         y=df_aux['avg_time'],
                         error_y=dict(type='data', array=df_aux['std_time'])))
    fig.update_layout(barmode='group')
    return fig
            

def avg_std_time_delivery(df, festival, op):
    """
        This fcuntion calculates the mean time and the standard deviation of the delivery time
        Parameters:
            Input:
                - df: DataFrame with the needed info for calculation
                - op: Type of operation needed for the calculus
                    'avg_time': calculates the mean time
                    'std_time': Calculates the standard deviation of time
            Output:
                - df: DataFrame with 2 columns and one line

    """
    df_aux = (
            df.loc[:, ['Festival', 'Time_taken(min)']]
                .groupby(['Festival'])
                .agg({'Time_taken(min)': ['median', 'std']})
            )

    df_aux.columns = ['avg_time', 'std_time']
    df_aux = df_aux.reset_index()

    df_aux = np.round(df_aux.loc[df_aux['Festival'] == festival, op], 2)
    return df_aux 


def distance(df, fig):
    if fig == False:
        cols = ['Delivery_location_latitude', 'Delivery_location_longitude', 'Restaurant_latitude', 'Restaurant_longitude']
        df['distance'] = (
                        df.loc[:, cols]
                        .apply(lambda x: haversine( (x['Restaurant_latitude'], 
                                                     x['Restaurant_longitude']), 
                                                   (x['Delivery_location_latitude'], 
                                                    x['Delivery_location_longitude'])), axis=1)
                        )
        avg_distance = np.round(df['distance'].mean(), 2)
        return avg_distance
    else:
        cols = ['Delivery_location_latitude', 'Delivery_location_longitude', 'Restaurant_latitude', 'Restaurant_longitude']
        df['distance'] = (
                        df.loc[:, cols]
                        .apply(lambda x: haversine( (x['Restaurant_latitude'], 
                                                     x['Restaurant_longitude']), 
                                                   (x['Delivery_location_latitude'], 
                                                    x['Delivery_location_longitude'])), axis=1)
                        )
        avg_distance = df.loc[:, ['City', 'distance']].groupby('City').mean().reset_index()
        fig = go.Figure(data=[go.Pie(labels=avg_distance['City'], values=avg_distance['distance'], pull=[0, 0.1, 0])])
        return fig

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

# Visao - Restaurantes
# ======================================
# Side Bar
# ======================================


st.header('Marketplace - Visão Restaurantes')

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
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            amount_unique_delivery_people = len(df.loc[:, 'Delivery_person_ID'].unique())
            col1.metric('Entregadores', amount_unique_delivery_people)

        with col2:
            avg_distance = distance(df, fig=False)
            col2.metric('A distância média', avg_distance)
            
        with col3:
            df_aux = avg_std_time_delivery(df,'Yes' , 'avg_time')
            col3.metric('Tempo Médio', df_aux)
            
        with col4:
            df_aux = avg_std_time_delivery(df,'Yes', 'std_time')
            col4.metric('STD de Entrega', df_aux)
            
        with col5:
            df_aux = avg_std_time_delivery(df,'No', 'avg_time')
            col5.metric('Tempo Médio', df_aux)
            
        with col6:
            df_aux = avg_std_time_delivery(df,'No', 'std_time')
            col6.metric('STD de Entrega', df_aux)
            

        
    with st.container():
        st.markdown("""___""")
        col1, col2 = st.columns(2)
        
        with col1:        
            fig = avg_std_time_graph(df)
            st.plotly_chart(fig)
            
        with col2:
            df_aux = (
                  df.loc[:, ['City', 'Type_of_order', 'Time_taken(min)']]
                    .groupby(['City', 'Type_of_order'])
                    .agg({'Time_taken(min)': ['median', 'std']}) 
                 )

            df_aux.columns = ['avg_time', 'std_time']

            df_aux = df_aux.reset_index()

            st.dataframe(df_aux)
                
    with st.container():
        st.markdown("""___""")
        st.title('Distribuição do Tempo')
        
        col1, col2 = st.columns(2)
        with col1:
            fig = distance(df, fig=True)
            st.plotly_chart(fig)
            
        with col2:
            fig = avg_std_time_on_traffic(df)
            st.plotly_chart(fig)
            

        