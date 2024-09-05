# Libraries 
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import folium
import re
import streamlit as st
from PIL import Image
from streamlit_folium import folium_static

st.set_page_config(page_title='Visão Empresa', page_icon='📈', layout='wide')

# -------------------------------------------
# Functions
# -------------------------------------------

def country_map(df):
    df_aux = (
              df.loc[:, ['City', 'Road_traffic_density', 'Delivery_location_latitude', 'Delivery_location_longitude']]
                .groupby(['City', 'Road_traffic_density'])
                .median()
                .reset_index()
              )

    map = folium.Map()

    for index, location_info in df_aux.iterrows():
        folium.Marker([location_info['Delivery_location_latitude'], 
                       location_info['Delivery_location_longitude']],
                    popup=location_info[['City', 'Road_traffic_density']]).add_to(map)

    folium_static(map, width=1024, height=600)
    return None

def order_share_by_week(df):
    delivery_per_week_df = df.loc[:, ['ID', 'week_of_year']].groupby('week_of_year').count().reset_index()
    people_per_week_df = (
                          df.loc[:, ['Delivery_person_ID', 'week_of_year']]
                            .groupby('week_of_year')
                            .nunique().reset_index()
                          )

    df_1 = pd.merge(delivery_per_week_df, people_per_week_df, how='inner')
    df_1['order_by_delivery'] = df_1['ID'] / df_1['Delivery_person_ID']

    fig = px.line(df_1, x = 'week_of_year', y = 'order_by_delivery')
    return fig


def order_by_week(df):
    # Create new column "week_of_year"
    df["week_of_year"] = df["Order_Date"].dt.strftime('%U')

    # Slice DF
    orders_by_week_df = df.groupby("week_of_year")["ID"].count().reset_index()

    # Display plot
    fig = px.line(orders_by_week_df, x="week_of_year", y="ID")
    return fig


def traffic_order_city(df):
    delivery_vol_df = df.groupby(["City", "Road_traffic_density"])["ID"].count().reset_index()

    fig = px.scatter(delivery_vol_df, x = "City", y = "Road_traffic_density", size = "ID", color="City")
    return fig


def traffic_order_share(df):
    # Slicing DF
    traffic_df = df.groupby("Road_traffic_density")["ID"].count().reset_index()

    # Creating new column "deliveries_perc"
    traffic_df["deliveries_perc"] = traffic_df["ID"] / traffic_df["ID"].sum()

    # Plot graph
    fig = px.pie(traffic_df, values = "deliveries_perc", names = "Road_traffic_density")
    return fig


def order_metric(df):
    """
    
    """
    # Columns
    cols = ['ID', 'Order_Date']

    # Line selection
    df_aux = df.loc[:, cols].groupby('Order_Date').count().reset_index()

    # Plot graph
    fig = px.bar(df_aux, x="Order_Date", y="ID")
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


    
# Visao - Empresa
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

tab1, tab2, tab3 = st.tabs(['Visão Gerencial', 'Visão Tática', 'Visão Geográfica'])

with tab1:
    with st.container():
        # Order Metric
        fig = order_metric(df)
        st.markdown('# Orders by Day')
        st.plotly_chart(fig, use_container_width = True)            


    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            fig = traffic_order_share(df)
            st.markdown('# Traffic Order Share')
            st.plotly_chart(fig, use_container_width = True)

            
        with col2:
            st.markdown('# Traffic Order City')
            fig = traffic_order_city(df)
            st.plotly_chart(fig, use_container_width = True)


with tab2:
    with st.container():
        st.markdown('# Order by Week')
        st.plotly_chart(fig, use_container_width = True)
        fig = order_by_week(df)
        
    with st.container():
        st.markdown('# Order Share by Week')
        fig = order_share_by_week(df)
        st.plotly_chart(fig, use_container_width = True)
        
        
with tab3:
    st.markdown('# Country Maps')
    country_map(df)
