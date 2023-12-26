import pandas as pd
import numpy as np

### GENERALS ###

def whitespace_remover(dataframe):
    for i in dataframe.columns:
        if dataframe[i].dtype == 'object':
            dataframe[i] = dataframe[i].apply(lambda x: x.strip() if isinstance(x, str) else x)
    return dataframe


def whitespace_remover_and_columns(dataframe):
    for i in dataframe.columns:
        if dataframe[i].dtype == 'object':
            dataframe[i] = dataframe[i].map(str.strip)
        else:
            pass
    dataframe.rename(columns=lambda x: x.strip(), inplace=True)
    return dataframe


### POINT A) ###

## FUNCTION TO TRANSFORM COLUMNS TO DATETIME ## 
def transformar_columnas_datetime(df):
    for columna in df.columns:
        try:
            df[columna] = pd.to_datetime(df[columna])
        except ValueError:
            pass
    return df


# FUNCTION TO CALCULATE DELTA TIMING ## 
def calcular_time_delta(df, columna_fecha_inicio, columna_fecha_fin):
    df['time_delta'] = (df[columna_fecha_fin] - df[columna_fecha_inicio]).dt.days
    return df


## FUNCTION TO CALCULATE REAL DELIVERY/WAITING TIME ##
def tiempo_de_espera_real(orders, is_delivered=True):
    if is_delivered:
        orders = orders.query("order_status=='delivered'").copy()
    orders.loc[:, 'tiempo_de_espera'] = \
        (orders['order_delivered_customer_date'] -
         orders['order_purchase_timestamp']) / np.timedelta64(24, 'h')
    return orders


## FUNCTION TO CALCULATE EXPECTED DELIVERY TIME ##
def tiempo_de_espera_esperado(orders, is_delivered=True):
    if is_delivered:
        orders = orders.query("order_status=='delivered'").copy()
    orders.loc[:, 'tiempo_de_espera_pronosticado'] = \
        (orders['order_estimated_delivery_date'] -
         orders['order_purchase_timestamp']) / np.timedelta64(24, 'h')
    return orders


## FUNCTION TO CALCULATE DELIVERY TIME DIFFERENCE BETWEEN REAL AND EXPECTED. IF REAL DELIVERY DATE IS AFTER EXPECTED DELIVERY DATE, RETURNS NUMBER OF DAYS BETWEEN THE TWO DATES; OTHERWISE, RETURNS 0 ##

def real_vs_esperado(orders, is_delivered=True):
    if is_delivered:
        orders = orders.query("order_status=='delivered'").copy()
        
    orders['real_vs_esperado'] = np.where(orders['tiempo_de_espera'] > orders['tiempo_de_espera_pronosticado'], 
                                           orders['tiempo_de_espera'] - orders['tiempo_de_espera_pronosticado'], 
                                           0)
    return orders



### POINT B) ###

## FUNCTION TO CALCULATE PURCHASE SCORE BETWEEN 1 AND 5 ##
def review_score(df):
    def es_cinco_estrellas(x):
        if x == 5:
            return 1
        else:
            return 0 
    df['es_cinco_estrellas'] = df['review_score'].apply(es_cinco_estrellas)
    
    def es_una_estrella(x):
        if x == 1:
            return 1 
        else:
            return 0
    df['es_una_estrella'] = df['review_score'].apply(es_una_estrella)
    
    return df[['order_id','es_cinco_estrellas', 'es_una_estrella', 'review_score']]



### POINT C) ###

## FUNCTION TO CALCULATE NUMBER OF PRODUCTS PER ORDER ##
def calcular_numero_de_productos(df):
    df1 = df.copy()
    order_items_df = df1['order_items'].copy()
    order_items_df = whitespace_remover(order_items_df)
    order_items_df.rename(columns=lambda x: x.strip(), inplace=True)
    return order_items_df.groupby('order_id').agg(num_de_produc = ('product_id','count')).reset_index()



### POINT D) ###

## FUNCTION TO CALCULATE UNIQUE SELLERS ##
def vendedores_unicos(df):
    df2 = df.copy()
    oitp4 = df2['order_items'].copy()
    oitp4 = whitespace_remover(oitp4)
    oitp4.rename(columns=lambda x: x.strip(), inplace=True)
    return oitp4.groupby('order_id').agg(vendedores_unicos = ('seller_id','nunique')).reset_index()



### POINT E) ###

## FUNCTION TO CALCULATE PRICE AND TRANSPORT ##
def calcular_precio_y_transporte(df):
    df2=df.copy()
    order_items_df2 = df2['order_items'].copy()
    order_items_df2 = whitespace_remover(order_items_df2)
    order_items_df2.rename(columns=lambda x: x.strip(), inplace=True)
    return order_items_df2.groupby('order_id').agg(precio = ('price','sum'), transporte=('freight_value','sum')).reset_index()



### POINT F) ###

# FUNCTION TO CALCULATE DISTANCE BETWEEN SELLER AND CUSTOMER ##

from math import radians, sin, cos, asin, sqrt
def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Computa distancia entre dos pares (lat, lng)
    Ver - (https://en.wikipedia.org/wiki/Haversine_formula)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))

def calcular_distancia(row):
    return haversine_distance(row['geolocation_lng_x'], row['geolocation_lat_x'], row['geolocation_lng_y'], row['geolocation_lat_y'])


def crear_columna(df):
    df2=df.copy()
    df2['distance_seller_customer'] = df2.apply(calcular_distancia, axis=1)
    return df2[['order_id', 'distance_seller_customer']]