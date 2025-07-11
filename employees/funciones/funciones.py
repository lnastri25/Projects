import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


## 1) FUNCIÓN PARA CARGAR EL DATAFRAME ##

def load_df(path):
    df = pd.read_csv(path)
    return df


## 2) GENERALES ##

def whitespace_remover_and_columns(dataframe):
    for i in dataframe.columns:
        if dataframe[i].dtype == 'object':
            dataframe[i] = dataframe[i].map(str.strip)
        else:
            pass
    dataframe.rename(columns=lambda x: x.strip(), inplace=True)
    return dataframe


## 3) FUNCIÓN PARA VALIDAR COLUMNAS ##

def validate_columns(df):
    summary_df = df.describe(include='all').T
    summary_df['Num_Null_Values'] = df.isnull().sum()
    summary_df['%_Null_Values'] = (summary_df['Num_Null_Values'] / len(df)) * 100
    sample_unique_values = df.sample(min(5, len(df)), axis=0).T
    summary_df['Sample_Unique_Values'] = sample_unique_values.values.tolist()
    summary_df = summary_df.rename(columns={'unique': 'Unique_Values', 'count': 'Num_Unique_Values'})
    summary_df = summary_df[['Unique_Values', 'Num_Unique_Values', 'Num_Null_Values', '%_Null_Values', 'Sample_Unique_Values']]
    return summary_df


## 4) FUNCIONES PARA MANEJO DE FALTANTES ##

def fill_missing_values(df, columns, fill_method, fill_value=None):
    for column in columns:
        if fill_method == 'bfill':
            df[column].fillna(method='bfill', inplace=True)
        elif fill_method == 'fillna':
            df[column].fillna(fill_value, inplace=True)
    return df

""""
# Ejemplo de uso:
from ... import fill_missing_values

columns_to_fill_bfill = ['Embarked', "Cabin"]
columns_to_fill_fillna = ['Age']

titanic_df = fill_missing_values(titanic_df, columns_to_fill_bfill, 'bfill')
titanic_df = fill_missing_values(titanic_df, columns_to_fill_fillna, 'fillna', fill_value=0)
"""


"""
def fill_missing_values(df, columns, fill_method, fill_value=None):
    for column in columns:
        if fill_method == 'bfill' and df[column].dtype == 'O':
            df[column].fillna(method='bfill', inplace=True)
        elif fill_method == 'fillna' and df[column].dtype in ['int64', 'float64']:
            df[column].fillna(fill_value, inplace=True)
        # Puedes agregar más casos para otros tipos de datos o métodos de relleno si es necesario
    return df

# SE PUEDE HACER ASI TAMBIEN --> ACA LE EXIGIS QUE SE CUMPLAN LAS CONDICIONES DEL APUNTE
"""


## 5) FUNCIÓN PARA GRAFICAR VALORES OUTLIERS ##

def plot_outliers(df):
    num_cols = df.select_dtypes(include=['float64', 'int64']).shape[1]
    num_rows = (num_cols - 1) // 2 + 1
    fig, axs = plt.subplots(num_rows, 2, figsize=(20, 5*num_rows))
    axs = axs.flatten()
    for i, col in enumerate(df.select_dtypes(include=['float64', 'int64']).columns):
        box = axs[i].boxplot(df[col], patch_artist=True, boxprops=dict(facecolor='#336fa2'), medianprops=dict(color='black'))
        for patch in box['fliers']:
            patch.set_markerfacecolor('black')
            patch.set_markeredgecolor('black')
        axs[i].set_title(col)
    plt.tight_layout()
    plt.show()


## 6) FUNCIÓN PARA DETERMINAR SI ¿ES OUTLIER O NO? ##

def reconocimiento_de_outliers(df, column, scale_factor=1.5):
    data = df[column]
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - scale_factor * iqr
    upper_bound = q3 + scale_factor * iqr
    
    outliers = (data < lower_bound) | (data > upper_bound)
    outlier_values = data[outliers]

    print(f"Outliers de {column}:")
    print(outlier_values)
    
    return outliers, outlier_values


""""
# Ejemplo de uso:
from ... import reconocimiento_de_outliers

outliers_admin_costs, outlier_values_admin_costs = reconocimiento_de_outliers(df_startups, "Admin_Costs", 1.5)
outliers_net_profit, outlier_values_net_profit = reconocimiento_de_outliers(df_startups, "Net_Profit", 1.5)
outliers_profit, outlier_values_profit = reconocimiento_de_outliers(df_startups, "Profit", 1.5)
"""


## 7) FUNCIÓN PARA REMOVER OUTLIERS ##
""""
def remove_outliers(df, column, outliers):
    if column in df.columns:
        df = df.drop(df[outliers].index)
    
    return df
"""""

def remove_outliers(df, column, outliers):
    if column in df.columns:
        df = df.drop(df[df[column].isin(outliers)].index)
    return df

""""
# Ejemplo de uso:
from ... import remove_outliers

df_startups = remove_outliers(df_startups, "Admin_Costs", outliers_admin_costs)
df_startups = remove_outliers(df_startups, "Net_Profit", outliers_net_profit)
df_startups = remove_outliers(df_startups, "Profit", outliers_profit)
"""


## 8) FUNCIÓN PARA CREAR HEATMAP DE CORRELACIÓN ENTRE VARIABLES NUMÉRICAS ##

# A) MATRIZ COMPLETA
def crear_heatmap_correlacion(df): # --> pasarle siempre el df original, nada de filtrado.
    def seleccionar_columnas_numericas(df):
        return df.select_dtypes(include=['float64', 'int64'])

    # Seleccionar las columnas numéricas del DataFrame
    df_numeric = seleccionar_columnas_numericas(df)

    # Eliminar las columnas con títulos no deseados
    df_numeric = df_numeric.drop(columns=[col for col in df_numeric.columns if col.startswith("Unnamed:") or col.strip() == ""])

    # Calcular la matriz de correlación
    corr_matrix = df_numeric.corr()

    # Crear el heatmap de correlación
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        round(corr_matrix, 2),
        cmap='coolwarm',
        annot=True,
        annot_kws={"size": 10}
    )
    plt.show()

# B) MATRIZ POR LA MITAD
def matriz_por_la_mitad(df):
    correlation_matrix = df.corr()

    plt.figure(figsize=(20, 20))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, mask=mask, cmap='coolwarm', vmax=1, vmin=-1,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, annot_kws={"size": 8})

    plt.title('Matriz de Correlación')
    plt.show()


## 9) FUNCIÓN PARA VER DISTRIBUCIÓN ENTRE VARIABLES ##

def distribucion_entre_variables(df, categorical_variable, variables):
    num_variables = len(variables)
    plt.figure(figsize=(22, 6 * num_variables))
    
    for i, variable in enumerate(variables, 1):
        plt.subplot(num_variables, 1, i)
        sns.boxplot(x=categorical_variable, y=variable, data=df)
        plt.title(f'Distribución de {variable} por {categorical_variable}')

    plt.tight_layout()
    plt.show()
"""
df_cultivos['Tipo_de_Cultivo'] = df_cultivos[cultivos].idxmax(axis=1) --> ¡OJO acá!


# Ejemplo de uso:
otras_variables = ['Temperatura_C', 'Humedad_Relativa', 'Precipitacion_mm']
distribucion_entre_variables(nombre_df, 'Tipo_de_Cultivo', otras_variables)
"""

'''
otras_variables = ['ventas', "precio_en_mercado_libre", "Kilometraje"]

plt.figure(figsize=(22, 24))
for i, variable in enumerate(otras_variables, 1):
    plt.subplot(3, 1, i)
    sns.boxplot(x='Pais_de_Origen', y=variable, data=df_ventas_autos)
    plt.title(f'Distribución de {variable} por País de Origen')

plt.tight_layout()
plt.show()'
'''



## FUNCIÓN PARA TRANSFORMAR COLUMNAS A DATETIME ##

def transformar_columnas_datetime(df):
    for columna in df.columns:
        try:
            df[columna] = pd.to_datetime(df[columna])
        except ValueError:
            pass
    return df


'''
from sklearn import preprocessing

# Función para encodear

def encoder(df, cat):

    le = preprocessing.LabelEncoder()

    clases = []

    for i in cat:

        df[i]=le.fit_transform(df[i]) 

        clases.append(le.classes_)

    return df, clases

'''










