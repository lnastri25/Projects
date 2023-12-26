import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



## FUNCTION TO PLOT OUTLIERS ##

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



## FUNCTION TO DETERMINE OUTLIERS ##

def outliers_recogntion(df, column, scale_factor=1.5):
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



## FUNCTION TO REMOVE OUTLIERS ##

def remove_outliers(df, column, outliers):
    if column in df.columns:
        df = df.drop(df[outliers].index)
    
    return df



## FUNCTION TO CREATE HEATMAP OF CORRELATION BETWEEN NUMERIC VARIABLES ##

def crear_heatmap_correlacion(df):
    def seleccionar_columnas_numericas(df):
        return df.select_dtypes(include=['float64', 'int64'])
    df_numeric = seleccionar_columnas_numericas(df)
    df_numeric = df_numeric.drop(columns=[col for col in df_numeric.columns if col.startswith("Unnamed:") or col.strip() == ""])
    corr_matrix = df_numeric.corr()

    plt.figure(figsize=(10, 10))
    sns.heatmap(
        round(corr_matrix, 2),
        cmap='coolwarm',
        annot=True,
        annot_kws={"size": 10}
    )
    plt.show()