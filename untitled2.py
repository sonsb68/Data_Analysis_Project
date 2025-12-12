# -*- coding: utf-8 -*-
"""
Analyse exploratoire et nettoyage du dataset Housing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# Chargement des données
# ============================
df = pd.read_csv(r"C:\Users\33777\Downloads\housing.csv")

# Supprimer index inutile
df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)

# ============================
# Gestion des valeurs "None"
# ============================
none_cols = ['Alley', 'Fireplace_Qu', 'Pool_QC', 'Fence', 'Misc_Feature',
             'Garage_Type', 'Garage_Qual', 'Garage_Cond', 'Garage_Finish',
             'Bsmt_Qual', 'Bsmt_Cond', 'Bsmt_Exposure', 'BsmtFin_Type_1',
             'BsmtFin_Type_2', 'Mas_Vnr_Type']
for col in none_cols:
    if col in df.columns:
        df[col] = df[col].replace('None', np.nan)

# ============================
# Conversion des colonnes ordinales en numérique
# ============================
qual_ordre = ['Very_Poor', 'Poor', 'Fair', 'Typical', 'Good', 'Very_Good', 'Excellent', 'Very_Excellent']
ordinal_cols = ['Overall_Qual', 'Overall_Cond', 'Exter_Qual', 'Exter_Cond', 'Bsmt_Qual', 
                'Bsmt_Cond', 'Heating_QC', 'Kitchen_Qual', 'Fireplace_Qu', 'Garage_Qual', 'Garage_Cond']
for col in ordinal_cols:
    if col in df.columns:
        df[col] = pd.Categorical(df[col], categories=qual_ordre, ordered=True).codes

# ============================
# Suppression des outliers
# ============================
def remove_outliers(df, cols):
    for col in cols:
        if np.issubdtype(df[col].dtype, np.number):
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
    return df

quant_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df = remove_outliers(df, quant_cols)

# ============================
# Statistiques descriptives
# ============================
print(df.describe(include='all'))

# ============================
# Visualisations
# ============================
# Quantitatives
for col in quant_cols:
    if col in df.columns:
        sns.histplot(df[col], kde=True)
        plt.title(col)
        plt.show()

# Qualitatives
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    sns.countplot(y=col, data=df, order=df[col].value_counts().index)
    plt.title(col)
    plt.show()

# Corrélation avec Sale_Price
if 'Sale_Price' in df.columns:
    sns.heatmap(df.corr(), cmap='coolwarm')
    plt.title("Corrélation")
    plt.show()
