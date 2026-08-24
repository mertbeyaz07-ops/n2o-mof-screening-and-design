# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 07:18:40 2022

@author: User
"""

from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Try specifying the separator if it's not a standard comma
df = pd.read_csv(r'C:/Users/mertb/OneDrive/Masaüstü/FINAL DATA/pm/core_1_5_arc/coremof_1_10_arcmof_hybrid.csv', sep=None, engine='python')

# Debug: Check how many columns were actually found
print(f"Columns found: {df.columns.tolist()}")

data = df.values
X, y = data[:, :-1], data[:, -1]
df.data = X
df.target = y

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(df.data, df.target,
                                                    train_size=0.80, test_size=0.20, random_state=42)


tpot = TPOTRegressor(generations=10, population_size=50,
                     verbosity=2, random_state=42, n_jobs=-1, cv=5)


...
# perform the search
tpot.fit(X_train, y_train)
# export the best model
tpot.export('TPOT_Pipeline-Gen10-Pop50-Random42-Cv5-MOF-_CoREMOF_Arc_1_10_N2O_1bar_80_20_PM_added.py')

extracted_best_model = tpot.fitted_pipeline_.steps[-1][1]
extracted_best_model.fit(X_train, y_train)
print(extracted_best_model.feature_importances_)

feature_importances = extracted_best_model.feature_importances_
feature_names = df.columns[:-1]

feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

print(feature_importance_df)