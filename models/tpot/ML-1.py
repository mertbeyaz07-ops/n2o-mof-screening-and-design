# -*- coding: utf-8 -*-
"""
Created on Thu May  7 12:45:29 2026

@author: mertb
"""

# ==========================================
# IMPORTS
# ==========================================
import numpy as np
import pandas as pd
import warnings

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import ScalarFormatter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import shap

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

warnings.filterwarnings("ignore")

# ==========================================
# SETTINGS
# ==========================================
target_col = 'N2O_uptake_1bar_molkg'

train_path = r'C:/Users/mertb/OneDrive/Masaüstü/FINAL DATA/10_50_FULL_CORE_QST0_KH/CoREMOF_N2O_single_full_kh_added.csv'

arc_path = r'C:/Users/mertb/OneDrive/Masaüstü/FINAL DATA/10_50_FULL_CORE_QST0_KH/ARCMOF_N2O_FULL_KH_added.xlsx'

qmof_path = r'C:/Users/mertb/OneDrive/Masaüstü/FINAL DATA/10_50_FULL_CORE_QST0_KH/QMOF_N2O_Merged_Dataset.xlsx'

save_path = r'D:/NEMO_Visioner_Mert_Beyaz/XGB_FULL_CORE_QST0_KH_SCIENTIFIC_SHAP_REVISED.png'

# ==========================================
# FUNCTIONS
# ==========================================
def clean_columns(df):

    df = df.copy()

    df.columns = (
        df.columns.astype(str)
        .str.replace('\ufeff', '', regex=False)
        .str.strip()
    )

    return df


def read_clean_excel(path):

    df = pd.read_excel(path)

    df = clean_columns(df)

    for col in df.columns:

        df[col] = (
            df[col]
            .astype(str)
            .str.replace(',', '.', regex=False)
            .str.replace(' ', '', regex=False)
        )

        df[col] = pd.to_numeric(
            df[col],
            errors='coerce'
        )

    df = df.dropna()

    return df


def metrics(y_true, y_pred):

    return (
        r2_score(y_true, y_pred),

        np.sqrt(
            mean_squared_error(y_true, y_pred)
        ),

        mean_absolute_error(y_true, y_pred)
    )

# ==========================================
# SCIENTIFIC FEATURE LABELS
# ==========================================
scientific_feature_names = {

    "LCD (A)": r"$\mathrm{LCD\ (Å)}$",

    "PLD (A)": r"$\mathrm{PLD\ (Å)}$",

    "Porosity": r"$\mathrm{Porosity}$",

    "Pore Volume (cm3/g)": r"$\mathrm{Pore\ Volume\ (cm^3/g)}$",

    "C_%": r"$\mathrm{C\ (\%)}$",

    "H_%": r"$\mathrm{H\ (\%)}$",

    "N_%": r"$\mathrm{N\ (\%)}$",

    "O_%": r"$\mathrm{O\ (\%)}$",

    "Halogen_%": r"$\mathrm{Halogen\ (\%)}$",

    "Metal_%": r"$\mathrm{Metal\ (\%)}$",

    "Nonmetal_%": r"$\mathrm{Nonmetal\ (\%)}$",

    "Metalloid_%": r"$\mathrm{Metalloid\ (\%)}$",

    "Qst0_kJmol": r"$\mathrm{Q_{st}^{0}\ (kJ/mol)}$",

    "KH": r"$\mathrm{K_H\ (mol/(kg\cdot Pa))}$"
}

# ==========================================
# LOAD TRAIN DATA
# ==========================================
df = pd.read_csv(
    train_path,
    sep=None,
    engine="python"
)

df = clean_columns(df)

X = df.drop(target_col, axis=1)

y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

feature_cols = list(X_train.columns)

# ==========================================
# MODEL
# ==========================================
model = XGBRegressor(
    learning_rate=0.1,
    max_depth=7,
    min_child_weight=5,
    n_estimators=100,
    subsample=0.45,
    objective="reg:squarederror",
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# TRAIN / TEST PREDICTIONS
# ==========================================
y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)

r2_tr, rmse_tr, mae_tr = metrics(
    y_train,
    y_train_pred
)

r2_te, rmse_te, mae_te = metrics(
    y_test,
    y_test_pred
)

# ==========================================
# LOAD UNSEEN DATA
# ==========================================
arc = read_clean_excel(arc_path)

qmof = read_clean_excel(qmof_path)

X_arc = arc.drop(target_col, axis=1)
y_arc = arc[target_col]

X_q = qmof.drop(target_col, axis=1)
y_q = qmof[target_col]

# ==========================================
# ALIGN FEATURES
# ==========================================
X_arc.columns = X_arc.columns.str.replace(
    '\ufeff',
    '',
    regex=False
)

X_q.columns = X_q.columns.str.replace(
    '\ufeff',
    '',
    regex=False
)

for col in feature_cols:

    if col not in X_arc.columns:
        X_arc[col] = 0

    if col not in X_q.columns:
        X_q[col] = 0

X_arc = X_arc[feature_cols]

X_q = X_q[feature_cols]

# ==========================================
# PREDICT UNSEEN
# ==========================================
y_arc_pred = model.predict(X_arc)

y_q_pred = model.predict(X_q)

r2_arc, rmse_arc, mae_arc = metrics(
    y_arc,
    y_arc_pred
)

r2_q, rmse_q, mae_q = metrics(
    y_q,
    y_q_pred
)

print("\nARC R2:", r2_arc)

print("QMOF R2:", r2_q)

# ==========================================
# SHAP — TEST SET
# ==========================================
X_sample = X_test.copy()

explainer = shap.TreeExplainer(model)

print("SHAP hesaplanıyor...")

shap_values = explainer.shap_values(X_sample)

shap_feature_names = [

    scientific_feature_names.get(col, col)

    for col in X_sample.columns
]

# ==========================================
# FIGURE
# ==========================================
fig = plt.figure(figsize=(34, 11))

ax1 = fig.add_axes([0.05, 0.18, 0.24, 0.68])

ax2 = fig.add_axes([0.39, 0.16, 0.23, 0.72])

cax = fig.add_axes([0.64, 0.16, 0.004, 0.72])

ax3 = fig.add_axes([0.74, 0.18, 0.24, 0.68])

# ==========================================
# AXIS RANGE REVISION
# Axis goes to 13, but labels show up to 12
# ==========================================
axis_min, axis_max = 0, 13.2

ticks = np.arange(0, 13, 3)   # 0, 3, 6, 9, 12


def format_scatter_axis(ax):

    ax.set_xlim(axis_min, axis_max)

    ax.set_ylim(axis_min, axis_max)

    ax.set_xticks(ticks)

    ax.set_yticks(ticks)

    ax.set_aspect(
        'equal',
        adjustable='box'
    )

    ax.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        color='black',
        linestyle='--',
        linewidth=2.0
    )

    ax.tick_params(
        axis='both',
        labelsize=22,
        length=6,
        width=1.4
    )

    for spine in ax.spines.values():

        spine.set_linewidth(2.0)

    ax.set_xlabel(
        r'Simulated $\mathrm{N}_{\mathrm{N_2O}}$ (mol/kg)',
        fontsize=25
    )

    ax.set_ylabel(
        r'Predicted $\mathrm{N}_{\mathrm{N_2O}}$ (mol/kg)',
        fontsize=25
    )

# ==========================================
# PANEL 1 — TRAIN / TEST
# ==========================================
ax1.scatter(
    y_train,
    y_train_pred,
    c='fuchsia',
    s=95,
    edgecolor='black',
    linewidth=0.6,
    alpha=0.85
)

ax1.scatter(
    y_test,
    y_test_pred,
    c='#2E8B57',
    s=95,
    edgecolor='black',
    linewidth=0.6,
    alpha=0.85
)

format_scatter_axis(ax1)

# Train text
ax1.text(
    0.04,
    0.94,
    'Train',
    color='fuchsia',
    fontsize=24,
    fontweight='bold',
    transform=ax1.transAxes
)

ax1.text(
    0.04,
    0.88,
    f'R²: {r2_tr:.3f}',
    color='fuchsia',
    fontsize=22,
    transform=ax1.transAxes
)

ax1.text(
    0.04,
    0.82,
    f'RMSE: {rmse_tr:.3f}',
    color='fuchsia',
    fontsize=22,
    transform=ax1.transAxes
)

ax1.text(
    0.04,
    0.76,
    f'MAE: {mae_tr:.3f}',
    color='fuchsia',
    fontsize=22,
    transform=ax1.transAxes
)

ax1.text(
    0.97,
    0.24,
    'Test',
    color='#2E8B57',
    fontsize=24,
    fontweight='bold',
    ha='right',
    transform=ax1.transAxes
)

ax1.text(
    0.97,
    0.18,
    f'R²: {r2_te:.3f}',
    color='#2E8B57',
    fontsize=22,
    ha='right',
    transform=ax1.transAxes
)

ax1.text(
    0.97,
    0.12,
    f'RMSE: {rmse_te:.3f}',
    color='#2E8B57',
    fontsize=22,
    ha='right',
    transform=ax1.transAxes
)

ax1.text(
    0.97,
    0.06,
    f'MAE: {mae_te:.3f}',
    color='#2E8B57',
    fontsize=22,
    ha='right',
    transform=ax1.transAxes
)
# ==========================================
# PANEL 2 — SHAP
# ==========================================
plt.sca(ax2)

matplotlib.rcParams.update({
    'font.size': 15,
    'axes.labelsize': 17,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15
})

shap_cmap = shap.plots.colors.red_blue

shap.summary_plot(
    shap_values,
    X_sample,
    feature_names=shap_feature_names,
    show=False,
    plot_size=None,
    max_display=13,
    color_bar=False,
    cmap=shap_cmap
)

ax2 = plt.gca()

ax2.set_xlabel(
    'SHAP value (impact on model output)',
    fontsize=20
)

ax2.tick_params(
    axis='x',
    labelsize=16
)

ax2.tick_params(
    axis='y',
    labelsize=16
)

formatter = ScalarFormatter(
    useMathText=True
)

formatter.set_scientific(True)

formatter.set_powerlimits((-2, 2))

ax2.xaxis.set_major_formatter(formatter)

ax2.ticklabel_format(
    axis='x',
    style='sci',
    scilimits=(-2, 2)
)

ax2.xaxis.get_offset_text().set_fontsize(14)

# ==========================================
# COLORBAR
# ==========================================
norm = Normalize(
    vmin=X_sample.values.min(),
    vmax=X_sample.values.max()
)

sm = ScalarMappable(
    cmap=shap_cmap,
    norm=norm
)

sm.set_array([])

cb = fig.colorbar(
    sm,
    cax=cax
)

cb.set_ticks([
    norm.vmin,
    norm.vmax
])

cb.set_ticklabels([
    'Low',
    'High'
])

cb.ax.tick_params(
    labelsize=18,
    length=0
)

cb.set_label(
    'Feature value',
    fontsize=20,
    rotation=90,
    labelpad=12
)

# ==========================================
# PANEL 3 — UNSEEN ARC-MOF / QMOF
# ==========================================
ax3.scatter(
    y_arc,
    y_arc_pred,
    c='blue',
    s=95,
    edgecolor='black',
    linewidth=0.6,
    alpha=0.85
)

ax3.scatter(
    y_q,
    y_q_pred,
    c='darkorange',
    s=95,
    edgecolor='black',
    linewidth=0.6,
    alpha=0.85
)

format_scatter_axis(ax3)

# ARC-MOF text
ax3.text(
    0.04,
    0.94,
    'ARC-MOF',
    color='blue',
    fontsize=24,
    fontweight='bold',
    transform=ax3.transAxes
)

ax3.text(
    0.04,
    0.88,
    f'R²: {r2_arc:.3f}',
    color='blue',
    fontsize=22,
    transform=ax3.transAxes
)

ax3.text(
    0.04,
    0.82,
    f'RMSE: {rmse_arc:.3f}',
    color='blue',
    fontsize=22,
    transform=ax3.transAxes
)

ax3.text(
    0.04,
    0.76,
    f'MAE: {mae_arc:.3f}',
    color='blue',
    fontsize=22,
    transform=ax3.transAxes
)

# QMOF text — truly lower-right
ax3.text(
    0.97,
    0.24,
    'QMOF',
    color='darkorange',
    fontsize=24,
    fontweight='bold',
    ha='right',
    transform=ax3.transAxes
)

ax3.text(
    0.97,
    0.18,
    f'R²: {r2_q:.3f}',
    color='darkorange',
    fontsize=22,
    ha='right',
    transform=ax3.transAxes
)

ax3.text(
    0.97,
    0.12,
    f'RMSE: {rmse_q:.3f}',
    color='darkorange',
    fontsize=22,
    ha='right',
    transform=ax3.transAxes
)

ax3.text(
    0.97,
    0.06,
    f'MAE: {mae_q:.3f}',
    color='darkorange',
    fontsize=22,
    ha='right',
    transform=ax3.transAxes
)
# ==========================================
# SAVE
# ==========================================
plt.savefig(
    save_path,
    dpi=300,
    bbox_inches='tight'
)

plt.show()