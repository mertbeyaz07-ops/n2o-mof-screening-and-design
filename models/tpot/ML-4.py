# -*- coding: utf-8 -*-
"""
Created on Thu May  7 09:56:28 2026

@author: mertb
"""

# ==========================================
# IMPORTS
# ==========================================
import numpy as np
import pandas as pd
import warnings
from copy import copy

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import ScalarFormatter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor
from tpot.export_utils import set_param_recursive

warnings.filterwarnings("ignore")

# ==========================================
# SETTINGS
# ==========================================
target_col = 'N2O_uptake_1bar_molkg'

train_path = r'C:/Users/mertb/OneDrive/Masaüstü/FINAL DATA/10_50_core_Arc_1_1_qst0_kh/coremof_1_10_arcmof.csv'
qmof_path  = r'C:/Users/mertb/OneDrive/Masaüstü/FINAL DATA/10_50_core_Arc_1_1_qst0_kh/QMOF_N2O_Merged_Dataset.xlsx'

save_path = r'D:/NEMO_Visioner_Mert_Beyaz/TPOT_POLY_XGB_IMPUTER_FULL_PLOT_REVISED_RIGHT10.png'

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

    return df


def metrics(y_true, y_pred):

    return (
        r2_score(y_true, y_pred),
        np.sqrt(mean_squared_error(y_true, y_pred)),
        mean_absolute_error(y_true, y_pred)
    )


def align_features(df, feature_cols, target_col):

    df = clean_columns(df)

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X = clean_columns(X)

    for col in feature_cols:

        if col not in X.columns:
            X[col] = np.nan

    X = X[feature_cols]

    return X, y

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
# LOAD TRAINING DATA
# ==========================================
df = pd.read_csv(
    train_path,
    sep=None,
    engine="python"
)

df = clean_columns(df)

X = df.drop(target_col, axis=1)
y = df[target_col]

X = clean_columns(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=0.8,
    test_size=0.2,
    random_state=42
)

feature_cols = list(X_train.columns)

# ==========================================
# IMPUTER
# ==========================================
imputer = SimpleImputer(
    strategy="median"
)

imputer.fit(X_train)

X_train_imp = imputer.transform(X_train)
X_test_imp = imputer.transform(X_test)

# ==========================================
# MODEL — TPOT PIPELINE
# ==========================================
model = make_pipeline(
    make_union(
        PolynomialFeatures(
            degree=2,
            include_bias=False,
            interaction_only=False
        ),
        FunctionTransformer(copy)
    ),
    XGBRegressor(
        learning_rate=0.1,
        max_depth=7,
        min_child_weight=15,
        n_estimators=100,
        n_jobs=1,
        objective="reg:squarederror",
        subsample=0.7500000000000001,
        verbosity=0,
        random_state=42
    )
)

set_param_recursive(
    model.steps,
    'random_state',
    42
)

model.fit(
    X_train_imp,
    y_train
)

# ==========================================
# TRAIN / TEST PREDICTIONS
# ==========================================
y_train_pred = model.predict(X_train_imp)
y_test_pred = model.predict(X_test_imp)

r2_tr, rmse_tr, mae_tr = metrics(
    y_train,
    y_train_pred
)

r2_te, rmse_te, mae_te = metrics(
    y_test,
    y_test_pred
)

# ==========================================
# LOAD + ALIGN QMOF UNSEEN DATA
# ==========================================
qmof = read_clean_excel(qmof_path)

X_q, y_q = align_features(
    qmof,
    feature_cols,
    target_col
)

X_q_imp = imputer.transform(X_q)

y_q_pred = model.predict(X_q_imp)

r2_q, rmse_q, mae_q = metrics(
    y_q,
    y_q_pred
)

print("\nTrain R2:", r2_tr)
print("Test R2 :", r2_te)
print("QMOF R2 :", r2_q)

# ==========================================
# SHAP — ORIGINAL FEATURE AGGREGATION
# ==========================================
X_sample_raw = X_test.copy()

X_sample_imp = imputer.transform(X_sample_raw)

X_sample_original_df = pd.DataFrame(
    X_sample_imp,
    columns=feature_cols,
    index=X_test.index
)

feature_transformer = model.named_steps["featureunion"]
xgb_model = model.named_steps["xgbregressor"]

poly = feature_transformer.transformer_list[0][1]

X_sample_transformed = feature_transformer.transform(X_sample_imp)

explainer = shap.TreeExplainer(xgb_model)

print("SHAP hesaplanıyor...")

shap_values_transformed = explainer.shap_values(X_sample_transformed)

# ==========================================
# AGGREGATE TRANSFORMED SHAP VALUES
# BACK TO ORIGINAL FEATURES
# ==========================================
n_original = len(feature_cols)
n_samples = X_sample_imp.shape[0]

aggregated_shap = np.zeros(
    (n_samples, n_original)
)

poly_powers = poly.powers_
n_poly_features = poly_powers.shape[0]

for j in range(n_poly_features):

    involved = np.where(poly_powers[j] > 0)[0]

    if len(involved) == 0:
        continue

    contribution = shap_values_transformed[:, j] / len(involved)

    for idx in involved:
        aggregated_shap[:, idx] += contribution

start_original = n_poly_features
end_original = start_original + n_original

aggregated_shap += shap_values_transformed[
    :,
    start_original:end_original
]

shap_values = aggregated_shap

X_sample = X_sample_original_df

shap_feature_names = [

    scientific_feature_names.get(col, col)

    for col in X_sample.columns
]

# ==========================================
# FIGURE — MANUAL LAYOUT
# ==========================================
fig = plt.figure(figsize=(34, 11))

ax1 = fig.add_axes([0.05, 0.18, 0.24, 0.68])

ax2 = fig.add_axes([0.39, 0.16, 0.23, 0.72])

cax = fig.add_axes([0.64, 0.16, 0.004, 0.72])

ax3 = fig.add_axes([0.74, 0.18, 0.24, 0.68])

# ==========================================
# AXIS RANGES
# Left panel: Train/Test
# Right panel: QMOF unseen
# ==========================================
axis_min_left, axis_max_left = 0, 13.2
ticks_left = np.arange(0, 13, 3)      # 0, 3, 6, 9, 12

axis_min_right, axis_max_right = 0, 10.0
ticks_right = np.arange(0, 10, 3)     # 0, 3, 6, 9


def format_scatter_axis(ax, axis_min, axis_max, ticks):

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

format_scatter_axis(
    ax1,
    axis_min_left,
    axis_max_left,
    ticks_left
)

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

# Test text — lower-right corner
ax1.text(
    0.97,
    0.22,
    'Test',
    color='#2E8B57',
    fontsize=24,
    fontweight='bold',
    ha='right',
    transform=ax1.transAxes
)

ax1.text(
    0.97,
    0.16,
    f'R²: {r2_te:.3f}',
    color='#2E8B57',
    fontsize=22,
    ha='right',
    transform=ax1.transAxes
)

ax1.text(
    0.97,
    0.10,
    f'RMSE: {rmse_te:.3f}',
    color='#2E8B57',
    fontsize=22,
    ha='right',
    transform=ax1.transAxes
)

ax1.text(
    0.97,
    0.04,
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
# MANUAL COLORBAR
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
# PANEL 3 — QMOF UNSEEN
# ==========================================
ax3.scatter(
    y_q,
    y_q_pred,
    c='darkorange',
    s=95,
    edgecolor='black',
    linewidth=0.6,
    alpha=0.85
)

format_scatter_axis(
    ax3,
    axis_min_right,
    axis_max_right,
    ticks_right
)

# QMOF text — lower-right corner
ax3.text(
    0.97,
    0.22,
    'QMOF',
    color='darkorange',
    fontsize=24,
    fontweight='bold',
    ha='right',
    transform=ax3.transAxes
)

ax3.text(
    0.97,
    0.16,
    f'R²: {r2_q:.3f}',
    color='darkorange',
    fontsize=22,
    ha='right',
    transform=ax3.transAxes
)

ax3.text(
    0.97,
    0.10,
    f'RMSE: {rmse_q:.3f}',
    color='darkorange',
    fontsize=22,
    ha='right',
    transform=ax3.transAxes
)

ax3.text(
    0.97,
    0.04,
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