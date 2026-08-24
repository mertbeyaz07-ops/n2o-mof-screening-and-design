# TPOT Descriptor-Based Models

This directory contains the scripts and Excel inputs used to develop the four descriptor-based TPOT models reported in the associated N₂O–MOF screening study.

## Important: update file names and paths before running

The Python scripts may contain Excel filenames and absolute or workstation-specific paths from the original computational environment. **Each user must update these values for their own system before executing a model.**

For every `ML-*.py` script, check and update:

- the input Excel filename;
- the path to the corresponding Excel directory;
- any output directory or output filename;
- Excel sheet names, if explicitly defined;
- feature and target column names, if the local spreadsheets use different headers.

Do not assume that a script will automatically discover the Excel files. The filename written in the script must exactly match the actual filename, including the `.xlsx` extension.

## Directory structure

```text
models/tpot/
├── ML-1_excels/
├── ML-2_excels/
├── ML-3_excels/
├── ML-4_excels/
├── ML-1.py
├── ML-2.py
├── ML-3.py
├── ML-4.py
├── TPOT_model_maker.py
└── README.md
```

Each model script should be connected to its corresponding Excel directory:

| Script | Input directory | Training-set composition |
|---|---|---|
| `ML-1.py` | `ML-1_excels/` | 3346 CoRE MOFs |
| `ML-2.py` | `ML-2_excels/` | 3346 CoRE MOFs + 1324 ARC-MOFs |
| `ML-3.py` | `ML-3_excels/` | 3346 CoRE MOFs + 2649 ARC-MOFs |
| `ML-4.py` | `ML-4_excels/` | 3346 CoRE MOFs + 13246 ARC-MOFs |

`TPOT_model_maker.py` contains the shared TPOT model-generation functionality used by the model workflow. Its imports, paths, and output locations should also be checked before execution.


## Pre-run checklist

Before running a model, verify that:

1. The selected `ML-*.py` script points to the matching `ML-*_excels/` directory.
2. Every referenced Excel file exists and its filename is written exactly as stored.
3. Required worksheet names and column headers match the script.
4. The N₂O uptake target corresponds to the study condition of 1 bar and 298 K.
5. The intended output directory exists or can be created by the script.
6. The active Python environment contains the required packages.

Typical dependencies include:

```text
pandas
numpy
scikit-learn
tpot==0.12.2
xgboost
openpyxl
joblib
```

The exact dependency set should be confirmed from the imports in the individual scripts.

## Reproducibility note

The Excel inputs, feature definitions, data splits, random seeds, TPOT settings, and exported pipelines should be kept consistent with the corresponding model configuration. Changes to filenames or local paths do not affect the scientific model definition, whereas changes to spreadsheet contents, column order, preprocessing, or train/test membership may change the reported results.
