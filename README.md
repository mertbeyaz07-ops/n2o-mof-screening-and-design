# AI-Accelerated Computational Screening and Design of MOFs for N₂O Capture

This repository contains the data, trained-model resources, computational settings, generated structures, and analysis outputs associated with the manuscript:

> **AI-Accelerated Computational Screening and Design of Metal-Organic Frameworks for N₂O Capture Applications**  
> Mert Beyaz, Gokhan Onder Aksu, and Seda Keskin  
> Department of Chemical and Biological Engineering, Koç University, Istanbul, Türkiye

The study combines high-throughput molecular simulation, descriptor-based machine learning, a structure-native transformer, and AI-assisted inverse design to identify metal-organic frameworks (MOFs) for N₂O adsorption and separation from N₂O/CO₂/N₂ mixtures.

## Study overview

Three MOF databases were screened after geometric and structural preprocessing:

| Dataset | Structures used | Role in the study |
|---|---:|---|
| CoRE MOF 2025 | 3346 | Single-component screening, mixture screening, and model training |
| ARC-MOF | 13246 | Single-component screening and expansion of the model-training domain |
| QMOF | 3932 | Independent cross-database evaluation |

## Repository contents

```text
.
├── data/
│   ├── core_mof/
│   ├── arc_mof/
│   ├── qmof/
│   ├── hifimofs/
│   ├── screening/
│   └── ml_outputs/
├── models/
├── structures/
├── .gitignore
├── LICENSE
└── README.md
```

| Path | Contents |
|---|---|
| [`data/core_mof/`](data/core_mof/) | CoRE MOF structural descriptors, Widom results, single-component N₂O results, and N₂O/CO₂/N₂ mixture results |
| [`data/arc_mof/`](data/arc_mof/) | ARC-MOF structural descriptors, Widom results, and single-component N₂O results |
| [`data/qmof/`](data/qmof/) | QMOF structural descriptors, Widom results, and single-component N₂O results |
| [`data/hifimofs/`](data/hifimofs/) | HiFiMOFs structural descriptors, Widom results, and single-component N₂O results |
| [`data/screening/`](data/screening/) | Records associated with the multistage physical, chemical, DFT, and open-metal-site screening workflow |
| [`data/ml_outputs/`](data/ml_outputs/) | ML/TF predictions, PCA coordinates, and transformer attention outputs |
| [`models/`](models/) | TPOT scripts, configurations, exported pipelines, and PMTransformer scripts/configurations and prepared model inputs |
| [`structures/`](structures/) | Generated and screened HiFiMOF structures supplied as CIF files |

Individual subdirectories contain additional README files describing their files, parameters, naming conventions, and any required local-path changes.


## Machine-learning models

Two complementary model families were developed to predict single-component N₂O uptake at 1 bar and 298 K:

1. **ML-1 to ML-4:** descriptor-based pipelines optimized with TPOT 0.12.2.
2. **TF-1 to TF-4:** structure-native models fine-tuned from PMTransformer 2.2.0 using CIF-derived atom-graph and energy-grid representations.

| Model pair | Training-set composition | Total structures |
|---|---|---:|
| ML-1 / TF-1 | CoRE MOF | 3346 |
| ML-2 / TF-2 | CoRE MOF + every 10th uptake-ranked ARC-MOF | 4670 |
| ML-3 / TF-3 | CoRE MOF + every 5th uptake-ranked ARC-MOF | 5995 |
| ML-4 / TF-4 | CoRE MOF + full ARC-MOF set | 16592 |

Descriptor-based models used stratified 80/20 training/test splits. Transformer datasets used 80/10/10 training/validation/test splits. QMOF served as the independent cross-database evaluation set. ML-4 was the primary descriptor model, while TF-4 was integrated into the inverse-design workflow.

The repository also includes prediction tables, 768-dimensional PMTransformer representations projected onto principal components, and held-out-set attention outputs. PCA transformations were fitted using the training and validation representations before projecting the test and unseen sets.

### PMTransformer checkpoints

The trained `best.ckpt` files are not stored in this GitHub repository because of file-size limitations. They will be archived in Zenodo together with their model-to-file mapping and checksums.

```text
Zenodo DOI: TO BE ADDED
```

The smaller configuration, NPZ, prediction, and analysis files required to interpret and use the checkpoints are retained in this repository. The Zenodo filenames and checksums must be matched to the corresponding TF-1–TF-4 configurations before inference.


## License

Repository-authored code and documentation are released under the [MIT License](LICENSE). Third-party databases, pretrained models, software, and derived resources remain subject to their original licenses and citation requirements.

## Funding and contact

This work received funding from the European Union through the European Research Council (ERC, STARLET, grant agreement No. 101124002). Views and opinions expressed are those of the authors and do not necessarily reflect those of the European Union or the European Research Council.

For scientific questions, contact Prof. Seda Keskin at [skeskin@ku.edu.tr](mailto:skeskin@ku.edu.tr).