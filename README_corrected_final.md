# Attention Graph–Hypergraph and Fuzzy Fusion for Explainable Hallucination Risk Assessment in Medical Large Language Models

Official research code and reproducibility materials for the manuscript:

**Attention Graph–Hypergraph and Fuzzy Fusion for Explainable Hallucination Risk Assessment in Medical Large Language Models**

**Authors:** Ubaida Fatima and Mustafain Ali

- **Ubaida Fatima** — Department of Mathematics, NED University of Engineering and Technology, Karachi, Pakistan
- **Mustafain Ali** — School of Mathematics and Statistics, Rochester Institute of Technology, Rochester, NY, USA

**Corresponding author:** Ubaida Fatima  
**Email:** ubaida@neduet.edu.pk

---

## Overview

This repository contains the computational workflow for an explainable hallucination-risk assessment framework for medical large language models using internal transformer attention.

The pipeline:

1. loads paired factual and hallucinated responses from **MedHallu**;
2. extracts attention tensors from **`google/medgemma-4b-it`**;
3. aggregates subword-token attention into word-level attention;
4. constructs sparse directed attention graphs;
5. computes graph-theoretic and perturbation-based structural features;
6. controls response-length confounding through pair-aware cross-fitted normalization;
7. constructs overlapping weighted attention hypergraphs;
8. extracts higher-order hypergraph features;
9. fuses robust graph and hypergraph evidence through a zero-order Sugeno fuzzy system;
10. evaluates the framework using repeated and nested pair-aware cross-validation.

The study treats attention as a **structural signal**, not as a causal explanation of model reasoning.

---

## Main methodological contributions

The repository reproduces the following components of the study:

- matched factual–hallucinated analysis using `Pair_ID`;
- MedGemma final-four-layer attention aggregation;
- word-level directed attention graph construction;
- top-\(k\) attention sparsification with primary \(k=3\);
- GCCDC and perturbation-based Structural Influence;
- Question Attention Support (QAS);
- pair-aware cross-fitted response-length normalization;
- paired Wilcoxon signed-rank testing with Holm correction;
- weighted overlapping attention hypergraphs;
- hypergraph sensitivity analysis for \(k=3,5,10\);
- higher-order features including Overlap Pair Fraction and Hyperdegree Gini;
- zero-order Sugeno graph–hypergraph fuzzy risk fusion;
- repeated 5-fold pair-aware cross-validation;
- leakage-aware nested feature-selection sensitivity analysis;
- difficulty-stratified evaluation.

---

## Experimental cohort

The manuscript uses the `pqa_labeled` MedHallu working split.

- Working split: **1,000 paired observations**
- Methodological-validation subset: **100 question pairs**
- Total analyzed responses: **200**
- Difficulty distribution in the selected cohort:
  - Easy: **27**
  - Medium: **32**
  - Hard: **41**
- Sampling seed: **42**

The complete MedHallu dataset is **not redistributed** in this repository. The supplied pair identifiers and metadata support reproducible reconstruction of the study cohort from the official dataset source.

---

## Model and implementation

Primary model:

```text
google/medgemma-4b-it
```

Primary inference configuration:

```text
dtype: bfloat16
device_map: auto
attention implementation: eager
output_attentions: True
attention aggregation: mean over heads, then mean over final 4 layers
primary graph top-k: 3
random seed: 42
```

The model is accessed through the Hugging Face Hub using the Transformers ecosystem.

**Important:** MedGemma model weights are not redistributed in this repository. Users must obtain access from the official model provider and comply with the applicable access terms.

---

## Repository structure

```text
Medical-LLM-Hallucination-Graph-Hypergraph-Fuzzy/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── THIRD_PARTY.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── README.md
│   ├── 01_Dataset_Preparation.ipynb
│   ├── 02_MedGemma_Attention_Extraction.ipynb
│   ├── 03_Word_Level_Attention_Graph.ipynb
│   ├── 04_Graph_Feature_Analysis.ipynb
│   ├── 05_Full_100Pair_Pipeline.ipynb
│   ├── 06_Statistical_Validation.ipynb
│   ├── 06B_Length_Normalization.ipynb
│   ├── 07_Hypergraph_Analysis.ipynb
│   └── 08_Fuzzy_Risk_Fusion.ipynb
│
├── data/
│   ├── README.md
│   ├── selected_pair_ids.csv
│   └── sample_metadata.csv
│
└── results/
    ├── README.md
    ├── graph_length_normalized_tests.csv
    ├── graph_candidate_screen.csv
    ├── hypergraph_paired_tests.csv
    ├── hypergraph_candidate_screen.csv
    ├── fuzzy_repeated_cv_summary.csv
    ├── fuzzy_nested_cv_summary.csv
    ├── nested_feature_selection_frequency.csv
    └── difficulty_all10_summary.csv
```

---

## Notebook workflow

### Notebook 01 — Dataset preparation

`notebooks/01_Dataset_Preparation.ipynb`

- loads MedHallu `pqa_labeled`;
- inspects the working split and class distributions;
- creates the difficulty-stratified 100-pair cohort;
- preserves matched factual/hallucinated correspondence;
- assigns `Pair_ID`;
- exports pair identifiers and metadata.

### Notebook 02 — MedGemma attention extraction

`notebooks/02_MedGemma_Attention_Extraction.ipynb`

- loads `google/medgemma-4b-it`;
- applies the model chat template;
- extracts self-attention tensors;
- averages attention heads;
- aggregates the final four transformer layers;
- validates and stores the attention representation.

### Notebook 03 — Word-level attention graph

`notebooks/03_Word_Level_Attention_Graph.ipynb`

- reconstructs words from subword tokens;
- excludes control/chat-formatting tokens;
- aggregates token attention to words;
- keeps repeated words as distinct positional nodes;
- constructs sparse directed source-to-target attention graphs.

### Notebook 04 — Graph feature analysis

`notebooks/04_Graph_Feature_Analysis.ipynb`

Computes:

- degree-based features;
- PageRank;
- weighted betweenness;
- in/out strength;
- density;
- global clustering coefficient;
- GCCDC;
- communication efficiency;
- perturbation-based Structural Influence;
- Question Attention Support.

### Notebook 05 — Full 100-pair pipeline

`notebooks/05_Full_100Pair_Pipeline.ipynb`

- processes the complete 100-pair methodological-validation cohort;
- extracts factual and hallucinated response features;
- saves restart-safe checkpoints;
- stores response-level graph features;
- stores node-level centralities;
- stores compressed word-attention matrices.

### Notebook 06 — Statistical validation

`notebooks/06_Statistical_Validation.ipynb`

- paired descriptive statistics;
- Wilcoxon signed-rank tests;
- Holm multiple-testing correction;
- rank-biserial effect size;
- paired Cohen's \(d_z\);
- bootstrap confidence intervals;
- initial response-length sensitivity analysis.

### Notebook 06B — Length normalization

`notebooks/06B_Length_Normalization.ipynb`

- 5-fold `GroupKFold` by `Pair_ID`;
- label-independent cross-fitted response-length normalization;
- linear and spline sensitivity analysis;
- paired testing after confound control;
- robust graph candidate screening.

Primary robust graph signals used in the final framework:

```text
Max GCCDC
Max Structural Influence
```

### Notebook 07 — Hypergraph analysis

`notebooks/07_Hypergraph_Analysis.ipynb`

For target word \(i\), an overlapping hyperedge is formed as

```text
e_i = {i} ∪ S_i^(k)
```

where \(S_i^{(k)}\) contains the strongest positive non-self incoming answer-word attention sources.

Hypergraph sensitivity is evaluated at:

```text
k = 3, 5, 10
```

Primary robust hypergraph signals used in the final framework:

```text
Overlap Pair Fraction
Hyperdegree Gini
```

### Notebook 08 — Fuzzy risk fusion

`notebooks/08_Fuzzy_Risk_Fusion.ipynb`

Final structural inputs:

```text
Max GCCDC
Max Structural Influence
Overlap Pair Fraction
Hyperdegree Gini
```

The notebook applies complementary logistic low/high memberships and a zero-order Sugeno rule base to generate a continuous hallucination-risk score.

Evaluation includes:

- 10 repetitions of 5-fold pair-aware cross-validation;
- nested feature-selection sensitivity analysis;
- graph-only baseline;
- hypergraph-only baseline;
- logistic-regression reference;
- difficulty-stratified evaluation.

---

## Main manuscript results

The fixed four-feature fuzzy model achieved approximately:

```text
ROC-AUC:             0.665 ± 0.003
PR-AUC:              0.646 ± 0.006
Balanced accuracy:   0.631 ± 0.013
F1:                  0.665 ± 0.011
MCC:                 0.268 ± 0.026
Sensitivity:         0.734 ± 0.017
Specificity:         0.528 ± 0.021
```

Leakage-aware nested validation produced approximately:

```text
ROC-AUC:              0.664 ± 0.010
PR-AUC:               0.644 ± 0.011
Balanced accuracy:    0.642 ± 0.023
F1:                   0.661 ± 0.027
MCC:                  0.286 ± 0.047
Pairwise ordering:    0.656 ± 0.020
```

These results should be interpreted as **moderate but reproducible structural discrimination**, rather than diagnostic-grade hallucination detection.

---

## Difficulty-stratified nested results

Across all 10 repeated nested-CV runs:

| Difficulty | Pairs | ROC-AUC | PR-AUC | Balanced Accuracy | F1 | MCC | Pairwise Ordering |
|---|---:|---:|---:|---:|---:|---:|---:|
| Easy | 27 | 0.718 ± 0.014 | 0.745 ± 0.019 | 0.661 ± 0.037 | 0.663 ± 0.045 | 0.324 ± 0.074 | 0.737 ± 0.041 |
| Medium | 32 | 0.562 ± 0.029 | 0.546 ± 0.025 | 0.597 ± 0.033 | 0.610 ± 0.040 | 0.195 ± 0.067 | 0.478 ± 0.039 |
| Hard | 41 | 0.705 ± 0.013 | 0.679 ± 0.015 | 0.665 ± 0.027 | 0.695 ± 0.024 | 0.337 ± 0.055 | 0.741 ± 0.029 |

Detailed summary files are available in the `results/` directory.

---

## Installation

Python 3.10+ is recommended.

```bash
git clone https://github.com/Ubaidafatima/Medical-LLM-Hallucination-Graph-Hypergraph-Fuzzy.git
cd Medical-LLM-Hallucination-Graph-Hypergraph-Fuzzy
pip install -r requirements.txt
```

For MedGemma access, authenticate separately with Hugging Face after accepting the applicable access terms.

Example:

```python
from huggingface_hub import login
login()
```

Do **not** hard-code or commit Hugging Face access tokens.

---

## Reproducibility environment

The primary recorded environment used in the manuscript was:

```text
Execution environment: Google Colab
GPU: NVIDIA A100-SXM4-80GB
CUDA: 12.8
PyTorch: 2.11.0+cu128
Transformers: 5.18.0
NetworkX: 3.7
Numerical precision: bfloat16
Primary graph k: 3
Random seed: 42
```

Google Colab dynamically assigns host CPUs; therefore, a fixed CPU model is not claimed.

The statistical, hypergraph, and fuzzy-analysis stages are CPU-compatible once the attention and graph-feature outputs have been produced.

Intermediate response-level features, node-level metrics, and word-attention matrices were checkpointed during batch processing to support restart-safe execution.

---

## Data availability

This repository does **not** replace the official MedHallu distribution and does not redistribute MedGemma model weights.

Recommended reproducibility workflow:

1. obtain MedHallu from its official source;
2. run `notebooks/01_Dataset_Preparation.ipynb`;
3. reproduce or verify the 100-pair cohort using seed 42 and `data/selected_pair_ids.csv`;
4. obtain access to `google/medgemma-4b-it` from the official provider;
5. run the notebooks in numerical order;
6. compare reproduced summary outputs with the CSV files under `results/`.

The `data/` directory contains only reproducibility identifiers and non-sensitive metadata for the selected cohort.

---

## Results directory

The repository includes compact manuscript-supporting derived outputs:

```text
results/graph_length_normalized_tests.csv
results/graph_candidate_screen.csv
results/hypergraph_paired_tests.csv
results/hypergraph_candidate_screen.csv
results/fuzzy_repeated_cv_summary.csv
results/fuzzy_nested_cv_summary.csv
results/nested_feature_selection_frequency.csv
results/difficulty_all10_summary.csv
```

Large intermediate attention matrices are intentionally excluded from the main Git repository.

---

## Third-party resources

This repository uses or interfaces with third-party resources including:

- MedHallu;
- MedGemma;
- Hugging Face Hub / Transformers;
- PyTorch;
- NetworkX;
- SciPy;
- scikit-learn;
- statsmodels.

Third-party datasets, model weights, software, and documentation remain subject to their own licenses and terms.

See [`THIRD_PARTY.md`](THIRD_PARTY.md).

---

## License

Original code and repository documentation are released under the **MIT License** unless otherwise stated.

This license does **not** relicense:

- MedHallu data;
- MedGemma model weights;
- Hugging Face-hosted model assets;
- third-party software or documentation.

See [`LICENSE`](LICENSE) and [`THIRD_PARTY.md`](THIRD_PARTY.md).

---

## Citation

If you use this repository before publication of the associated journal article, please cite the repository as:

```bibtex
@misc{fatima2026attention,
  author       = {Ubaida Fatima and Mustafain Ali},
  title        = {Attention Graph--Hypergraph and Fuzzy Fusion for Explainable Hallucination Risk Assessment in Medical Large Language Models},
  year         = {2026},
  note         = {Manuscript and accompanying research code},
  howpublished = {GitHub repository},
  url          = {https://github.com/Ubaidafatima/Medical-LLM-Hallucination-Graph-Hypergraph-Fuzzy}
}
```

After journal publication, replace this temporary citation with the final published article citation and DOI.

---

## Contact

For questions about the methodology or reproducibility materials:

**Ubaida Fatima**  
Department of Mathematics  
NED University of Engineering and Technology  
Karachi, Pakistan  
Email: ubaida@neduet.edu.pk
