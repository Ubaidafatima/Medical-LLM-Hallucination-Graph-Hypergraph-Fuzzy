# Attention Graph–Hypergraph and Fuzzy Fusion for Explainable Hallucination Risk Assessment in Medical Large Language Models

Official research code and reproducibility materials for the manuscript:

**Attention Graph–Hypergraph and Fuzzy Fusion for Explainable Hallucination Risk Assessment in Medical Large Language Models**

**Authors:** Ubaida Fatima and Mustafain Ali

- Ubaida Fatima — Department of Mathematics, NED University of Engineering and Technology, Karachi, Pakistan
- Mustafain Ali — School of Mathematics and Statistics, Rochester Institute of Technology, Rochester, NY, USA

Corresponding author: **Ubaida Fatima**  
Email: **ubaida@neduet.edu.pk**

---

## Overview

This repository implements an explainable hallucination-risk assessment framework for medical large language models using internal transformer attention.

The pipeline:

1. loads paired factual and hallucinated responses from **MedHallu**;
2. extracts attention tensors from **`google/medgemma-4b-it`**;
3. aggregates subword-token attention into word-level attention;
4. constructs sparse directed attention graphs;
5. computes graph-theoretic and perturbation-based structural features;
6. removes response-length confounding through pair-aware cross-fitted normalization;
7. builds overlapping weighted attention hypergraphs;
8. extracts higher-order hypergraph features;
9. fuses robust graph and hypergraph evidence through a zero-order Sugeno fuzzy system;
10. evaluates the framework using repeated and nested pair-aware cross-validation.

The study treats attention as a **structural signal**, not as a causal explanation of model reasoning.

---

## Main methodological contributions

The repository reproduces the following methodological components:

- paired factual–hallucinated analysis using `Pair_ID`;
- final-four-layer MedGemma attention aggregation;
- word-level directed attention graph construction;
- top-\(k\) attention sparsification with primary \(k=3\);
- GCCDC and perturbation-based Structural Influence;
- Question Attention Support (QAS);
- pair-aware cross-fitted response-length normalization;
- Wilcoxon signed-rank testing with Holm correction;
- weighted overlapping attention hypergraphs;
- hypergraph sensitivity analysis for \(k=3,5,10\);
- higher-order features including Overlap Pair Fraction and Hyperdegree Gini;
- zero-order Sugeno fuzzy graph–hypergraph risk fusion;
- repeated 5-fold pair-aware cross-validation;
- leakage-aware nested feature-selection sensitivity analysis.

---

## Experimental cohort

The manuscript uses the `pqa_labeled` MedHallu working split.

- Working split: **1,000 paired observations**
- Methodological-validation subset: **100 question pairs**
- Total analyzed responses: **200**
- Difficulty distribution in the selected 100-pair cohort:
  - Easy: **27**
  - Medium: **32**
  - Hard: **41**
- Sampling seed: **42**

The repository should not be treated as a redistribution source for the complete MedHallu dataset. Use the official dataset source and reproduce the subset using the supplied sampling code / pair identifiers.

---

## Model

Primary model:

```text
google/medgemma-4b-it
```

The model is accessed through the Hugging Face Hub and loaded with the Transformers ecosystem.

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

**Important:** MedGemma model files are not redistributed in this repository. Users must obtain access from the official model provider and comply with the applicable Health AI / Hugging Face terms.

---

## Repository structure

```text
medical-llm-hallucination-graph-hypergraph/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── THIRD_PARTY.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 01_dataset_preparation.ipynb
│   ├── 02_medgemma_attention_extraction.ipynb
│   ├── 03_word_level_attention_graph.ipynb
│   ├── 04_graph_feature_analysis.ipynb
│   ├── 05_full_pair_pipeline.ipynb
│   ├── 06_statistical_validation.ipynb
│   ├── 06B_length_normalization.ipynb
│   ├── 07_hypergraph_analysis.ipynb
│   └── 08_fuzzy_risk_fusion.ipynb
│
├── src/
│   ├── attention_utils.py
│   ├── graph_features.py
│   ├── hypergraph_features.py
│   ├── normalization.py
│   └── fuzzy_risk.py
│
├── data/
│   ├── README.md
│   ├── selected_pair_ids.csv
│   └── sample_metadata.csv
│
├── results/
│   ├── README.md
│   ├── graph_statistics.csv
│   ├── hypergraph_statistics.csv
│   ├── fuzzy_cv_summary.csv
│   └── difficulty_summary.csv
│
└── figures/
    ├── dataset_visualization.png
    ├── methodology_framework.png
    └── fuzzy_membership_sugeno.png
```

---

## Notebook workflow

### Notebook 01 — Dataset preparation

- Load MedHallu `pqa_labeled`
- inspect schema and class distributions
- create a difficulty-stratified 100-pair cohort
- preserve paired factual/hallucinated correspondence
- assign `Pair_ID`
- export selected pair identifiers and metadata

### Notebook 02 — MedGemma attention extraction

- load `google/medgemma-4b-it`
- apply the model chat template
- extract attention tensors
- average attention heads
- aggregate the final four transformer layers
- save reproducible attention representations

### Notebook 03 — Word-level attention graph

- reconstruct words from subword tokens
- exclude control/chat-formatting tokens
- aggregate token attention to words
- preserve repeated words as distinct positional nodes
- construct sparse directed source-to-target attention graphs

### Notebook 04 — Graph-feature analysis

Computes:

- degree-based features
- PageRank
- weighted betweenness
- in/out strength
- density
- global clustering coefficient
- GCCDC
- communication efficiency
- perturbation-based Structural Influence
- Question Attention Support

### Notebook 05 — Full paired pipeline

- processes all selected factual/hallucinated pairs
- writes restart-safe checkpoints
- stores response-level features
- stores node-level centralities
- stores compressed word-attention matrices

### Notebook 06 — Statistical validation

- paired descriptive statistics
- Wilcoxon signed-rank tests
- Holm multiple-testing correction
- rank-biserial effect size
- paired Cohen's \(d_z\)
- bootstrap confidence intervals
- response-length sensitivity analysis

### Notebook 06B — Length normalization

- 5-fold `GroupKFold` by `Pair_ID`
- label-independent response-length normalization
- spline and linear normalization sensitivity
- paired testing after confound control
- robust candidate feature screening

Primary graph signals identified in the manuscript:

```text
Max GCCDC
Max Structural Influence
```

### Notebook 07 — Hypergraph analysis

For target word \(i\):

```text
e_i = {i} ∪ S_i^(k)
```

where \(S_i^{(k)}\) contains the strongest positive non-self incoming answer-word attention sources.

Primary higher-order signals identified in the manuscript:

```text
Overlap Pair Fraction
Hyperdegree Gini
```

Sensitivity is evaluated at:

```text
k = 3, 5, 10
```

### Notebook 08 — Fuzzy risk fusion

Final structural inputs:

```text
Max GCCDC
Max Structural Influence
Overlap Pair Fraction
Hyperdegree Gini
```

The model uses complementary logistic low/high memberships and a zero-order Sugeno rule base to produce a continuous hallucination-risk score.

Evaluation includes:

- 10 repetitions of 5-fold pair-aware cross-validation
- nested feature-selection sensitivity analysis
- graph-only baseline
- hypergraph-only baseline
- logistic-regression reference
- difficulty-stratified evaluation

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
ROC-AUC:                  0.664 ± 0.010
PR-AUC:                   0.644 ± 0.011
Balanced accuracy:        0.642 ± 0.023
F1:                       0.661 ± 0.027
MCC:                      0.286 ± 0.047
Pairwise ordering:        0.656 ± 0.020
```

These values should be interpreted as **moderate but reproducible structural discrimination**, not diagnostic-grade hallucination detection.

---

## Difficulty-stratified nested results

Across all 10 repeated nested-CV runs:

| Difficulty | Pairs | ROC-AUC | PR-AUC | Balanced Accuracy | F1 | MCC | Pairwise Ordering |
|---|---:|---:|---:|---:|---:|---:|---:|
| Easy | 27 | 0.718 ± 0.014 | 0.745 ± 0.019 | 0.661 ± 0.037 | 0.663 ± 0.045 | 0.324 ± 0.074 | 0.737 ± 0.041 |
| Medium | 32 | 0.562 ± 0.029 | 0.546 ± 0.025 | 0.597 ± 0.033 | 0.610 ± 0.040 | 0.195 ± 0.067 | 0.478 ± 0.039 |
| Hard | 41 | 0.705 ± 0.013 | 0.679 ± 0.015 | 0.665 ± 0.027 | 0.695 ± 0.024 | 0.337 ± 0.055 | 0.741 ± 0.029 |

---

## Installation

Python 3.10+ is recommended.

```bash
git clone https://github.com/Ubaidafatima/Medical-LLM-Hallucination-Graph-Hypergraph-Fuzzy.git
cd Medical-LLM-Hallucination-Graph-Hypergraph-Fuzzy
pip install -r requirements.txt
```

For MedGemma access, authenticate separately with Hugging Face after accepting the model's applicable access terms.

Example:

```python
from huggingface_hub import login
login()
```

Do not hard-code access tokens inside notebooks.

---

## Reproducibility notes

The manuscript's recorded primary environment was:

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

Statistical, hypergraph, and fuzzy stages are CPU-compatible.

Intermediate response-level features, node-level metrics, and word-attention matrices are checkpointed to support restart-safe execution.

---

## Data availability

This repository does **not** aim to replace the official MedHallu distribution.

Recommended reproducibility workflow:

1. obtain MedHallu from the official source;
2. run `01_dataset_preparation.ipynb`;
3. reproduce the 100-pair cohort using seed 42 and/or `data/selected_pair_ids.csv`;
4. obtain MedGemma from the official provider;
5. run the notebooks in numerical order.

Processed structural features and summary statistics may be provided under `results/`.

---

## Third-party resources

This repository uses or interfaces with third-party resources including:

- MedHallu
- MedGemma
- Hugging Face Hub / Transformers
- PyTorch
- NetworkX
- SciPy
- scikit-learn
- statsmodels

Third-party datasets, model weights, software, and documentation remain subject to their own licenses and terms. See [`THIRD_PARTY.md`](THIRD_PARTY.md).

---

## License

The original source code in this repository is released under the **MIT License** unless otherwise noted.

This license does **not** relicense:

- MedHallu data,
- MedGemma model weights,
- Hugging Face-hosted model assets,
- third-party software,
- third-party figures or documentation.

See [`LICENSE`](LICENSE) and [`THIRD_PARTY.md`](THIRD_PARTY.md).

---

## Citation

If you use this repository, please cite the associated manuscript:

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

After journal publication, replace this temporary repository citation with the final journal citation and DOI.

---

## Contact

For questions about the methodology or code:

**Ubaida Fatima**  
Department of Mathematics  
NED University of Engineering and Technology  
Karachi, Pakistan  
Email: ubaida@neduet.edu.pk
