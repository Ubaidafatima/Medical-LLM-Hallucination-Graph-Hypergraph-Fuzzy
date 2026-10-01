# Data directory

Do not use this folder as a replacement for the official MedHallu dataset distribution.

Recommended contents:

- `selected_pair_ids.csv` — identifiers for the 100 matched pairs used in the manuscript
- `sample_metadata.csv` — non-sensitive reproducibility metadata
- scripts/notebooks that regenerate the cohort from the official dataset

Avoid committing:
- the complete source dataset unless redistribution is clearly permitted and necessary;
- model weights;
- Hugging Face authentication tokens;
- local cache files.
