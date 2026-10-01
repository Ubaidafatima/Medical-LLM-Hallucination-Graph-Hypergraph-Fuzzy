# Third-Party Data, Models, and Software

The MIT License in this repository applies only to original code authored for
this project unless a file states otherwise.

It does not automatically apply to third-party resources.

## MedHallu

The study uses the MedHallu benchmark. Users should obtain the dataset from the
official distribution source and comply with the dataset's own license and
citation requirements.

This repository should preferably provide:
- sampling code,
- `Pair_ID` values,
- derived structural features,
rather than a duplicate redistribution of the complete dataset.

## MedGemma

The study uses:

`google/medgemma-4b-it`

MedGemma model weights and related hosted assets are **not redistributed** in
this repository. Users must obtain access from the official model provider /
Hugging Face repository and comply with all applicable terms of use.

## Hugging Face

Hugging Face Hub and Transformers are used to access and run the model.
Authentication tokens must never be committed to this repository.

## Other dependencies

PyTorch, NumPy, pandas, SciPy, scikit-learn, statsmodels, NetworkX,
Matplotlib, and other dependencies remain governed by their respective
licenses.

## Research outputs

Original analysis code, repository documentation, and original derived tables
or figures produced by the authors may be released under the repository's MIT
License unless otherwise noted.

When redistributing derived data, verify that doing so remains compatible with
the terms of the underlying source dataset and model.
