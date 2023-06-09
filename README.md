# Contextualized Policy Recovery

This repository contains the code for replicating the results presented in the paper Modeling and *Interpreting Medical Decisions with Adaptive Imitation Learning*.

## Installation

```bash
conda env create -f environment.yml
conda activate cpr
pip install --no-build-isolation --disable-pip-version-check -e .
```

## Experiments
The code to run the experiments can be found in `src/ADNI_experiments.py` and `src/MIMIC_experiments.py` and is used to create `ADNI_results.txt` and `MIMIC_results.txt`

## Data 
1) ADNI: After getting Data access from the [Alzheimer's Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu/), place the `ADNIMERGE.csv` file in the `data/adni/raw/` folder.
2) MIMIC: After getting Data access for the [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/), follow the instruction for [installing it in a Postgres database](https://mimic.physionet.org/tutorials/install-mimic-locally-ubuntu/).
