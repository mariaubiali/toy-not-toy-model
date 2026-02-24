# Toy-not-so-toy model

A repository for uncertainty quantification using different models on the T3 dataset.

## Structure

- `notebooks` contain a first toy implementation of a GP and the corresponding data generation in files with `data` in their name
- `T3gp` contains the overall implemenation and dataset generation for the GP, NTK, NN
    - `configs`: config files for NN and GP training
    - `data`: scripts for data generation
    - `GP_objective`: implementation using separate x* to math the DelDebbio paper
    - `plots`: plotting routines
    - `src`: Core computing space of NTK, NN and GP
- `T3nn` contains previous NN implementation for Level 0 and 1 closure tests using the T3 data from a former Student

## Quickstart for T3gp

### 1. Create Pyhton environment

Since PyMC and LHAPDF require different dependencies, two environments have to be created, one for the data generation and the other one for the GP sampling and NN/NTK evaluation.
The requirements for LHAPDF can be found in `t3_lhapdf.yaml` and the ones for pyMC in `t3_net.yaml`.

```bash
# LHAPDF env
conda env create -f envs/t3-lhapdf.yml
conda activate t3-lhapdf

# PyMC env
conda env create -f envs/t3-net.yml
conda activate t3-net
```

### 2. Generate Dataset
```bash
python data/data_T3.py
```

### 3. Run first fit
```bash
python src/main.py configs/config_nn.yaml
```
