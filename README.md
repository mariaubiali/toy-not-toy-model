# Toy-not-so-toy model

A repository for uncertainty quantification using different models on the T3 dataset.

## Structure

- `notebooks` contain a first implementation of a GP and the corresponding data generation in files with `data` in their name
- `T3gp` contains the overall implemenation and dataset generation for the GP, NTK, NN
    - `configs`: config files for NN and GP training
    - `data`: scripts for data generation
    - `plots`: plotting routines
    - `src`: Core computing space of NTK, NN and GP

## Quickstart for T3gp

### 1. Create Pyhton environment

Since PyMC and LHAPDF require different dependencies, two environments have to be created, one for the data generation and the other one for the GP sampling and NN/NTK evaluation.
The requirements for LHAPDF can be found in `t3-lhapdf.yaml`, the ones for pyMC in `t3-net.yaml` for the NN and GP approach and for the NTK runs in `t3-ntk.yaml`.

```bash
# LHAPDF env
conda env create -f envs/t3-lhapdf.yml
conda activate t3-lhapdf

# PyMC env
conda env create -f envs/t3-net.yml
conda activate t3-net

# NTK env
conda env create -f envs/t3-net.yml
conda activate t3-net
```

### 2. Generate Dataset
To generate the data set, the environment t3-lhapdf is necessary.
```bash
conda activate t3-lhapdf
cd T3gp/ (if not already in T3gp/)
python data/data_T3.py
```

### 3. Run first model
To run either the NN, NTK or GP, the environment t3-net is required.
```bash
conda activate t3-net
cd T3gp/ (if not already in T3gp/)
python src/main.py configs/config_nn.yaml
```
