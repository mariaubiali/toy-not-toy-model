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

**venv**
```bash
python -m venv .venv
source .venv/bin/activate
add requirements.txt to repo
```

### 2. Generate Dataset
```bash
python data/data_T3.py
```

### 3. Run first fit
```bash
python src/main.py configs/config_nn.yaml
```
