<p align="center">
  <img src="logo.png" alt="T3Net Logo" width="200"/>
</p>


This repository implements **T3Net**, a neural network framework for non-singlet PDF inference and minimal BSM sensitivity studies. It builds on:
- The NNPDF approach to unbiased PDF determination via neural networks and Monte Carlo replicas [1][2]
- The SIMUnet methodology for embedding theory parameters into PDF fits [3]
- Bayesian Gaussian process priors for PDFs [4]. This is the core inspiration for this analysis, and should be consulted in detail to understand the origin of the methodology.

It's also important to note that this work has been forked from NNPDF, so the user can actually see the available common data and run tutorial scripts as they are updated. 

If there is any confusion about what work is unique to this study, it is entirely contained in T3_beta.py, and the original NNPDF repository lives at [https://github.com/NNPDF/](https://github.com/NNPDF/)

It would also be helpful to go through the NNPDF documentation, if you have specific questions about either ValidPhys or any of the underlying theory [https://docs.nnpdf.science/](https://docs.nnpdf.science/)

**T3Net** replaces the GP prior from [4] with a small feed-forward network and focuses on the non-singlet combination  
T3(x) = u⁺(x) − d⁺(x),  
probed by the proton–deuteron structure-function difference. We generate pseudo-data with realistic correlations, perform closure tests, and study the impact of adding a single BSM distortion parameter.

---

## Abstract

Reliable collider predictions require both unbiased parton distribution functions (PDFs) and a clear separation between proton structure and potential effects from new physics. The NNPDF framework removes functional-form bias by fitting neural networks to Monte Carlo replicas of diverse data sets [1][2], and SIMUnet embeds additional theory parameters directly into the fit to prevent genuine beyond-Standard-Model (BSM) signals from being absorbed into PDFs [3]. Candido et al. introduced a complementary Bayesian approach, using Gaussian processes as flexible priors and performing full inference over both PDF parameters and hyperparameters [4]. Their benchmarks on deep-inelastic scattering demonstrated rigorous uncertainty quantification and posterior validation. Inspired by that methodology, **T3Net** replaces the Gaussian process prior with a compact neural network and again focuses on the non-singlet combination T3(x) = u⁺ − d⁺, probed by the proton–deuteron structure-function difference. Pseudo-data are generated from this difference with realistic experimental correlations as an input for the model. Closure tests on the fits confirm that fitting only standard QCD inputs recovers a reference distribution within its uncertainty band. Introducing a single extra theory parameter to capture generic BSM distortions uncovers a bias–variance trade-off. PDF uncertainties contract, coverage degrades, and the extra parameter is systematically underestimated. These artifacts trace back to uniform regularisation across all parameters and overly rigid constraint enforcement. By isolating these effects in a minimal setting, **T3Net** investigates pitfalls in this approach and suggests avenues for future research.


---
## Code Structure and Approach

NNPDF and SIMUnet are built for large-scale, global PDF fits—frameworks with many moving parts, extensive configuration files, and multi-step workflows. For smaller, focused studies like T3Net, this complexity can obscure the core logic and make rapid prototyping or targeted analyses cumbersome.

In T3_beta.py, we deliberately condense the entire workflow into a single, self-contained script. This design serves three main goals:

1. Clarity of data flow
   All data-loading, preprocessing, model setup, training loops, and plotting routines live in one file. You can trace every variable from its origin (e.g., validphys.API calls) through to its final use (loss computation, output figures).

2. Template for custom analysis
   By stripping away global-fit infrastructure, the script becomes a clear template:
   - Fetch exactly the inputs you need (BCDMS proton/deuteron tables, FK tables, covariance matrices).
   - Build your own simple network (T3Net or T3NetWithC) and training loop.
   - Generate outputs in model_states/, results/, and images/ with no extra scaffolding.

3. Encouraging hands-on understanding
   When everything lives in one script, it’s easy to experiment:
   - Change the definition of the loss function or regularization terms.
   - Swap in different ansatz functions (K1, K2, etc.).
   - Try alternative data splits or a different BSM parameterization.
   No need to rebuild or re-install a large package—just modify the code, rerun, and inspect the results.


---

## 1. Clone & initialize

git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/assessments/projects/lc2010.git 

---

## 2a. Install with Conda

```bash
conda env create -f environment.yml  
conda activate environment_nnpdf_full  
```

This may change over time, make sure to double check any changes to NNPDF installation.

---

## 2b. Install with pip + venv
```bash
python3 -m venv .env_nnpdf  
source .env_nnpdf/bin/activate  
pip install --upgrade pip  
pip install \  
  git+https://github.com/NNPDF/nnpdf.git@4.0.10 \  
  -r requirements.txt  

```

---

## 3. Running the script

python T3_beta.py  

This will:  
- Create model_states/ and results/  
- Fetch & preprocess BCDMS data via validphys  
- Build FK tables & covariance matrices  
- Train neural nets (standard & BSM)  
- Save results to training_results.pkl  
- Generate plots in images/  


It also uses iPython, rather than a traditional notebook to ensure formatters and linters can actually work.

---

## 4. Key files

- T3_beta.py – Main workflow  
- environment.yml & requirements.txt – Env spec  
- training_results.pkl – Pickled fit results  
- images/ – Generated figures  

---

## Downloading resources (theoryID 208, PDFs, fits)

Before running `T3_beta.py`, you should explicitly download the theory and PDF sets your analysis depends on. From your shell:

```bash
# 1) Download the theory definition with ID 208
vp-get theoryID 208

# 2) Download the corresponding PDF set
vp-get pdf NNPDF40_nnlo_as_01180

# 3) (If you want to include the fit itself)
vp-get fit NNPDF40_nlo_as_01180
```

For reproducing the results of this study, this should suffice. Note this should be done after the environment is created.
---

## References

1. R. D. Ball et al., “An open-source machine learning framework for global analyses of parton distributions,” Eur. Phys. J. C 81 (2021) 958, doi:10.1140/epjc/s10052-021-09747-9  
2. R. D. Ball et al., “The path to proton structure at 1 % accuracy: NNPDF Collaboration,” Eur. Phys. J. C 82 (2022) 10328, doi:10.1140/epjc/s10052-022-10328-7  
3. S. Iranipour and M. Ubiali, “A new generation of simultaneous fits to LHC data using deep learning,” JHEP 05 (2022) 032, doi:10.1007/JHEP05(2022)032  
4. A. Candido et al., “Bayesian inference with Gaussian processes for the determination of parton distribution functions,” Eur. Phys. J. C 84 (2024) 716, doi:10.1140/epjc/s10052-024-13100-1  


## AI Declaration


Portions of this repository—including the README, figures, in-code comments and this notebook—were drafted and refined with the assistance of AI tools. In particular, AI support was used to:

- Generate and polish all plotting routines (including LaTeX‐style labels and publication-ready layouts).  
- Interpret and streamline ValidPhys API calls for data loading, covariance assembly and FK-table construction.  
- Propose inline documentation, helpful guides for the README, code structure outlines and early drafts of the data pipeline and training loops.  
- Ensure consistent UK-English phrasing, formatting of markdown cells, and structuring of the README and notebook headers.

Every AI-generated suggestion has been personally reviewed, edited, tested and verified by the author to guarantee full scientific accuracy, reproducibility and clarity. All code and results presented here reflect definitive, author-approved iterations.

