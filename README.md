
# Deep Docking with Active Learning

## Project Description

This project implements a deep learning workflow for predicting molecular docking scores, incorporating an **active learning** strategy. The goal is to efficiently identify promising molecular candidates from a large unlabeled pool by iteratively training a model on a small labeled dataset and strategically selecting the most informative samples from the unlabeled pool to be labeled and added to the training data.  
This approach reduces the need for extensive experimental labeling while improving model performance on relevant data points (e.g., those with low docking scores).

**Use Case**  
This workflow is particularly useful in drug discovery and computational chemistry where obtaining experimental docking scores for a vast library of molecules is expensive and time‑consuming. By using active learning, we can prioritize which molecules to synthesize and test, focusing on those predicted to have strong binding affinities (low docking scores) or those where the model is most uncertain about its prediction.

## Workflow Overview

The provided Python script (`combined_docking_script.py`) consolidates the following steps, structured into logical modules:

1. **Data Handling** – Efficiently loads and processes a large compound library, including shuffling and splitting data without loading everything into memory.  
2. **Label Retrieval** – Fetches or merges docking scores (labels) for selected compounds.  
3. **Docking Score Prediction** – Trains a deep learning model to predict docking scores based on molecular features.  
4. **Metrics Calculation** – Evaluates model performance using various metrics relevant to regression and virtual screening.  
5. **Acquisition Strategy** – Implements different strategies to select the most informative compounds from the unlabeled pool for labeling.  
6. **Active Learning Orchestration** – Manages the iterative active‑learning loop, coordinating data updates, model retraining, and evaluation.  
7. **Visualization** – Provides plots to monitor training progress and model performance.

This workflow combines the principles of **MolPAL** (Molecular Pool‑based Active Learning) for intelligent sample selection with **Deep Docking’s** approach of using fast QSAR/deep‑learning models to accelerate the scoring process. citeturn0file0

## Visualizing the Workflow and Concepts

- **Figure A – Conceptual Docking:** The standard, brute‑force molecular docking process, where a large set of molecules are individually docked against a target protein to find the best binding poses and scores.  
- **Figure B – MolPAL Approach:** Shows a cycle involving prediction (surrogate model), selection (acquisition function), and docking (labeling the selected samples), which are then used to retrain the model, closing the active‑learning loop.  
- **Figure C – Deep Docking Speed‑up:** Demonstrates how Deep Docking can screen an *ultra‑large* database ~50× faster than standard docking by using a QSAR surrogate.  
- **Figure D – Molecular Docking Breakdown:** Decomposes a classical docking engine into the **search algorithm** component (e.g., Monte‑Carlo, genetic algorithm) and the **scoring function** component (e.g., force‑field, empirical, ML‑based).

## High‑Level Architecture

| Layer | What it Does | Key Classes / Files |
|-------|--------------|---------------------|
| **Data Structure** | Streams the 2.1 M‑compound library from multiple text files, shuffles on the fly, and exposes PyTorch‑style train/val/test splits that live on disk (so nothing ever has to fit in RAM). | `CompoundDataset`, `ShardedLoader` |
| **Label Retriever** | Queries your external docking pipeline (or merges pre‑computed labels) and caches results. | `DockingLabelFetcher` |
| **Surrogate Model** | Generates features (Morgan FP, RDKit descriptors, or graphs) and trains a model (MLP, Transformer, or Message‑Passing Network) with hyper‑parameter search. | `DockingRegressor` |
| **Metrics** | ROC, RMSE/MAE, Kendall‑τ, “library‑coverage” (percentage of the pool explored). | `MetricSuite` |
| **Acquisition** | Implements six strategies: Random, Greedy (μ), UCB (μ + βσ), Thompson Sampling, Expected Improvement (EI), Probability of Improvement (PI). | `AcquisitionStrategy` |
| **Orchestrator** | Runs the active‑learning loop, handles early stopping, logging (Weights & Biases), and plotting. | `DockingALRunner` |

### Why Combine MolPAL & Deep Docking?

MolPAL contributes the acquisition‑function toolbox that optimises **which** molecules we label next, while Deep Docking contributes the idea of using a fast QSAR/NN surrogate to **shortcut** classical docking. Together they let us screen libraries ≈ 50 × faster while still focusing on the most informative or highest‑affinity compounds.

## End‑to‑End Workflow

Below is the iterative active‑learning loop. Steps 3-9 have to be repeated for `N_ACTIVE_LEARNING_ITERATIONS` or until Δ‑validation‑loss < ε.

0. **Environment Setup & Dependencies**  
   ```bash
   conda create -n dds python=3.10 && conda activate dds
   pip install -r requirements.txt        # torchsparse wandb rdkit‑pypi etc.
   ```
1. **Seed Acquisition** – Randomly acquire *K* molecules to form the initial labelled dataset `(X_seed, y_seed)`.  
2. **Initial Training** – Train the `DockingRegressor` on the seed set with early stopping.  
3. **Inference** – Predict μ(x) and uncertainty σ(x) for all remaining unlabeled molecules.  
4. **Select**  – Apply an acquisition strategy (e.g., Greedy, UCB, EI) to choose *S* molecules from the pool.  
5. **Label**  – Retrieve the true docking scores for the selected molecules via your external pipeline.  
6. **Analyze**  – Compute metrics (e.g., mean MSE for the new batch) and log them.  
7. **Update** – Append `(X_new, y_new)` to the training set and optionally remove them from the pool.  
8. **Retrain**  – Warm‑start and retrain the model on the enlarged training set.  
9. **Validate**  – Evaluate on a held‑out validation split and plot loss curves.  
10. **Terminate** – Stop when the validation loss plateaus or the max number of iterations is reached.

**Suggested hyper‑parameters**  
`K = 10 000`, `S = 10 000`, `N = 15`.

## Acquisition‑Function Cheat‑Sheet (Regression)

| Name | Formula | Notes |
|------|---------|-------|
| Random | *x ∼ U(0, 1)* | Pure exploration |
| Greedy | *x = argmin μ̂(x)* | Exploitation – selects samples with the lowest predicted score |
| UCB | *argmin (μ̂(x) + β σ̂(x))* | β≈ 1–3. Balances exploration & exploitation; requires uncertainty estimate |
| Thompson Sampling | Sample *f ∼ N(μ̂(x), σ̂²(x))* and pick *argmin f(x)* | Stochastic; needs σ̂ |
| Expected Improvement (EI) | *EI(x) = γ(x) Φ(z) + σ̂(x) ϕ(z)* where *γ(x) = (f\* − μ̂(x)) / σ̂(x)* | Favors high potential for improvement; needs σ̂ |
| Probability of Improvement (PI) | *PI(x) = Φ(z)* with same γ(x) definition | More conservative than EI; needs σ̂ |

*(Φ and ϕ denote the CDF and PDF of the standard normal distribution.)*

## Practical Tips & TODOs

- **Sparse fingerprints:** Consider storing fingerprints as `torch.sparse_csr_tensor` for memory efficiency.  
- **`torchsparse` / `torchscatter`:** Useful for graph representations.  
- **Early stopping:** Use a patience of ~10 epochs.  
- **WandB logging:** Track wall‑clock time per acquisition iteration to compare strategies.  
- **Batch size:** Try acquisition sizes of 1 k, 5 k, and 10 k—larger batches don’t always accelerate discovery.

## Setup & Installation

1. **Clone or save the script** (`combined_docking_script.py`).  
2. **Install dependencies:**
   ```bash
   conda create -n dds python=3.10 && conda activate dds
   pip install pandas numpy torch scikit-learn tqdm matplotlib rdkit-pypi wandb
   ```
3. **Prepare your data:**  
   - **Labeled** data: a zip containing a CSV with a `SMILES` column and a `r_i_docking_score` column.  
   - **Unlabeled** data: a CSV with `SMILES` (and ideally `ZINCID`).  
4. **Update constants** in the script: paths, learning‑rate, batch‑size, etc.

## Running the Script

```bash
python combined_docking_script.py
```

The script:

1. Sets up logging and chooses CPU/GPU.  
2. Loads/featurizes data and builds DataLoaders.  
3. Trains the initial model and plots losses.  
4. Enters the active‑learning loop for `N_ACTIVE_LEARNING_ITERATIONS`.  
5. Logs *every* iteration (WandB is strongly recommended).  
6. Saves the best model to `best_model.pt` and writes plots to the `logs/` directory.

## Configuration & Customization

- **Acquisition function:** Pass `'greedy'`, `'mc_dropout'`, `'random'`, or `'EDL'` when constructing `DockingModelActiveLearning`.  
- **Model architecture:** Edit `DockingModel` to experiment with layers, activations, and dropout.  
- **Fingerprint parameters:** `get_morgan(radius=2, n_bits=1024)`—feel free to tweak.  
- **Multiprocessing:** `DataProcessor` uses `multiprocessing` equal to CPU cores.

## Interpreting Output

- **Logs:** Console and `logs/logfile.log`.  
- **Plots:** Training/validation losses and *true vs predicted* scatter.  
- **WandB:** Rich dashboards for every run & iteration.

## Potential Improvements

- Additional acquisition functions (e.g., Expected Model Change, QBC).  
- Alternate molecular representations (ECFP, descriptors, GNNs).  
- Containerize with Docker for reproducible deployment.  
- Robust error‑handling & input validation.  
- Smarter data splitting & CV strategies.

---
