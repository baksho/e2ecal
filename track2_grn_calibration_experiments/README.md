# Lightweight GRN inference and calibration experiments

## Scope
This project is a compact empirical extension of Track 1 to gene-regulatory-network (GRN) inference. It covers:
1. **Task 4:** compare predictive-only training with end-to-end calibration-aware training for probabilistic directed-edge prediction, while reporting ranking metrics.
2. **SERGIO:** a small synthetic single-cell expression dataset with known simulated edges.
3. **DREAM5:** a real expression compendium with challenge gold-standard labels.
4. **Optional task:** a controlled covariate-shift experiment with known conditional posterior, followed by an optional DREAM5 cross-network transfer analysis.

The Transformer is intentionally small for Colab. It encodes each ordered TF-target pair as a sequence of expression values across conditions/cells. It has no positional encoding because condition order is not a meaningful time axis in the DREAM5 static compendium.

## Notebook order
1. `GRN_00_Setup_and_Utilities.ipynb` — install dependencies and smoke-test utilities.
2. `GRN_01_SERGIO_Simulate_and_Prepare.ipynb` — run SERGIO and save a pair dataset.
3. `GRN_02_DREAM5_Load_and_Prepare.ipynb` — upload a DREAM5 expression matrix and matching gold standard; save a pair dataset.
4. `GRN_03_Train_Transformer_and_Compare.ipynb` — train/evaluate the Transformer on whichever prepared datasets are present.
5. `GRN_04_OptionalTask_Controlled_CovariateShift.ipynb` — controlled shift with the Track 1 known-posterior setup.
6. `GRN_05_OptionalTask_DREAM5_CrossNetwork.ipynb` — optional cross-network check after preparing two different DREAM5 networks.

## Running in Google Colab
- Download and extract the complete ZIP.
- Upload the extracted folder to Google Drive, or upload the notebooks and `grn_experiment_utils.py` together to Colab.
- Open notebooks in order. If using Drive, set `PROJECT_DIR` in each notebook to the folder where the helper file and `data/` directory live.
- For DREAM5, download the official challenge data from https://zenodo.org/records/17854236 or https://www.synapse.org/Synapse:syn2787209 and upload the matching expression and gold-standard TSV files in Notebook 2.
- Use a GPU runtime if available, though the default model is small enough to run on CPU. Start with the default 30 epochs and 12,000 pair cap.

## Model objective
The thesis convention is retained:
`L_lambda = (1 - lambda) * calibration_penalty + lambda * base_loss`, where base_loss is BCE or Brier.
Thus `lambda=1` is the predictive-only BCE baseline, and smaller values place more weight on the calibration penalty. The smooth binned penalty is an experimental surrogate, not an unbiased estimator of canonical calibration error.

## Important limitations
- Pair examples share genes and expression profiles. Random pair splits do not establish i.i.d. sampling and may permit gene-specific information to appear across splits.
- DREAM5 results use a sampled negative candidate set for tractability. AUPR and calibration depend on the sampled prevalence and should not be compared directly to official DREAM5 leaderboard values.
- The first DREAM5 preparation notebook does not restrict candidate regulators to the official TF list unless that list is added. Report this clearly or adapt the notebook to use the TF list.
- The small SERGIO run is a feasibility experiment, not a comprehensive benchmark. Repeat multiple seeds before making stability claims.
- Cross-network DREAM5 transfer is not guaranteed to be pure covariate shift; regulatory mechanisms and class prevalence can differ. Treat it as a descriptive domain-shift analysis.
- A Transformer attention pattern does not prove causality. GRN labels benchmark edge prediction; causal interpretation requires additional assumptions/evidence.
- The SERGIO upstream repository is older software. The notebook attempts to use the original API and will stop with an error if upstream compatibility changes; it does not silently substitute a different simulator.

## Suggested thesis reporting
Report AUROC, AUPR, Brier score, ECE (10 bins), class prevalence, pair-sampling protocol, number of genes/conditions, architecture size, selected lambda, and the split design. Use validation data for model selection and post-hoc temperature fitting; never select a method by its test result. Repeat at least five random seeds before making comparative claims.

## References
- Dibaeinia, P. & Sinha, S. (2020). SERGIO: a single-cell expression simulator guided by gene regulatory networks. *Cell Systems*, 11(3), 252–271. https://doi.org/10.1016/j.cels.2020.08.003
- Marbach, D. et al. (2012). Wisdom of crowds for robust gene network inference. *Nature Methods*, 9, 796–804. https://doi.org/10.1038/nmeth.2016
- SERGIO original implementation: https://github.com/PayamDiba/SERGIO
- DREAM5 challenge data: https://zenodo.org/records/17854236
