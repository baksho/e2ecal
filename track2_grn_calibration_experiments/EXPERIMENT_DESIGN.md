# Experiment design for Task 4 and the optional task

## Primary question
In GRN inference, does adding an end-to-end calibration penalty improve the trustworthiness of predicted edge probabilities while retaining ranking performance?

## Minimal comparison
- Same small Transformer architecture in every condition.
- Predictive-only BCE (`lambda=1`).
- End-to-end calibration-aware objective (`lambda=0.5`, `lambda=0.1`).
- Post-hoc temperature scaling fitted only on validation predictions.
- Metrics: AUROC, AUPR, Brier score, ECE (10 bins); report prevalence and candidate sampling details.

## Data tracks
**SERGIO:** 20-gene, one-bin, 120-cell simulator run, known directed edge labels. Small enough for Colab. Pair examples are not independent merely because cells were simulated independently; pairs share genes.

**DREAM5:** Network 1 first, then another network if available. The challenge's official gold-standard data are required. Candidate negatives are sampled for runtime, so reported AUPR/calibration refer to that sampled candidate set, not official leaderboard evaluation.

## Optional task
**Stage A, controlled shift:** use the Track 1 known posterior, keep `P(Y|X)` fixed, alter the marginal distribution of X, and compare calibration/predictive metrics in- vs out-of-distribution.

**Stage B, DREAM5 cross-network:** train on one network and evaluate on another only when both expression and gold-standard files are available. This is a realistic domain-shift case, but not necessarily pure covariate shift because the biological mechanism and class prevalence can change.

## What is deliberately omitted
The detailed original Track 2/Track 3 grid (multiple calibration estimators, broad capacity sweeps, many seeds, numerous secondary metrics) is omitted to keep the Colab experiment focused. The first implementation is a feasibility experiment; repeat across seeds and add TF-only candidate restriction before making thesis-level claims.
