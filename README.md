# End-to-End Calibration

**Code and experiments for *End-to-end Calibration: Theory and Empirical
Evaluation in Biomedical Context*.**

This repository accompanies a master's thesis investigating whether
incorporating a calibration objective directly into model training can
improve the reliability of predicted probabilities while maintaining
predictive performance. The work combines controlled simulation with
gene regulatory network (GRN) inference experiments.

> **Project status:** Research code accompanying a thesis. Interpret
> reported results in the context of the final thesis and each
> experiment's stated settings.

## Overview

![Overview of the experimental tracks, training and calibration
workflow, and evaluation criteria](thesis_overview.png)

The experiments are organized into three tracks:

1.  **Track 1 --- Synthetic one-dimensional problem.** A known
    posterior, (`\eta`{=tex}(x)), provides ground truth for evaluating
    calibration, refinement, and the accuracy of estimated
    probabilities.
2.  **Track 2 --- SERGIO.** Simulated single-cell RNA-sequencing data
    provide a GRN inference setting with a known underlying network and
    cell-level train/test splitting.
3.  **Track 3 --- DREAM5.** A real benchmark provides an empirical GRN
    inference case study.

Across the tracks, predictive-only training is compared with end-to-end
calibration and post-hoc calibration baselines where applicable.

## Method at a glance

The end-to-end objective combines a predictive base loss with a
calibration penalty:

$$ `\mathcal{L}`{=tex}*{`\mathrm{total}`{=tex}} =
`\mathcal{L}`{=tex}*{`\mathrm{base}`{=tex}} +
`\lambda`{=tex},`\widehat{\mathrm{CE}}`{=tex}. $$

The thesis and implementation define the precise calibration estimator
and weighting convention used in each experiment. The base loss may be
cross-entropy (CE) or Brier loss; the calibration penalty uses
differentiable/soft calibration estimators. The hyperparameter
(`\lambda`{=tex}) controls the trade-off between predictive fitting and
calibration.

Baselines include post-hoc temperature scaling, Platt scaling, isotonic
regression, and beta calibration, as well as focal-loss training.

## Evaluation

The thesis evaluates several complementary aspects of probabilistic
predictions:

-   **Calibration:** expected calibration error (ECE) and reliability
    diagrams.
-   **Calibration--refinement decomposition:** decomposition of risk
    into calibration and refinement components where applicable.
-   **Discrimination:** AUROC, AUPR, and early precision for GRN edge
    prediction.
-   **Robustness:** changes in predictive and calibration metrics under
    distribution or cross-network shift.
-   **Generalisation:** theoretical and empirical analyses, including
    PAC-Bayes/Rademacher-related discussion where applicable.

Not every metric or analysis is used in every track. See the thesis for
the exact protocol and scope of each result.

## Repository

-   [Browse the repository](https://github.com/baksho/e2ecal)

The overview image is at `thesis_overview.png`.

## Getting started

Clone the repository and inspect its current dependency and experiment
instructions:

``` bash
git clone https://github.com/baksho/e2ecal.git
cd e2ecal
```

**Reproducibility note:** Exact commands, dependency versions,
input-data paths, and expected outputs should be taken from the files in
the repository. Benchmark datasets may require separate download or
preparation; follow their applicable terms and cite the original dataset
sources.

## Reproducibility and interpretation

-   Results can depend on the data split, random seed, model capacity,
    calibration estimator, binning choices, and training budget.
-   Some controlled experiments use a known posterior, whereas real GRN
    benchmarks do not provide the true posterior for each candidate
    edge.
-   Calibration values are specific to the evaluated candidate set and
    data distribution; do not automatically interpret them as calibrated
    probabilities for all possible biological interactions.
-   Consult the final thesis for detailed protocols, limitations,
    tables, and interpretation.

## Citation

If you use this code or build on these experiments, cite the associated
thesis:

> Basak, S. (2026). *End-to-end Calibration: Theory and Empirical
> Evaluation in Biomedical Context*. Master's thesis, Technische
> Universität Braunschweig.

## License

Consult the repository's `LICENSE` file for the terms governing use,
modification, and redistribution. If no license is provided, do not
assume that the code is available for unrestricted reuse.
