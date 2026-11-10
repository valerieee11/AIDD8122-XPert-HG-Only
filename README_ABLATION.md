# XPert-HG-only: exploratory Pearson-loss ablation

## Purpose
Test whether adding per-sample Pearson correlation supervision across 978 L1000 genes improves drug-induced expression-change prediction beyond the HG-only MSE-only course baseline. This is **not** a reproduction of the full UniMol-enabled XPert.

## Conditions
- **Experiment A (MSE-only):** MSE(delta_pred, delta_true) + 0.2 MSE(treated_direct, treated_true) + 0.1 MSE(control_direct, control_true). Run in Google Colab (Tesla T4, PyTorch 2.14.1).
- **Experiment B (Pearson loss):** loss A + 0.1 × [1 − PCC(delta_pred, delta_true)]; PCC computed per sample across 978 genes. Run in Kaggle (Tesla T4, PyTorch 2.11.0+cu130).
- Both use public processed L1000 SDST; 1,000 train / 200 validation / 200 test observations; seed 2026; batch size 1; Adam learning rate 1e−4; 10 epochs. Best checkpoint selected using validation MSE.
- Identical held-out observations and control expression were verified for A and B.

## Held-out test results

| Model | MSE ↓ | RMSE ↓ | MAE ↓ | Mean Pearson delta ↑ | Mean abs predicted delta |
|---|---:|---:|---:|---:|---:|
| No-change | 0.519923 | 0.721057 | 0.442325 | undefined | 0.000000 |
| A: MSE-only | 0.510133 | 0.714236 | 0.445866 | 0.131860 | 0.074675 |
| B: + Pearson | 0.506850 | 0.711934 | 0.445918 | 0.154565 | 0.085900 |

Adding the correlation loss decreased test MSE by **0.003282 (0.6434%)** compared with A, and increased mean Pearson delta by **0.022705**. MAE increased slightly. B's amplitude prediction (0.085900) remained much smaller than observed expression-change amplitude (0.442325).

Best saved B validation MSE: **0.488104**, epoch **10**, from the current saved training artifact.

## Paired drug-cluster bootstrap (exploratory)
- 200 held-out records, 196 unique drug identifiers, 5,000 resamples (seed 2026).
- Test MSE(A) − MSE(B) = **0.003282**; percentile 95% CI **[0.002618, 0.004012]**.
- The resample-level distribution was **not contained in the submitted final ZIP**; the interval is transcribed from the Kaggle output and recorded in `ablation_bootstrap_summary.csv`.

## Caveats
The improvement is exploratory: A and B were trained on different hosted runtime versions, only one seed was trained per condition, and the sampled split is not strictly cold-drug. Bootstrap over test-drug IDs does not quantify training variability. Both use HG-only features and simplified losses, not original full XPert.

## Artifacts
- `ablation_comparison.csv`: all metrics
- `training_history_pearson.csv`: Experiment B training
- `training_summary.json`: configuration and saved best checkpoint summary
- `ablation_bootstrap_summary.csv`: user-recorded paired cluster-bootstrap summary
- `figure_ablation_test_metrics.png`: error and response-fidelity plot
- `figure_ablation_validation_bootstrap.png`: validation curve and confidence interval

GitHub project: https://github.com/valerieee11/AIDD8122-XPert-HG-Only
