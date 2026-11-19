# Reviewer concern → script → output map

All outputs are produced on the training workstation by `run_all.bat` / `run_all.sh`.
**The manuscript revision (highlighted and clean versions) and the response to reviewers are written
afterwards on the author's computer, not on the training workstation.** Smoke-test results
(`results_smoke/`) are never used in the manuscript.

Paths: `audit/` = `results_audit/`; `A/` = `results_full/budget_*/analysis/`;
`F/` = `results_full/budget_*/final_analysis/`; `H/` = `results_full/budget_*/hyperparam_sensitivity/`.

## Reviewer 1

| # | Concern (short) | Script | Output used in the response / manuscript |
|---|---|---|---|
| 1 | Title not informative | — | Text only (new title) |
| 5 | Highlight difficulties and contributions | — | Text only (abstract, introduction); facts from `audit/audit_summary.md` |
| 6 | Compare with similar studies | — | Text and comparison table built from the literature + `A/T_nested_rkf_*.csv` |
| 7 | More explanation of results | `02_aggregate.py`, `03_final_analysis.py` | `A/T_residuals_*.csv`, `A/F_residuals_*`, `A/T_lopo_pergroup_*.csv`, `F/T_shap_grouped_*.csv`, `F/F_shap_dependence_*` |
| 8 | Methodology flowchart | `05_flowchart.py` | `audit/F_methodology_flowchart.png/pdf` |
| 9 | Parameter rationale, other values, sensitivity | `run_experiments.py` (block `sens`), `04_hyperparam_sensitivity.py`, `01_mrmr_scores.py` | `A/T_eo_sensitivity_*.csv`, `A/F_eo_sensitivity_*`, `H/T_hp_sensitivity_summary.csv`, `H/F_hp_sensitivity_*`, `audit/T_mrmr_stability_*.csv` |
| 10 | mRMR relevance/redundancy scores and ranks | `01_mrmr_scores.py` | `audit/T_mrmr_Y1.csv`, `audit/T_mrmr_Y2.csv`, `audit/T_mrmr_summary.md` |
| 11 | Why X22 excluded? | `00_data_audit.py` | `audit/T_descriptive_stats.csv` (X22 constant = 245 g water), `audit/T_X21_X22_check.csv` |
| 12 | Mixture design without PCE dosage? | `00_data_audit.py` | `audit/T_X21_X22_check.csv` (X21 = PCE dosage, is an input); GUI (`app.py`) |
| 14 | Preprocessing inside CV folds? | `rev/features.py`, `rev/tuners.py`, `rev/jobs.py` | Method text; all `A/` tables are nested |
| 15 | Nested CV / independent test | block `rkf_main` | `A/T_nested_rkf_*.csv`, `A/F_actual_vs_pred_rkf_*` |
| 17 | Group / leave-PCE-out validation | blocks `lopo`, `lofa` | `A/T_lopo_*.csv`, `A/T_lopo_pergroup_*.csv`, `A/F_lopo_pergroup_*`, `A/T_lofa_*.csv` |
| 18 | Duplicates / near-duplicates | `00_data_audit.py`, blocks `nodup_*` | `audit/T_duplicates.csv`, `A/T_nodup_Y1.csv` |
| 19 | Descriptors = PCE identifiers? | `00_data_audit.py`, blocks `lopo`, `base_pceid`, `base_mixonly*` | `audit/T_pce_descriptors.csv`, `A/T_baselines_*.csv`, `A/T_lopo_*.csv` |
| 20 | Learning curve for Y1; "300 samples" | `03_final_analysis.py` | `F/F_learning_curve_Y1/Y2`, `F/T_learning_curve_*.csv` |
| 21 | Complete Table 4 | block `final`, `02_aggregate.py` | `A/T4_hyperparameters.csv` |
| 22 | EO settings | `02_aggregate.py` | `A/T_eo_settings.csv` |
| 23 | Objective function | `rev/tuners.py` | `A/T_eo_settings.csv` (mean inner 3-fold MSE) |
| 24 | EO vs random search / Bayesian | block `rkf_tuners` | `A/T_tuner_comparison_*.csv`, `A/F_tuner_convergence_*` |
| 25 | Descriptive statistics (original units) | `00_data_audit.py` | `audit/T_descriptive_stats.csv` |
| 26 | Missing proportions and pattern | `00_data_audit.py` | `audit/T_missingness.csv`, `audit/T_missing_pattern.csv` |
| 27 | Number of MICE imputations | `rev/data.py`, block `legacy_mice` | No imputation in the revised pipeline; `A/T_legacy_mice_*.csv` shows the effect of the original single global imputation |
| 30 | Exact mean \|SHAP\| and ranks | `03_final_analysis.py` | `F/T_shap_Y1.csv`, `F/T_shap_Y2.csv`, `F/T_shap_grouped_*.csv` |
| 31 | Dependence plots, correlation matrix | `03_final_analysis.py`, `00_data_audit.py` | `F/F_shap_dependence_*`, `audit/F_correlation_matrix`, `audit/T_correlation_*.csv` |
| 32 | How SHAP was computed | `03_final_analysis.py` | `F/shap_settings_*.json` |
| 33 | X13 absorbing dosage/type effects? | `03_final_analysis.py`, baselines | `F/T_shap_interactions_*.csv`, `F/F_shap_X13_vs_X21_*`, `F/shap_settings_*.json` (corr X13–X21), `A/T_baselines_*.csv` |
| 38 | Figure 6 separate panels + values | `03_final_analysis.py` | `F/F_shap_bar_*`, `F/F_shap_beeswarm_*` (600 dpi, PNG + PDF), `F/T_shap_*.csv` |
| 41 | DOI / archive, GUI URL | `deploy/` | `deploy/README_DEPLOY.md` (IEEE DataPort DOI, GitHub repository, Hugging Face Space) |

## Reviewer 2

| # | Concern (short) | Script | Output |
|---|---|---|---|
| 1 | Y2 mostly imputed; imputation of targets; PMM claim | `rev/data.py`, `00_data_audit.py`, block `legacy_mice` | `audit/T_missingness.csv`, `A/T_legacy_mice_Y2.csv` (all rows vs measured rows vs imputed rows, imputed range), `A/T_nested_rkf_Y2.csv` (measured only) |
| 2 | X21 is PCE dosage, X22 water | `00_data_audit.py`, `rev/data.py`, `app.py` | `audit/T_X21_X22_check.csv`, `audit/audit_summary.md`; new Y1 interpretation from `F/T_shap_Y1.csv`, `F/T_shap_grouped_Y1.csv` |
| 4 | Optimistic validation; hold out whole PCEs | blocks `rkf_main`, `lopo`, `lofa` | `A/T_nested_rkf_*.csv`, `A/T_lopo_*.csv`, `A/T_lofa_*.csv` |
| 5 | Table 4 lacks optimum values; CatBoost range | block `final` | `A/T4_hyperparameters.csv` (ranges from code; integer rounding fixed) |
| 8 | CatBoost vs LightGBM overlap; repeated CV + paired test; default/random search | blocks `rkf_main`, `rkf_tuners` | `A/T_model_tests_*.csv`, `A/T_tuner_comparison_*.csv` |
