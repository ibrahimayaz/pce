# Dataset Description

> **Mandatory citation:** Any use of this dataset, in whole or in part, requires citing the main article:
> *Predicting Marsh Funnel Flow Time and Mini-Slump of Fly Ash–Cement Pastes from PCE Superplasticizer Descriptors: Equilibrium Optimizer-Tuned Models under Nested and Leave-One-Admixture-Out Validation with SHAP Interpretation* (İ. Ayaz, S. Ekinci, D. Izci, M. Abdel-Salam, M. İ. Şahin, H. Çoban, S. Özteber, A. Mardani).
> License: [CC BY-NC 4.0](../LICENSE) — commercial use is not permitted.
> The dataset is also archived on IEEE DataPort: doi: [10.21227/fjdp-ss56](https://dx.doi.org/10.21227/fjdp-ss56).

This dataset was obtained from an experimental program on the fresh-state compatibility of fly ash–cement pastes with polycarboxylate ether (PCE) superplasticizers. The program followed a full-factorial design: 22 PCEs × 4 fly-ash replacement levels (0, 15, 30 and 45 % by mass of binder) × 7 PCE dosages (0.5, 0.75, 1.0, 1.25, 1.5, 1.75 and 2.0 % by mass of binder) = 616 pastes. One portland cement and one fly ash were used, the binder mass was 700 g and the mixing water was 245 g (w/b = 0.35).

The experimental approach is based on standardized laboratory tests performed on cementitious systems with different material compositions. Material amounts are given in grams `(g)`. The mixture identifier (`index`) encodes the fly-ash level, the admixture and the PCE dosage, e.g. `UK15-K4:1-1,0` = 15 % fly ash, admixture K4:1, 1.0 % PCE.

## Input Variables

The main input parameters considered in this study are as follows:

| Abbreviation | Variable | Description |
|---|---|---|
| X1 | C<sub>3</sub>S | C<sub>3</sub>S content of the cement |
| X2 | C<sub>2</sub>S | C<sub>2</sub>S content of the cement |
| X3 | C<sub>3</sub>A | C<sub>3</sub>A content of the cement |
| X4 | C<sub>4</sub>AF | C<sub>4</sub>AF content of the cement |
| X5 | SSAC | Specific surface area of the cement |
| X6 | SGC | Specific gravity of the cement |
| X7 | SiO<sub>2</sub> | SiO<sub>2</sub> content of the fly ash |
| X8 | Al<sub>2</sub>O<sub>3</sub> | Al<sub>2</sub>O<sub>3</sub> content of the fly ash |
| X9 | Fe<sub>2</sub>O<sub>3</sub> | Fe<sub>2</sub>O<sub>3</sub> content of the fly ash |
| X10 | CaO | CaO content of the fly ash |
| X11 | SSAF | Specific surface area of the fly ash |
| X12 | SGF | Specific gravity of the fly ash |
| X13 | Density | Density of the water-reducing admixture |
| X14 | Solid Content | Solid content of the water-reducing admixture |
| X15 | pH | pH of the water-reducing admixture |
| X16 | M<sub>w</sub> | Weight-average molecular weight of the water-reducing admixture |
| X17 | PDI | Polydispersity index of the water-reducing admixture |
| X18 | Viscosity | Viscosity of the water-reducing admixture |
| X19 | Cement | Amount of cement |
| X20 | Fly Ash | Amount of fly ash |
| X21 | PCE dosage | Amount of PCE admixture (3.5–14 g = 0.5–2.0 % of the 700 g binder) |
| X22 | Water | Amount of mixing water (245 g in every row, w/b = 0.35) |

## Output Variables

The following measured values of the different mixtures were defined as the output `(dependent)` variables, representing the primary performance indicators:

| Abbreviation | Description |
|---|---|
| Y1 | Marsh funnel flow time (s); measured for all 616 pastes |
| Y2 | Mini-slump spread (cm); measured for 252 pastes (all 28 pastes of 9 admixtures) |

These output variables are predicted from the input parameters described above.

## Purpose of the Dataset

This dataset can be used to investigate the relationship between material properties and the fresh-state behavior of cementitious systems. In particular, it serves the following purposes:

- Prediction of flow time
- Prediction of mini-slump
- Analysis of the effect of material composition on performance
- Development of regression and machine learning-based prediction models

## File Format

`dataset.csv` contains 616 mixtures and 25 columns:

- `index`: mixture identifier
- `X1`–`X22`: input variables
- `Y1`, `Y2`: output variables

A dash (`-`) has two meanings:

- in **X20**: mixture without fly ash (structural zero; use 0 g);
- in **Y2**: mini-slump not measured. Y2 is missing block-wise for all pastes of 13 admixtures and should not be imputed.

Because one cement, one fly ash and a fixed water content were used, X1–X12 and X22 are constant; X19 + X20 = 700 g in every row. Each admixture has its own combination of X13–X18 (22 distinct descriptor sets).

> **Note.** In the first version of this repository X21 was labelled as water and X22 as admixture amount. The labels above are the corrected ones (X21 equals the dosage encoded in the mixture identifier for all 616 rows).
