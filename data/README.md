# Dataset Description

> **Mandatory citation:** Any use of this dataset, in whole or in part, requires citing the main article:
> *Equilibrium Optimizer-Tuned Machine Learning Models for Predicting Fresh-State Properties of PCE-Modified Cement-Fly Ash Systems: A SHAP-Based Interpretation* (İ. Ayaz, S. Ekinci, D. Izci, M. Abdel-Salam, M. İ. Şahin, H. Çoban, S. Özteber, A. Mardani).
> License: [CC BY-NC 4.0](../LICENSE) — commercial use is not permitted.

This dataset was obtained from a comprehensive experimental program conducted to investigate the effects of mixture components, such as the amounts of cement, mineral admixture, aggregate, water and water-reducing admixture, on the flow time and mini-slump value of cementitious mixtures.

The experimental approach is based on standardized laboratory tests performed on cementitious systems with different material compositions. All mixtures were evaluated in terms of flow time and mini-slump values. Material amounts reported in the study are given in grams `(g)`.

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
| X21 | Water | Amount of water |
| X22 | Water-Reducing Admixture | Amount of admixture |

## Output Variables

The following measured values of the different mixtures were defined as the output `(dependent)` variables, representing the primary performance indicators:

| Abbreviation | Description |
|---|---|
| Y1 | Marsh-funnel flow time |
| Y2 | Mini-slump value |

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

A dash (`-`) denotes a missing value.
