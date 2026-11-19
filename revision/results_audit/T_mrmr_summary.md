# mRMR scores (F-test relevance / mean |Pearson| redundancy / FCQ score)

## Y1 (n = 616)
| feature   | name                             | zero_variance   |   relevance_F |   relevance_p |   rank_relevance_F |   relevance_MI |   rank_MI |   redundancy_at_selection |   mrmr_score_at_selection |   mrmr_rank | selected_top9   | note                                      |
|:----------|:---------------------------------|:----------------|--------------:|--------------:|-------------------:|---------------:|----------:|--------------------------:|--------------------------:|------------:|:----------------|:------------------------------------------|
| X21       | PCE dosage                       | False           |      254.27   |        0      |                  1 |         0.4537 |         3 |                  nan      |                  254.27   |           1 | True            |                                           |
| X19       | Cement                           | False           |      251.326  |        0      |                  2 |         0.7396 |         2 |                    0.001  |               251325      |           2 | True            |                                           |
| X16       | PCE Mw                           | False           |        6.9692 |        0.0085 |                  4 |         0.3016 |         6 |                    0.001  |                 6969.19   |           3 | True            |                                           |
| X20       | Fly ash                          | False           |      251.326  |        0      |                  3 |         0.746  |         1 |                    0.334  |                  752.471  |           4 | True            |                                           |
| X18       | PCE viscosity                    | False           |        3.492  |        0.0621 |                  5 |         0.2792 |         7 |                    0.0697 |                   50.1069 |           5 | True            |                                           |
| X14       | PCE solid content                | False           |        2.1102 |        0.1468 |                  6 |         0.3193 |         4 |                    0.1615 |                   13.0639 |           6 | True            |                                           |
| X13       | PCE density                      | False           |        0.0457 |        0.8308 |                  7 |         0.0727 |         8 |                    0.1349 |                    0.3386 |           7 | True            |                                           |
| X17       | PCE PDI                          | False           |        0.0009 |        0.9761 |                  8 |         0.0725 |         9 |                    0.0424 |                    0.0212 |           8 | True            |                                           |
| X15       | PCE pH                           | False           |        0      |        0.9986 |                  9 |         0.3172 |         5 |                    0.2008 |                    0      |           9 | True            |                                           |
| X1        | C3S                              | True            |      nan      |      nan      |                nan |       nan      |       nan |                  nan      |                  nan      |         nan | False           | zero variance (constant) - not selectable |
| X10       | CaO (fly ash)                    | True            |      nan      |      nan      |                nan |       nan      |       nan |                  nan      |                  nan      |         nan | False           | zero variance (constant) - not selectable |
| X11       | Specific surface area of fly ash | True            |      nan      |      nan      |                nan |       nan      |       nan |                  nan      |                  nan      |         nan | False           | zero variance (constant) - not selectable |
| X12       | Specific gravity of fly ash      | True            |      nan      |      nan      |                nan |       nan      |       nan |                  nan      |                  nan      |         nan | False           | zero variance (constant) - not selectable |
| X2        | C2S                              | True            |      nan      |      nan      |                nan |       nan      |       nan |                  nan      |                  nan      |         nan | False           | zero variance (constant) - not selectable |
| X22       | Water                            | True            |      nan      |      nan      |                nan |       nan      |       nan |                  nan      |                  nan      |         nan | False           | zero variance (constant) - not selectable |
| X3        | C3A                              | True            |      nan      |      nan      |                nan |       nan      |       nan |                  nan      |                  nan      |         nan | False           | zero variance (constant) - not selectable |
| X4        | C4AF                             | True            |      nan      |      nan      |                nan |       nan      |       nan |                  nan      |                  nan      |         nan | False           | zero variance (constant) - not selectable |
| X5        | Specific surface area of cement  | True            |      nan      |      nan      |                nan |       nan      |       nan |                  nan      |                  nan      |         nan | False           | zero variance (constant) - not selectable |
| X6        | Specific gravity of cement       | True            |      nan      |      nan      |                nan |       nan      |       nan |                  nan      |                  nan      |         nan | False           | zero variance (constant) - not selectable |
| X7        | SiO2 (fly ash)                   | True            |      nan      |      nan      |                nan |       nan      |       nan |                  nan      |                  nan      |         nan | False           | zero variance (constant) - not selectable |
| X8        | Al2O3 (fly ash)                  | True            |      nan      |      nan      |                nan |       nan      |       nan |                  nan      |                  nan      |         nan | False           | zero variance (constant) - not selectable |
| X9        | Fe2O3 (fly ash)                  | True            |      nan      |      nan      |                nan |       nan      |       nan |                  nan      |                  nan      |         nan | False           | zero variance (constant) - not selectable |

Stability over outer training folds:
| feature   |   mean_rank |   min_rank |   max_rank |   selected_in_top9_% |
|:----------|------------:|-----------:|-----------:|---------------------:|
| X21       |        1.47 |          1 |          2 |                  100 |
| X19       |        2    |          1 |          4 |                  100 |
| X20       |        3    |          1 |          4 |                  100 |
| X16       |        4.27 |          3 |          6 |                  100 |
| X18       |        4.53 |          3 |          6 |                  100 |
| X14       |        6    |          3 |          9 |                  100 |
| X15       |        7.8  |          6 |          9 |                  100 |
| X17       |        7.87 |          7 |          9 |                  100 |
| X13       |        8.07 |          7 |          9 |                  100 |
| X1        |      nan    |        nan |        nan |                    0 |
| X2        |      nan    |        nan |        nan |                    0 |
| X3        |      nan    |        nan |        nan |                    0 |
| X4        |      nan    |        nan |        nan |                    0 |
| X5        |      nan    |        nan |        nan |                    0 |
| X6        |      nan    |        nan |        nan |                    0 |
| X7        |      nan    |        nan |        nan |                    0 |
| X8        |      nan    |        nan |        nan |                    0 |
| X9        |      nan    |        nan |        nan |                    0 |
| X10       |      nan    |        nan |        nan |                    0 |
| X11       |      nan    |        nan |        nan |                    0 |
| X12       |      nan    |        nan |        nan |                    0 |
| X22       |      nan    |        nan |        nan |                    0 |

## Y2 (n = 252)
| feature   | name                             | zero_variance   |   relevance_F |   relevance_p |   rank_relevance_F |   relevance_MI |   rank_MI |   redundancy_at_selection |   mrmr_score_at_selection |   mrmr_rank | selected_top9   | note                                      |
|:----------|:---------------------------------|:----------------|--------------:|--------------:|-------------------:|---------------:|----------:|--------------------------:|--------------------------:|------------:|:----------------|:------------------------------------------|
| X21       | PCE dosage                       | False           |       78.3677 |        0      |                  1 |         0.2228 |         1 |                  nan      |                   78.3677 |           1 | True            |                                           |
| X19       | Cement                           | False           |       27.5418 |        0      |                  2 |         0.1441 |         8 |                    0.001  |                27541.8    |           2 | True            |                                           |
| X18       | PCE viscosity                    | False           |        8.1154 |        0.0048 |                  4 |         0.1982 |         2 |                    0.001  |                 8115.36   |           3 | True            |                                           |
| X20       | Fly ash                          | False           |       27.5418 |        0      |                  3 |         0.1538 |         6 |                    0.334  |                   82.4605 |           4 | True            |                                           |
| X17       | PCE PDI                          | False           |        3.6291 |        0.0579 |                  5 |         0.0955 |         9 |                    0.0888 |                   40.8846 |           5 | True            |                                           |
| X14       | PCE solid content                | False           |        0.6026 |        0.4383 |                  7 |         0.17   |         5 |                    0.0181 |                   33.2212 |           6 | True            |                                           |
| X15       | PCE pH                           | False           |        1.7118 |        0.1919 |                  6 |         0.185  |         3 |                    0.1218 |                   14.0566 |           7 | True            |                                           |
| X13       | PCE density                      | False           |        0.1153 |        0.7344 |                  8 |         0.1495 |         7 |                    0.2418 |                    0.4769 |           8 | True            |                                           |
| X16       | PCE Mw                           | False           |        0.0674 |        0.7953 |                  9 |         0.1842 |         4 |                    0.2155 |                    0.3129 |           9 | True            |                                           |
| X1        | C3S                              | True            |      nan      |      nan      |                nan |       nan      |       nan |                  nan      |                  nan      |         nan | False           | zero variance (constant) - not selectable |
| X10       | CaO (fly ash)                    | True            |      nan      |      nan      |                nan |       nan      |       nan |                  nan      |                  nan      |         nan | False           | zero variance (constant) - not selectable |
| X11       | Specific surface area of fly ash | True            |      nan      |      nan      |                nan |       nan      |       nan |                  nan      |                  nan      |         nan | False           | zero variance (constant) - not selectable |
| X12       | Specific gravity of fly ash      | True            |      nan      |      nan      |                nan |       nan      |       nan |                  nan      |                  nan      |         nan | False           | zero variance (constant) - not selectable |
| X2        | C2S                              | True            |      nan      |      nan      |                nan |       nan      |       nan |                  nan      |                  nan      |         nan | False           | zero variance (constant) - not selectable |
| X22       | Water                            | True            |      nan      |      nan      |                nan |       nan      |       nan |                  nan      |                  nan      |         nan | False           | zero variance (constant) - not selectable |
| X3        | C3A                              | True            |      nan      |      nan      |                nan |       nan      |       nan |                  nan      |                  nan      |         nan | False           | zero variance (constant) - not selectable |
| X4        | C4AF                             | True            |      nan      |      nan      |                nan |       nan      |       nan |                  nan      |                  nan      |         nan | False           | zero variance (constant) - not selectable |
| X5        | Specific surface area of cement  | True            |      nan      |      nan      |                nan |       nan      |       nan |                  nan      |                  nan      |         nan | False           | zero variance (constant) - not selectable |
| X6        | Specific gravity of cement       | True            |      nan      |      nan      |                nan |       nan      |       nan |                  nan      |                  nan      |         nan | False           | zero variance (constant) - not selectable |
| X7        | SiO2 (fly ash)                   | True            |      nan      |      nan      |                nan |       nan      |       nan |                  nan      |                  nan      |         nan | False           | zero variance (constant) - not selectable |
| X8        | Al2O3 (fly ash)                  | True            |      nan      |      nan      |                nan |       nan      |       nan |                  nan      |                  nan      |         nan | False           | zero variance (constant) - not selectable |
| X9        | Fe2O3 (fly ash)                  | True            |      nan      |      nan      |                nan |       nan      |       nan |                  nan      |                  nan      |         nan | False           | zero variance (constant) - not selectable |

Stability over outer training folds:
| feature   |   mean_rank |   min_rank |   max_rank |   selected_in_top9_% |
|:----------|------------:|-----------:|-----------:|---------------------:|
| X21       |        1    |          1 |          1 |                  100 |
| X20       |        3.13 |          2 |          5 |                  100 |
| X19       |        3.27 |          2 |          5 |                  100 |
| X18       |        3.4  |          3 |          6 |                  100 |
| X15       |        5.6  |          2 |          8 |                  100 |
| X17       |        5.6  |          3 |          7 |                  100 |
| X14       |        6.33 |          4 |          9 |                  100 |
| X16       |        8.27 |          7 |          9 |                  100 |
| X13       |        8.4  |          6 |          9 |                  100 |
| X1        |      nan    |        nan |        nan |                    0 |
| X2        |      nan    |        nan |        nan |                    0 |
| X3        |      nan    |        nan |        nan |                    0 |
| X4        |      nan    |        nan |        nan |                    0 |
| X5        |      nan    |        nan |        nan |                    0 |
| X6        |      nan    |        nan |        nan |                    0 |
| X7        |      nan    |        nan |        nan |                    0 |
| X8        |      nan    |        nan |        nan |                    0 |
| X9        |      nan    |        nan |        nan |                    0 |
| X10       |      nan    |        nan |        nan |                    0 |
| X11       |      nan    |        nan |        nan |                    0 |
| X12       |      nan    |        nan |        nan |                    0 |
| X22       |      nan    |        nan |        nan |                    0 |
