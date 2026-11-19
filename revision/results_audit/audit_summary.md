# Data audit summary

* Constant (zero-variance) variables: X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X22
* Non-constant inputs: X13, X14, X15, X16, X17, X18, X19, X20, X21

## Missingness
* X20 missing: 154 rows, all with 0% fly ash (structural zero).
* Y2 missing: 364 / 616 rows; measured only for admixtures F9-2.3:1, F9-3:1, F9-4:1, K2.3:1, K3:1, K4:1, S5-2.3:1, S5-3:1, S5-4:1.

## Admixtures
* 22 admixtures, 22 distinct descriptor sets (X13-X18); X13-X18 are therefore admixture-level constants (each takes one value per admixture).

## Design
* Full factorial: 22 admixtures x 4 fly-ash levels x 7 dosages = 616 rows.
* X19 + X20 = [700.0] g in every row (binder fixed) -> X19 and X20 are perfectly collinear (one design factor: fly-ash replacement level).

## Variable meaning (R2 #2)
* X21/700x100 equals the dosage (%) encoded in the mixture ID for every row: True -> X21 is the PCE dosage (g), 0.5-2.0% by binder mass.
* X22 takes the single value [245] g -> mixing water, w/b = 0.35.

## Duplicates (R1 #18)
* Exact duplicate rows (inputs+targets): 0
* Duplicate input vectors: 0
* Near-duplicate admixture pairs (identical or almost identical descriptors and Y1 records):
  * K3:1 vs K: identical Y1 27/28, max |dY1| = 10.0 s, descriptor differences {'X13': (1076, 1080)}
  * S5-3:1 vs S5: identical Y1 27/28, max |dY1| = 1.2 s, descriptor differences {'X13': (1080, 3260), 'X15': (3.26, 1.08)}
  * F9-3:1 vs F9: identical Y1 27/28, max |dY1| = 1.0 s, descriptor differences {'X13': (1084, 1080)}

## Correlation with targets (Pearson)
|     |     Y1 |     Y2 |
|:----|-------:|-------:|
| X13 |  0.009 |  0.021 |
| X14 | -0.059 |  0.049 |
| X15 |  0     |  0.082 |
| X16 |  0.106 |  0.016 |
| X17 | -0.001 | -0.12  |
| X18 | -0.075 |  0.177 |
| X19 | -0.539 |  0.315 |
| X20 |  0.539 | -0.315 |
| X21 | -0.541 |  0.489 |
