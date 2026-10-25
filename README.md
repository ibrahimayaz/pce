# PCE Fresh-State Property Predictor

Web-based GUI and pre-trained models for predicting the fresh-state properties of
polycarboxylate-ether (PCE) modified cement–fly ash systems:

| Target | Property | Model |
|---|---|---|
| **Y1** | Marsh-funnel flow time | CatBoost (EO-tuned) |
| **Y2** | Mini-slump value | LightGBM (EO-tuned) |

This repository accompanies the article:

> **Equilibrium Optimizer-Tuned Machine Learning Models for Predicting Fresh-State Properties of PCE-Modified Cement-Fly Ash Systems: A SHAP-Based Interpretation**  
> *İbrahim Ayaz, Serdar Ekinci, Davut Izci, Mahmoud Abdel-Salam, Muhammed İkbal Şahin, Hakan Çoban, Selin Özteber, Ali Mardani*

![Methodology](docs/figures/methodology.png)

---

> [!IMPORTANT]
> **Citation is mandatory.** Any use of the dataset (`data/dataset.csv`) or the
> pre-trained models (`models/`), in whole or in part, including derived or
> modified versions, **must cite the main article** listed in [Citation](#citation).
> See [License](#license) for details.

---

## Repository structure

```
PCE/
├── app.py                 # Flask GUI + JSON API
├── templates/index.html   # GUI page
├── static/styles.css      # GUI styles
├── models/                # Pre-trained artifacts used by the GUI
│   ├── best_model_Y1.pkl          # CatBoost  – Marsh-funnel flow time
│   ├── best_model_Y2.pkl          # LightGBM  – mini-slump
│   ├── selected_features_Y1.pkl   # mRMR-selected input features for Y1
│   ├── selected_features_Y2.pkl   # mRMR-selected input features for Y2
│   └── scaler.pkl                 # StandardScaler fitted on X1–X22
├── data/
│   ├── dataset.csv        # Experimental dataset (616 mixtures)
│   └── README.md          # Variable descriptions
├── docs/
│   ├── figures/           # Methodology and SHAP figures
│   └── diagrams/          # Editable draw.io sources
├── main.py, src/          # Training pipeline used in the article (not needed for the GUI)
├── requirements.txt       # GUI dependencies (pip)
├── environment.yml        # GUI dependencies (conda)
├── CITATION.cff
└── LICENSE                # CC BY-NC 4.0 (non-commercial)
```

## Installation

Python **3.10** is recommended. The models were serialized with
**scikit-learn 1.3.2**; other versions may fail to load `scaler.pkl` or
produce warnings, so keep the pinned versions.

### Option A – conda (recommended)

```bash
conda env create -f environment.yml
conda activate pce
```

### Option B – pip

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Running the GUI

From the repository root:

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in a browser.

Optional arguments:

| Command | Description |
|---|---|
| `python app.py --port 8080` | Use a different port |
| `python app.py --host 0.0.0.0` | Make the GUI reachable from other devices on the local network |
| `python app.py --debug` | Flask debug mode (development only; never on a public host) |
| `flask --app app run` | Alternative: run with the Flask CLI |

To serve a different set of model artifacts, point `PCE_MODEL_DIR` at a folder
containing the same five `.pkl` files:

```bash
# Windows (PowerShell)
$env:PCE_MODEL_DIR="path\to\models"; python app.py
# Linux/macOS
PCE_MODEL_DIR=path/to/models python app.py
```

### Using the GUI

The form exposes the nine features selected by mRMR (PCE characteristics and
mixture proportions). Fill them in and press **Predict** to obtain Y1 and Y2.

| Input | Variable | Description |
|---|---|---|
| X13 | Density | Density of the water-reducing admixture |
| X14 | Solid Content | Solid content of the admixture |
| X15 | pH | pH of the admixture |
| X16 | M<sub>w</sub> | Weight-average molecular weight of the admixture |
| X17 | PDI | Polydispersity index of the admixture |
| X18 | Viscosity | Viscosity of the admixture |
| X19 | Cement | Cement amount (g) |
| X20 | Fly Ash | Fly ash amount (g) |
| X21 | Water | Water amount (g) |

Notes:
- `-` denotes a missing value.
- Decimal comma or point are both accepted (`3,5` = `3.5`).
- Features not shown in the form (X1–X12, X22) are fixed to the reference
  material values used in the experiments and are only needed by the scaler.

### JSON API

`POST /api/predict` accepts all 22 inputs (`X1`–`X22`) and returns both predictions:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"X1":47.6,"X2":20.22,"X3":3.6,"X4":16.04,"X5":3754,"X6":3.2,"X7":59.22,
       "X8":22.86,"X9":6.31,"X10":3.09,"X11":4300,"X12":2.31,"X13":1080,"X14":40.78,
       "X15":3.91,"X16":55000,"X17":2.1,"X18":154,"X19":700,"X20":"-","X21":3.5,"X22":245}'
```

```json
{"predictions": {"Y1": 77.556, "Y2": 17.475}}
```

## Dataset

`data/dataset.csv` contains 616 cementitious mixtures with 22 input variables
(cement clinker phases and physical properties, fly ash oxide composition and
physical properties, PCE admixture characteristics, and mixture proportions)
and two outputs (Y1 Marsh-funnel flow time, Y2 mini-slump). Full variable
descriptions are given in [`data/README.md`](data/README.md).

**Use of the dataset requires citing the main article** (see below).

## Citation

If you use the dataset, the pre-trained models, or this software, you **must** cite:

```bibtex
@article{ayaz_pce_eo_shap,
  title   = {Equilibrium Optimizer-Tuned Machine Learning Models for Predicting
             Fresh-State Properties of PCE-Modified Cement-Fly Ash Systems:
             A SHAP-Based Interpretation},
  author  = {Ayaz, {\.I}brahim and Ekinci, Serdar and Izci, Davut and
             Abdel-Salam, Mahmoud and {\c{S}}ahin, Muhammed {\.I}kbal and
             {\c{C}}oban, Hakan and {\"O}zteber, Selin and Mardani, Ali},
  journal = {(to be updated upon publication)},
  year    = {(to be updated)},
  doi     = {(to be updated)}
}
```

GitHub's **"Cite this repository"** button uses [`CITATION.cff`](CITATION.cff).

## License

The entire repository (source code, dataset, pre-trained models and figures)
is licensed under the
[Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](LICENSE).

- **Non-commercial use only.** Commercial use of any part of this repository,
  including derived or modified versions, is not permitted without prior
  written permission from the authors.
- **Mandatory citation.** Any use must cite the main article listed in
  [Citation](#citation), indicate if changes were made, and link to the license.

## Authors

İbrahim Ayaz¹, Serdar Ekinci², Davut Izci³, Mahmoud Abdel-Salam⁴, Muhammed İkbal Şahin³,
Hakan Çoban⁴, Selin Özteber⁵, Ali Mardani⁵\*

1. Department of Computer Technologies, Bitlis Eren University, Bitlis 13100, Türkiye
2. Department of Computer Engineering, Bitlis Eren University, Bitlis 13100, Türkiye
3. Department of Electrical and Electronic Engineering, Bursa Uludag University, Bursa 16059, Türkiye
4. Faculty of Computer and Information Science, Mansoura University, Mansoura 35516, Egypt
5. Department of Civil Engineering, Bursa Uludag University, Bursa 16059, Türkiye

\* Corresponding author
