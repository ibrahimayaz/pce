---
title: PCE Fresh-State Predictor
emoji: 🧪
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: Marsh funnel flow time and mini-slump of PCE fly ash pastes
---

# PCE fresh-state predictor

Online interface for the final EO-tuned CatBoost models of the article *Predicting Marsh Funnel Flow Time and
Mini-Slump of Fly Ash–Cement Pastes from PCE Superplasticizer Descriptors* (IEEE Access, Access-2026-47475).

- Code, data and the Flask GUI: https://www.github.com/ibrahimayaz/pce
- Dataset: IEEE DataPort, doi: [10.21227/fjdp-ss56](https://dx.doi.org/10.21227/fjdp-ss56)

The models apply to the cement, fly ash and water content of the experimental programme. Accuracy for
admixtures that were not part of the training data is lower (leave-one-admixture-out R2 = 0.87 for flow time,
about 0.20 for mini-slump).
