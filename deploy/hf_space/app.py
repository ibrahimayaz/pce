"""Online GUI (Hugging Face Space, Gradio SDK) for the revised PCE fresh-state models.

Uses the same final EO-tuned CatBoost pipelines (models_v2/) as the Flask GUI in the GitHub repository
(https://www.github.com/ibrahimayaz/pce).  The constant inputs X1-X12 and X22 take the values of the
experimental programme; the nine non-constant inputs are entered by the user.
"""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

try:  # available on Hugging Face ZeroGPU Spaces only
    import spaces

    @spaces.GPU
    def _zerogpu_placeholder():
        """ZeroGPU requires one @spaces.GPU function at startup; it is never called (predictions run on CPU)."""
except ImportError:
    pass

HERE = Path(__file__).resolve().parent
META = json.loads((HERE / "models_v2" / "metadata.json").read_text(encoding="utf-8"))
PIPES = {t: joblib.load(HERE / "models_v2" / f"pipeline_{t}.joblib") for t in ("Y1", "Y2")}
ADMIXTURES = json.loads((HERE / "admixtures.json").read_text(encoding="utf-8"))
FEATURES = [f"X{i}" for i in range(1, 23)]
INPUTS = ["X13", "X14", "X15", "X16", "X17", "X18", "X19", "X20", "X21"]
LABELS = {
    "X13": "X13 PCE density (dataset scale)",
    "X14": "X14 PCE solid content (%)",
    "X15": "X15 PCE pH",
    "X16": "X16 PCE Mw (g/mol)",
    "X17": "X17 PCE PDI",
    "X18": "X18 PCE viscosity (dataset scale)",
    "X19": "X19 Cement (g)",
    "X20": "X20 Fly ash (g; 0 = no fly ash)",
    "X21": "X21 PCE dosage (g per 700 g binder)",
}
CUSTOM = "Custom descriptors"
# example of the paper (Fig. 13): admixture K4:1, 15% fly ash, 1.0% PCE
EXAMPLE = {"admixture": "K4:1", "X19": 595.0, "X20": 105.0, "X21": 7.0}


def fill_descriptors(name):
    if name not in ADMIXTURES:
        return [gr.update() for _ in INPUTS[:6]]
    return [ADMIXTURES[name][c] for c in INPUTS[:6]]


def predict(name, *vals):
    values = {c: META["inputs"][c]["default"] for c in FEATURES if META["inputs"][c]["constant"]}
    for c, v in zip(INPUTS, vals):
        if v is None:
            raise gr.Error(f"{LABELS[c]} is required.")
        values[c] = float(v)
    x = pd.DataFrame([values], columns=FEATURES)
    y1 = float(PIPES["Y1"].predict(x)[0])
    y2 = float(PIPES["Y2"].predict(x)[0])

    notes = []
    for c in INPUTS:
        info = META["inputs"][c]
        if not info["min"] <= values[c] <= info["max"]:
            notes.append(f"- {c} = {values[c]:g} is outside the training range "
                         f"[{info['min']:g}, {info['max']:g}]; the prediction is an extrapolation.")
    if abs(values["X19"] + values["X20"] - 700) > 1e-6:
        notes.append("- X19 + X20 should equal 700 g (binder mass of the experimental programme).")
    if name == CUSTOM or name not in ADMIXTURES:
        notes.append("- For an admixture that was not part of the training data the accuracy is lower "
                     "(leave-one-admixture-out R² = 0.87 for Y1 and about 0.20 for Y2).")
    elif not ADMIXTURES[name]["y2_measured"]:
        notes.append("- Mini-slump was not measured for this admixture; the Y2 prediction is an extrapolation "
                     "to an admixture without Y2 data.")
    return f"{y1:.1f} s", f"{y2:.1f} cm", "\n".join(notes) or "All inputs are within the training range."


with gr.Blocks(title="PCE fresh-state predictor") as demo:
    gr.Markdown(
        "## Marsh funnel flow time and mini-slump of fly ash–cement pastes with PCE superplasticizers\n"
        "Final EO-tuned CatBoost models (nested CV: Y1 R² = 0.954, RMSE = 7.68 s; Y2 R² = 0.784, "
        "RMSE = 0.46 cm). One cement, one fly ash, 700 g binder and 245 g water (w/b = 0.35) were used; "
        "these constant inputs are fixed. The predictions support preliminary screening and do not replace "
        "laboratory tests.")
    with gr.Row():
        with gr.Column():
            adm = gr.Dropdown([CUSTOM] + sorted(ADMIXTURES), value=EXAMPLE["admixture"],
                              label="Admixture (fills X13–X18)")
            boxes = {c: gr.Number(label=LABELS[c],
                                  value=ADMIXTURES[EXAMPLE["admixture"]][c] if c in ADMIXTURES[EXAMPLE["admixture"]]
                                  else EXAMPLE[c]) for c in INPUTS}
            btn = gr.Button("Predict", variant="primary")
        with gr.Column():
            out1 = gr.Textbox(label="Y1 Marsh funnel flow time")
            out2 = gr.Textbox(label="Y2 Mini-slump spread")
            warn = gr.Markdown()
            gr.Markdown(
                "Mixture inputs: 0/15/30/45% fly ash correspond to X20 = 0/105/210/315 g and "
                "X19 = 700/595/490/385 g; PCE dosage 0.5–2.0% of binder corresponds to X21 = 3.5–14 g.\n\n"
                "Data: IEEE DataPort, doi: [10.21227/fjdp-ss56](https://dx.doi.org/10.21227/fjdp-ss56) · Code: "
                "[github.com/ibrahimayaz/pce](https://www.github.com/ibrahimayaz/pce)")
    adm.change(fill_descriptors, adm, [boxes[c] for c in INPUTS[:6]])
    btn.click(predict, [adm] + [boxes[c] for c in INPUTS], [out1, out2, warn])

if __name__ == "__main__":
    # SSR (Node.js server-side rendering) is disabled: it is not needed and stops on Spaces
    demo.launch(ssr_mode=False)
