from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT_DIR = Path(__file__).resolve().parent
# The revised pipelines contain the leakage-free preprocessor from revision/rev
sys.path.insert(0, str(ROOT_DIR / "revision"))

MODEL_DIR = Path(os.environ.get("PCE_MODEL_DIR", ROOT_DIR / "models"))  # original (v1) artifacts
MODEL_DIR_V2 = Path(os.environ.get("PCE_MODEL_DIR_V2", ROOT_DIR / "models_v2"))  # revised artifacts
FEATURE_COLUMNS = [f"X{i}" for i in range(1, 23)]
SELECTED_FEATURE_COLUMNS = ["X13", "X14", "X15", "X16", "X17", "X18", "X19", "X20", "X21"]
FEATURE_LABELS = {
    "X1": "C3S",
    "X2": "C2S",
    "X3": "C3A",
    "X4": "C4AF",
    "X5": "SSAC",
    "X6": "SGC",
    "X7": "SiO2",
    "X8": "Al2O3",
    "X9": "Fe2O3",
    "X10": "CaO",
    "X11": "SSAF",
    "X12": "SGF",
    "X13": "PCE density",
    "X14": "PCE solid content, %",
    "X15": "PCE pH",
    "X16": "PCE Mw, g/mol",
    "X17": "PCE PDI",
    "X18": "PCE viscosity",
    "X19": "Cement, g",
    "X20": "Fly ash, g (0 = no fly ash)",
    "X21": "PCE dosage, g per 700 g binder",
    "X22": "Water, g",
}
TARGET_LABELS = {"Y1": ("Marsh Funnel Flow Time", "s"), "Y2": ("Mini-Slump Spread", "cm")}
DEFAULT_FEATURE_VALUES = dict(
    zip(
        FEATURE_COLUMNS,
        ["47.6", "20.22", "3.6", "16.04", "3754", "3.2", "59.22", "22.86", "6.31", "3.09", "4300", "2.31",
         "1080", "40.78", "3.91", "55000", "2.1", "154", "700", "0", "3.5", "245"],
    )
)


class ArtifactLoadError(RuntimeError):
    """Raised when one or more model artifacts cannot be loaded."""


def load_artifacts() -> Dict[str, object]:
    """Load the revised (v2) pipelines if present, otherwise the original artifacts."""
    meta_path = MODEL_DIR_V2 / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        arts: Dict[str, object] = {"version": 2, "meta": meta}
        for t in ("Y1", "Y2"):
            p = MODEL_DIR_V2 / f"pipeline_{t}.joblib"
            if not p.exists():
                raise ArtifactLoadError(f"Missing required artifact file: {p.name}")
            arts[t] = joblib.load(p)
        return arts

    paths = {
        "model_y1": MODEL_DIR / "best_model_Y1.pkl",
        "model_y2": MODEL_DIR / "best_model_Y2.pkl",
        "selected_features_y1": MODEL_DIR / "selected_features_Y1.pkl",
        "selected_features_y2": MODEL_DIR / "selected_features_Y2.pkl",
        "scaler": MODEL_DIR / "scaler.pkl",
    }
    missing = [str(path.name) for path in paths.values() if not path.exists()]
    if missing:
        raise ArtifactLoadError(f"Missing required artifact files: {', '.join(missing)}")
    artifacts = {key: joblib.load(path) for key, path in paths.items()}
    artifacts["version"] = 1
    return artifacts


def form_columns(artifacts: Mapping[str, object]) -> List[str]:
    if artifacts.get("version") == 2:
        meta = artifacts["meta"]
        used = set()
        for t in meta["targets"].values():
            used.update(t["selected_features"])
        return [c for c in FEATURE_COLUMNS if c in used]
    return SELECTED_FEATURE_COLUMNS


def parse_features(input_map: Mapping[str, object]) -> Tuple[Dict[str, float], Dict[str, str]]:
    values: Dict[str, float] = {}
    errors: Dict[str, str] = {}

    for column in FEATURE_COLUMNS:
        raw_value = input_map.get(column, "")
        raw_text = "" if raw_value is None else str(raw_value).strip()

        if raw_text == "":
            errors[column] = "This field is required."
            continue

        if raw_text == "-":
            # In the dataset '-' in X20 means "no fly ash"
            values[column] = 0.0 if column == "X20" else float("nan")
            continue

        try:
            values[column] = float(raw_text.replace(",", "."))
        except ValueError:
            errors[column] = "Please enter a valid numeric value."

    return values, errors


def range_warnings(values: Dict[str, float], artifacts: Mapping[str, object]) -> List[str]:
    if artifacts.get("version") != 2:
        return []
    warns = []
    meta = artifacts["meta"]["inputs"]
    for c in form_columns(artifacts):
        v, info = values.get(c), meta[c]
        if v is None:
            continue
        if v < info["min"] or v > info["max"]:
            warns.append(f"{c} ({info['name']}) = {v:g} is outside the training range "
                         f"[{info['min']:g}, {info['max']:g}]; the prediction is an extrapolation.")
    return warns


def run_prediction(values: Dict[str, float], artifacts: Dict[str, object]) -> Dict[str, float]:
    input_df = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    if artifacts.get("version") == 2:
        return {t: float(artifacts[t].predict(input_df)[0]) for t in ("Y1", "Y2")}

    scaled_array = artifacts["scaler"].transform(input_df)
    scaled_df = pd.DataFrame(scaled_array, columns=FEATURE_COLUMNS)
    x_y1 = scaled_df[artifacts["selected_features_y1"]]
    x_y2 = scaled_df[artifacts["selected_features_y2"]]
    return {"Y1": float(artifacts["model_y1"].predict(x_y1)[0]),
            "Y2": float(artifacts["model_y2"].predict(x_y2)[0])}


def create_app() -> Flask:
    app = Flask(__name__)

    try:
        artifacts = load_artifacts()
        artifact_error = ""
    except Exception as exc:  # pragma: no cover - surface startup issues in UI and API
        artifacts = {}
        artifact_error = str(exc)

    columns = form_columns(artifacts) if artifacts else SELECTED_FEATURE_COLUMNS
    defaults = DEFAULT_FEATURE_VALUES.copy()
    if artifacts.get("version") == 2:
        # fixed (constant) inputs take the value used in the experimental programme
        for c, info in artifacts["meta"]["inputs"].items():
            if info["constant"]:
                defaults[c] = f"{info['default']:g}"

    @app.route("/", methods=["GET", "POST"])
    def index():
        values = defaults.copy()
        field_errors: Dict[str, str] = {}
        result = None
        warnings: List[str] = []
        prediction_error = ""

        if request.method == "POST":
            for column in columns:
                values[column] = str(request.form.get(column, "")).strip()

            parsed_values, field_errors = parse_features(values)
            if not field_errors and not artifact_error:
                try:
                    result = run_prediction(parsed_values, artifacts)
                    warnings = range_warnings(parsed_values, artifacts)
                except Exception as exc:
                    prediction_error = f"Prediction failed: {exc}"

        return render_template(
            "index.html",
            feature_columns=columns,
            feature_labels=FEATURE_LABELS,
            target_labels=TARGET_LABELS,
            input_feature_count=len(columns),
            values=values,
            field_errors=field_errors,
            result=result,
            warnings=warnings,
            artifact_error=artifact_error,
            prediction_error=prediction_error,
            model_version=artifacts.get("version", 0),
        )

    @app.route("/api/predict", methods=["POST"])
    def predict_api():
        if artifact_error:
            return jsonify({"error": artifact_error}), 500

        payload = dict(defaults)
        payload.update(request.get_json(silent=True) or {})
        parsed_values, field_errors = parse_features(payload)
        if field_errors:
            return jsonify({"error": "Validation failed", "field_errors": field_errors}), 400

        try:
            result = run_prediction(parsed_values, artifacts)
        except Exception as exc:
            return jsonify({"error": f"Prediction failed: {exc}"}), 500

        return jsonify({"predictions": result, "units": {k: v[1] for k, v in TARGET_LABELS.items()},
                        "warnings": range_warnings(parsed_values, artifacts)})

    return app


app = create_app()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PCE fresh-state property prediction GUI")
    parser.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 to expose the GUI on the local network.")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode (never on a public host).")
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=args.debug)
