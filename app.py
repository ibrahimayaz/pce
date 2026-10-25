from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Mapping, Tuple

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("PCE_MODEL_DIR", ROOT_DIR / "models"))
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
    "X13": "Density",
    "X14": "Solid Content",
    "X15": "pH",
    "X16": "Mw",
    "X17": "PDI",
    "X18": "Viscosity",
    "X19": "Cement",
    "X20": "Fly Ash",
    "X21": "Water",
    "X22": "Water-Reducing Admixture",
}
DEFAULT_FEATURE_VALUES = dict(
    zip(
        FEATURE_COLUMNS,
        [
            "47.6",
            "20.22",
            "3.6",
            "16.04",
            "3754",
            "3.2",
            "59.22",
            "22.86",
            "6.31",
            "3.09",
            "4300",
            "2.31",
            "1080",
            "40.78",
            "3.91",
            "55000",
            "2.1",
            "154",
            "700",
            "-",
            "3.5",
            "245",
        ],
    )
)


class ArtifactLoadError(RuntimeError):
    """Raised when one or more model artifacts cannot be loaded."""


def load_artifacts() -> Dict[str, object]:
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
    return artifacts


def parse_features(input_map: Mapping[str, object]) -> Tuple[Dict[str, float], Dict[str, str]]:
    values: Dict[str, float] = {}
    errors: Dict[str, str] = {}

    for column in FEATURE_COLUMNS:
        raw_value = input_map.get(column, "")
        if raw_value is None:
            raw_text = ""
        else:
            raw_text = str(raw_value).strip()

        if raw_text == "":
            errors[column] = "This field is required."
            continue

        if raw_text == "-":
            # Preserve the dataset convention where '-' means a missing numeric value.
            values[column] = float("nan")
            continue

        normalized = raw_text.replace(",", ".")
        try:
            values[column] = float(normalized)
        except ValueError:
            errors[column] = "Please enter a valid numeric value."

    return values, errors


def run_prediction(values: Dict[str, float], artifacts: Dict[str, object]) -> Dict[str, float]:
    input_df = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    scaled_array = artifacts["scaler"].transform(input_df)
    scaled_df = pd.DataFrame(scaled_array, columns=FEATURE_COLUMNS)

    x_y1 = scaled_df[artifacts["selected_features_y1"]]
    x_y2 = scaled_df[artifacts["selected_features_y2"]]

    prediction_y1 = float(artifacts["model_y1"].predict(x_y1)[0])
    prediction_y2 = float(artifacts["model_y2"].predict(x_y2)[0])

    return {"Y1": prediction_y1, "Y2": prediction_y2}


def create_app() -> Flask:
    app = Flask(__name__)

    try:
        artifacts = load_artifacts()
        artifact_error = ""
    except Exception as exc:  # pragma: no cover - surface startup issues in UI and API
        artifacts = {}
        artifact_error = str(exc)

    @app.route("/", methods=["GET", "POST"])
    def index():
        values = DEFAULT_FEATURE_VALUES.copy()
        field_errors: Dict[str, str] = {}
        result = None
        prediction_error = ""

        if request.method == "POST":
            for column in SELECTED_FEATURE_COLUMNS:
                values[column] = str(request.form.get(column, "")).strip()

            parsed_values, field_errors = parse_features(values)
            if not field_errors and not artifact_error:
                try:
                    result = run_prediction(parsed_values, artifacts)
                except Exception as exc:
                    prediction_error = f"Prediction failed: {exc}"

        return render_template(
            "index.html",
            feature_columns=SELECTED_FEATURE_COLUMNS,
            feature_labels=FEATURE_LABELS,
            input_feature_count=len(SELECTED_FEATURE_COLUMNS),
            values=values,
            field_errors=field_errors,
            result=result,
            artifact_error=artifact_error,
            prediction_error=prediction_error,
        )

    @app.route("/api/predict", methods=["POST"])
    def predict_api():
        if artifact_error:
            return jsonify({"error": artifact_error}), 500

        payload = request.get_json(silent=True) or {}
        parsed_values, field_errors = parse_features(payload)
        if field_errors:
            return jsonify({"error": "Validation failed", "field_errors": field_errors}), 400

        try:
            result = run_prediction(parsed_values, artifacts)
        except Exception as exc:
            return jsonify({"error": f"Prediction failed: {exc}"}), 500

        return jsonify({"predictions": result})

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
