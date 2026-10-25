import shap
import matplotlib.pyplot as plt
import pandas as pd

from src.config import config

class ShapAnalyzer:
    @staticmethod
    def analyze_and_plot(model, X_train: pd.DataFrame, target_name: str, model_name: str):
        print(f"[{target_name}] Generating SHAP plots...")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)
        except Exception as e:
            print(f"[{target_name}] TreeExplainer failed: {e}. Trying KernelExplainer...")
            X_sample = shap.sample(X_train, 100) if len(X_train) > 100 else X_train
            predict_fn = lambda x: model.predict(x)
            explainer = shap.KernelExplainer(predict_fn, X_sample)
            shap_values = explainer.shap_values(X_train)

        # SHAP Summary Plot
        plt.figure()
        shap.summary_plot(shap_values, X_train, show=False)
        plt.title(f"SHAP Summary Plot - {target_name} ({model_name})")
        plt.tight_layout()
        plt.savefig(config.output_path(f"shap_summary_{target_name}.png"))
        plt.close()

        # SHAP Bar Plot
        plt.figure()
        shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance - {target_name} ({model_name})")
        plt.tight_layout()
        plt.savefig(config.output_path(f"shap_bar_{target_name}.png"))
        plt.close()
