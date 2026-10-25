import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from src.config import config
from src.data.data_loader import DataLoader
from src.data.imputer import IterativeImputerStrategy
from src.data.preprocessor import DataPreprocessor
from src.models.feature_selection import MRMRSelector
from src.models.optimizer import EquilibriumOptimizerStrategy
from src.models.model_factory import ModelFactory
from src.evaluation.evaluator import Evaluator
from src.visualization.visualizer import Visualizer
from src.visualization.shap_analyzer import ShapAnalyzer

class ExperimentPipeline:
    def __init__(self):
        self.data_loader = DataLoader(config.dataset_path)
        self.imputer = IterativeImputerStrategy(random_state=config.random_state)
        self.preprocessor = DataPreprocessor(random_state=config.random_state)
        self.feature_selector = MRMRSelector()
        self.optimizer = EquilibriumOptimizerStrategy(
            epoch=config.optimization.epoch, 
            pop_size=config.optimization.pop_size
        )

    def run(self):
        # 1. Load Data
        df = self.data_loader.load_data(apply_cleaning=config.apply_data_cleaning)

        # 2. Impute missing values
        if config.apply_imputation:
            df = self.imputer.impute(df)
        else:
            print("Imputation skipped.")

        # 3. Split features and targets
        X, y = self.preprocessor.split_features_targets(df)

        # 4. Standardize Full Data (Eğitim/Test ayrımı yok!)
        if config.apply_standardization:
            print("\nStandardizing full dataset variables (StandardScaler)...")
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
            X = X_scaled
            joblib.dump(scaler, config.model_path("scaler.pkl"))
            print(f"Scaler saved to '{config.model_path('scaler.pkl')}'.")

        # 5. Iterate through targets (Y1, Y2)
        for target_name in y.columns:
            print(f"\n{'='*50}")
            print(f"STARTING MODELING FOR TARGET: {target_name}")
            print(f"{'='*50}")

            y_target = y[target_name]
            X_target = X.copy()
            selected_features = []

            # 5.1 Feature Selection
            if config.apply_feature_selection:
                selected_features = self.feature_selector.select(
                    X_target, y_target, k=config.num_features_to_select
                )
                X_target = X_target[selected_features]
                Visualizer.plot_boxplot(X_target, target_name)

            # 5.2 Modeling & Optimization
            model_configs = ModelFactory.get_model_configs()
            models = {}
            best_hyperparameters = {}
            opt_times = {}

            if config.apply_hyperparam_opt:
                print(f"\n[{target_name}] Running Hyperparameter Optimization (EO)...")
                for mod_name, mod_config in model_configs.items():
                    best_params, elapsed_time = self.optimizer.optimize(mod_name, mod_config, X_target, y_target)
                    best_hyperparameters[mod_name] = best_params
                    opt_times[mod_name] = elapsed_time
                    models[mod_name] = ModelFactory.create_model(mod_name, best_params)
            else:
                print(f"\n[{target_name}] Hyperparameter Optimization IS OFF. Using default params.")
                for mod_name, mod_config in model_configs.items():
                    best_params = mod_config["fixed_params"].copy()
                    best_hyperparameters[mod_name] = best_params
                    opt_times[mod_name] = 0.0
                    models[mod_name] = ModelFactory.create_model(mod_name, best_params)

            # Save Hyperparameters
            hyperparams_rows = []
            for mod_name, mod_config in model_configs.items():
                param_names = mod_config["param_names"]
                bounds = mod_config.get("bounds", [])
                best_params = best_hyperparameters[mod_name]
                opt_time = opt_times.get(mod_name, 0.0)
                
                # Fetch bounds
                bounds_dict = {}
                for b in bounds:
                    bounds_dict[b.name] = f"[{b.lb}, {b.ub}]"
                
                is_first_row = True
                
                # Handle MLP specific case first
                if mod_name == "MLP":
                    val = best_params.get("hidden_layer_sizes", "N/A")
                    b1 = bounds_dict.get("hidden_layer_size_1", "N/A")
                    b2 = bounds_dict.get("hidden_layer_size_2", "N/A")
                    rng = f"[{b1}, {b2}]"
                    hyperparams_rows.append({
                        "Models": mod_name,
                        "Hyper parameters": "hidden_layer_sizes",
                        "Range of values": rng,
                        target_name: val,
                        "Optimization Time": f"{opt_time:.2f}s"
                    })
                    is_first_row = False
                
                for p_name in param_names:
                    if mod_name == "MLP" and p_name in ["hidden_layer_size_1", "hidden_layer_size_2"]:
                        continue
                        
                    val = best_params.get(p_name, "N/A")
                    rng = bounds_dict.get(p_name, "N/A")
                    
                    hyperparams_rows.append({
                        "Models": mod_name if is_first_row else "",
                        "Hyper parameters": p_name,
                        "Range of values": rng,
                        target_name: val,
                        "Optimization Time": f"{opt_time:.2f}s" if is_first_row else ""
                    })
                    is_first_row = False

            df_hyperparams = pd.DataFrame(hyperparams_rows)
            df_hyperparams.to_csv(config.output_path(f"best_hyperparameters_{target_name}.csv"), index=False)

            # 5.3 Model Evaluation
            for name, model in models.items():
                model.fit(X_target, y_target)
                
            results, oof_preds = Evaluator.evaluate_models(models, X_target, y_target, target_name)
            Evaluator.print_and_save_metrics(results, target_name)

            # 5.4 Detailed Analysis for Best Model
            best_model_name = Evaluator.get_best_model_name(results)
            print(f"\n[{target_name}] Generating detailed analyses for BEST MODEL: {best_model_name}...")

            best_model = clone(models[best_model_name])
            if "XGB" in best_model_name:
                best_model.set_params(base_score=0.5)

            # Retrain best_model on all available data for final evaluation inferences
            best_model.fit(X_target, y_target)
            
            # Artık Out-of-Fold tahminlerini kullanıyoruz. Test seti yok.
            Evaluator.save_predictions(y_target, oof_preds[best_model_name], target_name)
            Evaluator.save_model(best_model, selected_features, target_name)

            # Deep plotting
            Visualizer.plot_scatter_all(results, oof_preds, y_target, target_name)
            
            y_pred_best = oof_preds[best_model_name]
            Visualizer.plot_prediction_curve(y_target, y_pred_best, target_name, best_model_name)
            
            Visualizer.plot_learning_curve(best_model, X_target, y_target, target_name, best_model_name)
            
            ShapAnalyzer.analyze_and_plot(best_model, X_target, target_name, best_model_name)

            print(f"[{target_name}] All analyses, plots, and CSV files successfully generated and saved!")

if __name__ == "__main__":
    pipeline = ExperimentPipeline()
    pipeline.run()
