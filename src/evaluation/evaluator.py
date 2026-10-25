from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold
from sklearn.base import clone
import joblib

from src.config import config

class Evaluator:
    @staticmethod
    def evaluate_models(models: Dict[str, Any], X: pd.DataFrame, y: pd.Series, target_name: str, cv: int = 5) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
        results = {}
        oof_preds = {}

        print(f"\n[{target_name}] Evaluating model performances over {cv}-Fold CV...")
        
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        for name, model in models.items():
            cv_r2_list, cv_rmse_list, cv_mae_list, cv_mse_list, cv_mape_list = [], [], [], [], []
            
            if isinstance(X, np.ndarray):
                X_arr, y_arr = X, y
            else:
                X_arr, y_arr = X.values, y.values

            # Out-of-fold tahminlerini toplamak için boş array
            oof_pred_array = np.zeros(len(y_arr))

            for train_idx, val_idx in kf.split(X_arr):
                X_fold_train, y_fold_train = X_arr[train_idx], y_arr[train_idx]
                X_fold_val, y_fold_val = X_arr[val_idx], y_arr[val_idx]
                
                fold_model = clone(model)
                fold_model.fit(X_fold_train, y_fold_train)
                
                y_pred_fold = fold_model.predict(X_fold_val)
                oof_pred_array[val_idx] = y_pred_fold
                
                cv_r2_list.append(r2_score(y_fold_val, y_pred_fold))
                cv_mse_list.append(mean_squared_error(y_fold_val, y_pred_fold))
                cv_rmse_list.append(np.sqrt(mean_squared_error(y_fold_val, y_pred_fold)))
                cv_mae_list.append(mean_absolute_error(y_fold_val, y_pred_fold))
                cv_mape_list.append(mean_absolute_percentage_error(y_fold_val, y_pred_fold))
            
            oof_preds[name] = oof_pred_array

            results[name] = {
                "CV_R2_Mean": np.mean(cv_r2_list),
                "CV_R2_Std": np.std(cv_r2_list),
                "CV_RMSE_Mean": np.mean(cv_rmse_list),
                "CV_RMSE_Std": np.std(cv_rmse_list),
                "CV_MAE_Mean": np.mean(cv_mae_list),
                "CV_MAE_Std": np.std(cv_mae_list),
                "CV_MSE_Mean": np.mean(cv_mse_list),
                "CV_MSE_Std": np.std(cv_mse_list),
                "CV_MAPE_Mean": np.mean(cv_mape_list),
                "CV_MAPE_Std": np.std(cv_mape_list)
            }

        return results, oof_preds

    @staticmethod
    def print_and_save_metrics(results: Dict[str, Dict[str, float]], target_name: str):
        formatted_results = {}
        for k, v in results.items():
            r2_str = f"{v['CV_R2_Mean']:.4f} ± {v['CV_R2_Std']:.4f}"
            rmse_str = f"{v['CV_RMSE_Mean']:.4f} ± {v['CV_RMSE_Std']:.4f}"
            mae_str = f"{v['CV_MAE_Mean']:.4f} ± {v['CV_MAE_Std']:.4f}"
            mse_str = f"{v['CV_MSE_Mean']:.4f} ± {v['CV_MSE_Std']:.4f}"
            mape_str = f"{v['CV_MAPE_Mean']:.4f} ± {v['CV_MAPE_Std']:.4f}"
            
            print(f"[{k}] Performance Metrics:")
            print(f"   CV R2    : {r2_str}")
            print(f"   CV RMSE  : {rmse_str}")
            print(f"   CV MAE   : {mae_str}")
            print(f"   CV MSE   : {mse_str}")
            print(f"   CV MAPE  : {mape_str}")
            
            formatted_results[k] = {
                "CV_R2": r2_str,
                "CV_RMSE": rmse_str,
                "CV_MAE": mae_str,
                "CV_MSE": mse_str,
                "CV_MAPE": mape_str
            }

        df_metrics = pd.DataFrame(formatted_results).T
        df_metrics.index.name = "Model"
        metrics_file = config.output_path(f"performance_metrics_{target_name}.csv")
        df_metrics.to_csv(metrics_file)
        print(f"[{target_name}] Performance metrics saved to '{metrics_file}'.")

    @staticmethod
    def get_best_model_name(results: Dict[str, Dict[str, float]]) -> str:
        return min(results, key=lambda k: results[k]['CV_RMSE_Mean'])


    @staticmethod
    def save_predictions(y: pd.Series, y_pred: np.ndarray, target_name: str):
        df_preds = pd.DataFrame({'Actual_Value': y, 'Predicted_Value': y_pred}, index=y.index)
        preds_file = config.output_path(f'prediction_results_{target_name}.csv')
        df_preds.to_csv(preds_file)
        print(f"[{target_name}] Predictions saved to '{preds_file}'.")

    @staticmethod
    def save_model(model: Any, selected_features: list, target_name: str):
        joblib.dump(model, config.model_path(f"best_model_{target_name}.pkl"))
        if selected_features:
            joblib.dump(selected_features, config.model_path(f"selected_features_{target_name}.pkl"))
        print(f"[{target_name}] Best model saved to '{config.models_dir}'.")
