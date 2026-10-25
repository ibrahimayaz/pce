from typing import Dict, Any, Type
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from mealpy import FloatVar, IntegerVar

class ModelFactory:
    @staticmethod
    def get_model_configs() -> Dict[str, Dict[str, Any]]:
        return {
            "LightGBM": {
                "model_class": LGBMRegressor,
                "bounds": [
                    IntegerVar(lb=20, ub=150, name="n_estimators"),
                    IntegerVar(lb=2, ub=5, name="max_depth"),
                    FloatVar(lb=0.001, ub=0.1, name="learning_rate"),
                    FloatVar(lb=0.1, ub=10.0, name="reg_alpha"),
                    FloatVar(lb=0.1, ub=10.0, name="reg_lambda")
                ],
                "fixed_params": {"verbose": -1, "random_state": 42},
                "param_names": ["n_estimators", "max_depth", "learning_rate", "reg_alpha", "reg_lambda"]
            },
            "XGBoost": {
                "model_class": XGBRegressor,
                "bounds": [
                    IntegerVar(lb=20, ub=150, name="n_estimators"),
                    IntegerVar(lb=2, ub=5, name="max_depth"),
                    FloatVar(lb=0.001, ub=0.1, name="learning_rate"),
                    FloatVar(lb=0.1, ub=10.0, name="alpha"),
                    FloatVar(lb=0.1, ub=10.0, name="lambda")
                ],
                "fixed_params": {"objective": "reg:squarederror", "random_state": 42, "n_jobs": -1},
                "param_names": ["n_estimators", "max_depth", "learning_rate", "alpha", "lambda"]
            },
            "AdaBoost": {
                "model_class": AdaBoostRegressor,
                "bounds": [
                    IntegerVar(lb=20, ub=100, name="n_estimators"),
                    FloatVar(lb=0.01, ub=0.5, name="learning_rate")
                ],
                "fixed_params": {"random_state": 42},
                "param_names": ["n_estimators", "learning_rate"]
            },
            "CatBoost": {
                "model_class": CatBoostRegressor,
                "bounds": [
                    IntegerVar(lb=50, ub=150, name="iterations"),
                    IntegerVar(lb=2, ub=5, name="depth"),
                    FloatVar(lb=0.001, ub=0.1, name="learning_rate"),
                    FloatVar(lb=1.0, ub=10.0, name="l2_leaf_reg")
                ],
                "fixed_params": {"verbose": 0, "random_state": 42},
                "param_names": ["iterations", "depth", "learning_rate", "l2_leaf_reg"]
            },
            "MLP": {
                "model_class": MLPRegressor,
                "bounds": [
                    IntegerVar(lb=16, ub=128, name="hidden_layer_size_1"),
                    IntegerVar(lb=8, ub=64, name="hidden_layer_size_2"),
                    FloatVar(lb=0.01, ub=1.0, name="alpha"),
                    FloatVar(lb=0.0001, ub=0.05, name="learning_rate_init")
                ],
                "fixed_params": {"hidden_layer_sizes": (32, 16), "max_iter": 2000, "early_stopping": True, "random_state": 42},
                "param_names": ["hidden_layer_size_1", "hidden_layer_size_2", "alpha", "learning_rate_init"]
            }
        }

    @staticmethod
    def create_model(model_name: str, params: Dict[str, Any]) -> Any:
        configs = ModelFactory.get_model_configs()
        if model_name not in configs:
            raise ValueError(f"Unknown model {model_name}")
        
        return configs[model_name]["model_class"](**params)
