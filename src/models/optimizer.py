from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

class BaseOptimizer(ABC):
    @abstractmethod
    def optimize(self, model_name: str, config: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Dict[str, Any], float]:
        """Returns best params and elapsed time."""
        pass

class EquilibriumOptimizerStrategy(BaseOptimizer):
    def __init__(self, epoch: int = 100, pop_size: int = 20):
        self.epoch = epoch
        self.pop_size = pop_size

    def _convert_params(self, solution: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        params = config["fixed_params"].copy()
        
        # Mapping values to params
        for idx, p_name in enumerate(config["param_names"]):
            if p_name in ["n_estimators", "max_depth", "iterations", "depth", "hidden_layer_size_1", "hidden_layer_size_2"]:
                params[p_name] = int(solution[idx])
            else:
                params[p_name] = float(solution[idx])
                
        # Handle specific MLP parameter remapping
        if "hidden_layer_size_1" in params:
            s1 = params.pop("hidden_layer_size_1")
            s2 = params.pop("hidden_layer_size_2")
            params["hidden_layer_sizes"] = (s1, s2)
            
        return params

    def optimize(self, model_name: str, config: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Dict[str, Any], float]:
        from mealpy.physics_based import EO
        import time
        
        def fitness_function(solution):
            params = self._convert_params(solution, config)
            model = config["model_class"](**params)
            
            # Negative Mean Squared Error is maximized by Mealpy natively (needs to be minimized by taking negative of it?)
            # Wait, mealpy solves for minmax. If we set "min", it minimizes the return of obj_func.
            # cross_val returns negative MSE mapping (higher is better, i.e., closer to 0).
            # So mean(neg_mse) is negative. To minimize error, we return negative of that (which becomes positive MSE).
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
            return -np.mean(scores)
            
        problem_dict = {
            "obj_func": fitness_function,
            "bounds": config["bounds"],
            "minmax": "min",
        }
        
        print(f"Starting EO Optimization for {model_name}...")
        start_time = time.time()
        opt = EO.OriginalEO(epoch=self.epoch, pop_size=self.pop_size)
        g_best = opt.solve(problem_dict)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        
        # Convert best solution to actual param dictionary
        best_params = self._convert_params(g_best.solution, config)
        return best_params, elapsed_time
