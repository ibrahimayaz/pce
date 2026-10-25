from abc import ABC, abstractmethod
import pandas as pd
from typing import List

class BaseFeatureSelector(ABC):
    @abstractmethod
    def select(self, X_train: pd.DataFrame, y_train: pd.Series, k: int) -> List[str]:
        pass

class MRMRSelector(BaseFeatureSelector):
    def select(self, X_train: pd.DataFrame, y_train: pd.Series, k: int) -> List[str]:
        try:
            from mrmr import mrmr_regression
            print(f"Applying mRMR algorithm to select top {k} features...")
            selected_features = mrmr_regression(X=X_train, y=y_train, K=k)
            
            if len(selected_features) == 0:
                print("Warning: mRMR selected no features! Falling back to all features.")
                selected_features = X_train.columns.tolist()
                
            return selected_features
        except ImportError:
            print("mrmr library not installed. Falling back to all features.")
            return X_train.columns.tolist()
