from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer as SklearnIterativeImputer

class BaseImputer(ABC):
    @abstractmethod
    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputing missing values in the given dataframe."""
        pass

class IterativeImputerStrategy(BaseImputer):
    def __init__(self, random_state: int = 42, max_iter: int = 10):
        self.random_state = random_state
        self.max_iter = max_iter

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Imputing missing values with Iterative Imputer (MICE)...")
        # Replace dash string with NaN
        df = df.replace(r'^\s*-\s*$', np.nan, regex=True)
        df = df.apply(lambda x: pd.to_numeric(x, errors='ignore'))
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Keep index and columns
        imputer = SklearnIterativeImputer(random_state=self.random_state, max_iter=self.max_iter)
        df_imputed = df.copy()
        df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        return df_imputed
