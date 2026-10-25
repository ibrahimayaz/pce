import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

class DataPreprocessor:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def split_features_targets(self, df: pd.DataFrame, target_start_idx: int = 22):
        X = df.iloc[:, :target_start_idx]
        y = df.iloc[:, target_start_idx:]  # e.g., Y1, Y2
        return X, y

    def split_train_test(self, X: pd.DataFrame, y: pd.DataFrame, test_size: float = 0.20):
        print(f"\nSplitting data into train ({(1-test_size)*100:.0f}%) and test ({test_size*100:.0f}%)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        return X_train, X_test, y_train, y_test

    def standardize(self, X_train: pd.DataFrame, X_test: pd.DataFrame, save_path: str = "scaler.pkl"):
        print("\nStandardizing variables (StandardScaler)...")
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )
        
        joblib.dump(scaler, save_path)
        print(f"Scaler saved as '{save_path}'.")
        return X_train_scaled, X_test_scaled
