import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self, apply_cleaning: bool = False) -> pd.DataFrame:
        print(f"\nLoading data from {self.file_path}...")
        df = pd.read_csv(self.file_path)

        if apply_cleaning:
            print("Applying data cleaning (Removing dash '-' rows)...")
            df.columns = [str(col).strip().replace("\n", " ") for col in df.columns]
            mask_dash_rows = df.astype(str).apply(lambda col: col.str.strip()).eq("-").any(axis=1)
            print("Rows to be removed:", mask_dash_rows.sum())
            df = df.loc[~mask_dash_rows].copy()
            print(f"New data size: {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            print("Data cleaning skipped.")

        if 'index' in df.columns:
            df = df.set_index('index')

        # Correcting column data types
        if "X20" in df.columns:
            df["X20"] = pd.to_numeric(df["X20"], errors="coerce").astype("Float64")
        if "Y2" in df.columns:
            df["Y2"] = pd.to_numeric(df["Y2"], errors="coerce").astype(float)
            
        return df
