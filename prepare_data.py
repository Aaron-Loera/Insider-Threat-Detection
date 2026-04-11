# Imports 
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from scripts.UEBAIsolationForest import UEBAIsolationForest  

# CONSTANTS 
SCALER_PATH = os.path.join(r"encoders\encoder_model_1\feature_scaler.pkl")
ENCODER_PATH = os.path.join(r"encoders\encoder_model_1\encoder_model.keras")
IF_PATH  = os.path.join(r"isolation_forests\iforest_model_1\iforest_model.pkl")


def chronological_split(csv_path: str | None=None, df: pd.DataFrame | None=None, split_ratio: float=0.9) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and creates a chronological split for a UEBA-enhanced dataset.
    
    Either a path to a CSV file or a DataFrame can be provided. The split ratio determines the percentage that
    will be used for model training. The remaining percentage will be used for model validation.
    
    Args:
        csv_path: Path where the processed UEBA dataset is stored
        df: UEBA-enhanced dataset
        split_ratio: The ratio to dedicate to model training
        
    Returns:
        tuple: A training and testing DataFrame 
    """
    if csv_path is None and df is None:
        raise ValueError("Please provide either a CSV path or a DataFrame to create a split.")
    
    if df is None:
        df = pd.read_csv(csv_path, index_col=0)
    
    # Normalize "user" and "day" columns
    df["user"] = df["user"].str.strip().str.lower()
    df["day"] = pd.to_datetime(df["day"]).dt.normalize()

    # Ensure sorted globally by time
    df = df.sort_values("day").reset_index(drop=True)

    unique_days = np.sort(df["day"].unique())
    cutoff_index = int(len(unique_days) * split_ratio)
    cutoff_day = unique_days[cutoff_index]

    train_df = df[df["day"] <= cutoff_day]
    test_df  = df[df["day"] > cutoff_day]

    return train_df, test_df


def get_insiders(path: str, version: str | float, return_all: bool=False) -> pd.DataFrame:
    """
    Loads the raw insiders table and returns the scenarios for a specific version. If specified, the
    full raw table across all CERT versions can be returned.
    
    Args:
        path: The location of the insiders CSV file
        version: CERT version to extract
        return_all: If true, returns the full insider table along with the version-specific table
        
    Returns:
        pd.DataFrame: A table describing the window of abnormal activity for each insider
    """
    df = pd.read_csv(path)
    
    # Normalizing table
    df.columns = df.columns.str.strip()
    df["user"] = df["user"].str.strip().str.lower()
    df["start"] = pd.to_datetime(df["start"].str.strip())
    df["end"] = pd.to_datetime(df["end"].str.strip())
    df["start_day"] = df["start"].dt.normalize()
    df["end_day"] = df["end"].dt.normalize()
    
    if isinstance(version, float):
        version = str(version)
    
    # Extracting only version-specific insiders
    insiders_df = df[df["dataset"].astype(str).str.strip() == version]
    insiders_df = insiders_df[["user", "start_day", "end_day", "scenario"]].reset_index(drop=True)
    
    if return_all:
        return insiders_df, df
    else:
        return insiders_df
    

def build_insider_mask(df: pd.DataFrame, windows: pd.DataFrame) -> pd.Series:
    """
    Returns a boolean mask for any (user, day) pair that falls within a known threat window.
    
    Args:
        df: The UEBA dataset intended for model training
        windows: DataFrame holding the insider windows
        
    Returns:
        pd.Series: A boolean mask where `False=normal` and `True=insider`
    """
    mask = pd.Series(False, index=df.index)
    for _, row in windows.iterrows():
        mask |= (
            (df["user"] == row["user"]) &
            (df["day"] >= row["start_day"]) &
            (df["day"] <= row["end_day"])
        )
    return mask    


def get_scores(newData_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in a raw data and returns a DataFrame with a corresponding anomaly score.
    
    Args:
        newData_df: The raw DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with corresponding anomaly score
    """
    # Use the parameter, work on a copy
    live_df = newData_df.copy()

    # remember user IDs if present
    user_ids = live_df["user"] if "user" in live_df.columns else None
    
    # Conditionally drop only existing columns for scaling
    cols_to_drop = ["user", "pc", "day"]
    existing_cols = [col for col in cols_to_drop if col in live_df.columns]
    live_df.drop(columns=existing_cols, inplace=True)
    
    # Load pre-fitted scaler
    scaler = joblib.load(SCALER_PATH)
    live_scaled = scaler.transform(live_df.values.astype("float32"))
    
    # Load encoder and generate embeddings in batch
    encoder = load_model(ENCODER_PATH)
    embeddings = encoder.predict(live_scaled)
    
    # Load IF and generate anomaly scores
    iforest = UEBAIsolationForest()
    iforest.load(IF_PATH)
    scores = iforest.anomaly_score(embeddings)

    # build a results dataframe
    result_df = pd.DataFrame({"anomaly_score": scores})
    if user_ids is not None:
        # insert user column at front
        result_df.insert(0, "user", user_ids.values)
    # align index with input
    result_df.index = newData_df.index

    return result_df


if __name__ == "__main__":
    # Split Table 
    train_df, test_df = chronological_split(r"processed_datasets\ueba_dataset_3b.csv")

    # Convert to CSV
    train_df.to_csv("processed_datasets/train.csv", index=False)
    test_df.to_csv("processed_datasets/test_stream.csv", index=False)