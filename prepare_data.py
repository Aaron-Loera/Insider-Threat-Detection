# Imports 
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import joblib
from scripts.UEBAIsolationForest import UEBAIsolationForest  
import time
import threading

# Split Data for Simulation
def chronological_split(csv_path, split_ratio=0.9):

    df = pd.read_csv(csv_path, index_col=0)

    # Ensure sorted globally by time
    df = df.sort_values("day")

    unique_days = df["day"].unique()
    cutoff_index = int(len(unique_days) * split_ratio)
    cutoff_day = unique_days[cutoff_index]

    train_df = df[df["day"] <= cutoff_day]
    test_df  = df[df["day"] > cutoff_day]

    return train_df, test_df

# Split Table 
train_df, test_df = chronological_split(r"processed_datasets\ueba_dataset.csv")

# Convert to CSV
train_df.to_csv("processed_datasets/train.csv", index=False)
test_df.to_csv("processed_datasets/test_stream.csv", index=False)

# Paths 
SCALER_PATH = os.path.join(r"encoders\encoder_model_1\feature_scaler.pkl")
ENCODER_PATH = os.path.join(r"encoders\encoder_model_1\encoder_model.keras")
IF_PATH  = os.path.join(r"isolation_forests\iforest_model_1\iforest_model.pkl")

# Take in raw data df then returns a df with userid and anomaly score.
def get_scores(newData_df):
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