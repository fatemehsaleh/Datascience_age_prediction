import numpy as np
import pandas as pd


def preprocess_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Id" in df.columns:
        df["Id"] = pd.to_numeric(df["Id"], errors="coerce").astype("Int64")

    if "gender" in df.columns:
        gender_map = {
            "female": 0,
            "f": 0,
            "woman": 0,
            "male": 1,
            "m": 1,
            "man": 1,
        }

        df["gender"] = (
            df["gender"]
            .astype(str)
            .str.lower()
            .str.strip()
            .map(gender_map)
        )

    if "ethnicity" in df.columns:
        df["ethnicity"] = df["ethnicity"].astype(str).str.strip()
        df = pd.get_dummies(df, columns=["ethnicity"], drop_first=True)

    if "max_pitch" in df.columns and "min_pitch" in df.columns:
        df["pitch_range"] = df["max_pitch"] - df["min_pitch"]

    if "jitter" in df.columns and "shimmer" in df.columns:
        denom = pd.to_numeric(df["shimmer"], errors="coerce").replace(0, np.nan)
        df["jitter_shimmer_ratio"] = pd.to_numeric(df["jitter"], errors="coerce") / denom

    if "mean_pitch" in df.columns and "min_pitch" in df.columns and "max_pitch" in df.columns:
        df["pitch_position"] = (df["mean_pitch"] - df["min_pitch"]) / (
            df["max_pitch"] - df["min_pitch"] + 1e-6
        )

    if "num_characters" in df.columns and "num_words" in df.columns:
        df["chars_per_word"] = df["num_characters"] / (df["num_words"] + 1e-6)

    if "silence_duration" in df.columns and "num_pauses" in df.columns:
        df["silence_per_pause"] = df["silence_duration"] / (df["num_pauses"] + 1e-6)

    return df


def build_feature_matrix(df_final: pd.DataFrame, has_target: bool = True):
    exclude_cols = ["Id", "age", "path"]

    X = df_final.drop(
        columns=[c for c in exclude_cols if c in df_final.columns],
        errors="ignore"
    )

    X = X.select_dtypes(include=["number", "bool"]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    for col in X.select_dtypes(include=["bool"]).columns:
        X[col] = X[col].astype(int)

    if has_target:
        y = df_final["age"].astype(float)
        return X, y

    return X