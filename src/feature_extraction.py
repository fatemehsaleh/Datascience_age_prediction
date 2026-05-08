from pathlib import Path

import numpy as np
import pandas as pd
import librosa


def resolve_audio_path(row: pd.Series, audio_dir: Path) -> Path:
    """
    Find the correct audio file path for one sample.

    It first tries the path column.
    If that does not work, it tries Id + common audio extensions.
    """

    if "path" in row and pd.notna(row["path"]):
        raw_path = Path(str(row["path"]))

        if raw_path.exists():
            return raw_path

        candidate = audio_dir / raw_path.name
        if candidate.exists():
            return candidate

    sample_id = str(row["Id"])

    possible_extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]

    for ext in possible_extensions:
        candidate = audio_dir / f"{sample_id}{ext}"
        if candidate.exists():
            return candidate

    return audio_dir / str(row.get("path", ""))


def extract_features_age_core(
    file_path: str,
    n_mfcc: int = 13,
    frame_length: int = 2048,
    hop_length: int = 512
) -> dict:
    """
    Extract audio features from one audio file.

    Features include:
    - duration
    - sampling rate
    - RMS energy
    - zero-crossing rate
    - spectral centroid
    - spectral bandwidth
    - spectral rolloff
    - MFCCs
    - MFCC deltas
    - estimated pitch / f0
    """

    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)

        if y is None or len(y) < 10:
            return {"aud_duration_audio": np.nan}

        duration = float(librosa.get_duration(y=y, sr=sr))

        def mean_std(arr):
            arr = np.asarray(arr)
            return float(np.mean(arr)), float(np.std(arr))

        def summarize_extended(arr, prefix):
            arr = np.asarray(arr)

            return {
                f"{prefix}_mean": float(np.mean(arr)),
                f"{prefix}_std": float(np.std(arr)),
                f"{prefix}_p25": float(np.percentile(arr, 25)),
                f"{prefix}_p75": float(np.percentile(arr, 75)),
                f"{prefix}_iqr": float(
                    np.percentile(arr, 75) - np.percentile(arr, 25)
                ),
            }

        feats = {
            "aud_duration_audio": duration,
            "aud_sampling_rate_audio": float(sr),
        }

        # Energy
        rms = librosa.feature.rms(
            y=y,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            y,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

        # Spectral features
        centroid = librosa.feature.spectral_centroid(
            y=y,
            sr=sr,
            hop_length=hop_length
        )[0]

        bandwidth = librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr,
            hop_length=hop_length
        )[0]

        rolloff = librosa.feature.spectral_rolloff(
            y=y,
            sr=sr,
            hop_length=hop_length,
            roll_percent=0.85
        )[0]

        feats["aud_rms_mean"], feats["aud_rms_std"] = mean_std(rms)
        feats["aud_zcr_mean"], feats["aud_zcr_std"] = mean_std(zcr)

        feats["aud_spectral_centroid_mean"], feats["aud_spectral_centroid_std"] = mean_std(centroid)
        feats["aud_spectral_bandwidth_mean"], feats["aud_spectral_bandwidth_std"] = mean_std(bandwidth)
        feats["aud_spectral_rolloff_mean"], feats["aud_spectral_rolloff_std"] = mean_std(rolloff)

        feats.update(summarize_extended(rms, "aud_rms"))
        feats.update(summarize_extended(zcr, "aud_zcr"))
        feats.update(summarize_extended(centroid, "aud_spectral_centroid"))
        feats.update(summarize_extended(bandwidth, "aud_spectral_bandwidth"))
        feats.update(summarize_extended(rolloff, "aud_spectral_rolloff"))

        # MFCC features
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=n_mfcc,
            hop_length=hop_length
        )

        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        for i in range(n_mfcc):
            feats[f"aud_mfcc_{i + 1}_mean"] = float(np.mean(mfcc[i]))
            feats[f"aud_mfcc_{i + 1}_std"] = float(np.std(mfcc[i]))

            feats[f"aud_mfcc_delta_{i + 1}_mean"] = float(np.mean(mfcc_delta[i]))
            feats[f"aud_mfcc_delta_{i + 1}_std"] = float(np.std(mfcc_delta[i]))

            feats[f"aud_mfcc_delta2_{i + 1}_mean"] = float(np.mean(mfcc_delta2[i]))
            feats[f"aud_mfcc_delta2_{i + 1}_std"] = float(np.std(mfcc_delta2[i]))

        # Pitch / fundamental frequency
        try:
            f0 = librosa.yin(
                y,
                fmin=50,
                fmax=500,
                sr=sr,
                frame_length=frame_length,
                hop_length=hop_length
            )

            f0 = np.asarray(f0)
            f0 = f0[np.isfinite(f0)]

            if len(f0) > 0:
                feats.update(summarize_extended(f0, "aud_f0"))
                feats["aud_f0_range"] = float(np.max(f0) - np.min(f0))
            else:
                feats["aud_f0_mean"] = np.nan
                feats["aud_f0_std"] = np.nan
                feats["aud_f0_range"] = np.nan

        except Exception:
            feats["aud_f0_mean"] = np.nan
            feats["aud_f0_std"] = np.nan
            feats["aud_f0_range"] = np.nan

        return feats

    except Exception as e:
        print(f"Audio extraction failed for {file_path}: {e}")
        return {"aud_duration_audio": np.nan}


def extract_audio_features_for_dataframe(
    df: pd.DataFrame,
    audio_dir: Path
) -> pd.DataFrame:
    """
    Extract audio features for every row in the dataframe.
    Returns one dataframe where each row corresponds to one audio sample.
    """

    rows = []

    for idx, row in df.iterrows():
        sample_id = row["Id"]
        audio_path = resolve_audio_path(row, audio_dir)

        feats = extract_features_age_core(str(audio_path))
        feats["Id"] = sample_id

        rows.append(feats)

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(df)} audio files")

    return pd.DataFrame(rows)