# ============================================================
# Data Science Lab - Speaker Age Prediction
#
# train.py:
# 1. Load data
# 2. Extract/load audio features
# 3. Preprocess data
# 4. Compare models
# 5. Tune best model
# 6. Save CSV results and final submission
#
# All plots are created in make_report_outputs.py
# ============================================================

import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance

from src.config import (
    RANDOM_STATE,
    DEVELOPMENT_CSV,
    EVALUATION_CSV,
    DEVELOPMENT_AUDIO_DIR,
    EVALUATION_AUDIO_DIR,
    OUTPUT_DIR,
    TABLES_DIR,
    CACHE_DIR,
)

from src.preprocessing import preprocess_metadata, build_feature_matrix
from src.feature_extraction import extract_audio_features_for_dataframe


warnings.filterwarnings("ignore")


# ============================================================
# Helper function
# ============================================================

def cv_rmse_for_columns(
    X_data: pd.DataFrame,
    y_data: pd.Series,
    columns,
    label: str,
    cv,
):
    """
    Evaluate one feature group using cross-validation RMSE.
    Used for feature-group comparison.
    """

    columns = [c for c in columns if c in X_data.columns]

    if len(columns) == 0:
        return {
            "feature_group": label,
            "num_features": 0,
            "cv_rmse_mean": np.nan,
            "cv_rmse_std": np.nan,
        }

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
    ])

    scores = -cross_val_score(
        model,
        X_data[columns],
        y_data,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    return {
        "feature_group": label,
        "num_features": len(columns),
        "cv_rmse_mean": scores.mean(),
        "cv_rmse_std": scores.std(),
    }


# ============================================================
# 1. Print project paths
# ============================================================

print("Development CSV:", DEVELOPMENT_CSV)
print("Evaluation CSV:", EVALUATION_CSV)
print("Development audio folder:", DEVELOPMENT_AUDIO_DIR)
print("Evaluation audio folder:", EVALUATION_AUDIO_DIR)
print("Output folder:", OUTPUT_DIR)
print("Tables folder:", TABLES_DIR)


# ============================================================
# 2. Load development data
# ============================================================

df_development_raw = pd.read_csv(DEVELOPMENT_CSV)

print("\nDevelopment data shape:", df_development_raw.shape)
print(df_development_raw.head())
print("\nDevelopment columns:")
print(df_development_raw.columns.tolist())


# ============================================================
# 3. Load or extract development audio features
# ============================================================

DEV_AUDIO_FEATURES_PATH = CACHE_DIR / "audio_features_dev.pkl"

if DEV_AUDIO_FEATURES_PATH.exists():
    audio_features_dev = joblib.load(DEV_AUDIO_FEATURES_PATH)
    print("\nLoaded development audio features from cache.")
else:
    print("\nExtracting development audio features...")
    audio_features_dev = extract_audio_features_for_dataframe(
        df_development_raw,
        DEVELOPMENT_AUDIO_DIR,
    )
    joblib.dump(audio_features_dev, DEV_AUDIO_FEATURES_PATH)
    print("Saved development audio features to cache.")

print("Development audio features shape:", audio_features_dev.shape)


# ============================================================
# 4. Preprocess metadata and merge audio features
# ============================================================

df_development = preprocess_metadata(df_development_raw)

df_final = df_development.merge(
    audio_features_dev,
    on="Id",
    how="left",
)

SKEWED_AUDIO_COLS = [
    "aud_rms_mean",
    "aud_spectral_centroid_mean",
    "aud_spectral_rolloff_mean",
]

for col in SKEWED_AUDIO_COLS:
    if col in df_final.columns:
        df_final[col] = pd.to_numeric(df_final[col], errors="coerce")
        df_final[col] = np.log1p(df_final[col].clip(lower=0))


# ============================================================
# 5. Build X and y
# ============================================================

X, y = build_feature_matrix(df_final, has_target=True)

print("\nFinal feature matrix shape:", X.shape)
print("Target shape:", y.shape)
print("Total missing values in X:", int(X.isna().sum().sum()))


# ============================================================
# 6. Train/test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ============================================================
# 7. Baseline model
# ============================================================

baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_test)
baseline_mse = mean_squared_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(baseline_mse)

print(f"\nBaseline test RMSE: {baseline_rmse:.4f}")


# ============================================================
# 8. Model comparison before hyperparameter tuning
# ============================================================

models = {
    "Baseline Mean": DummyRegressor(strategy="mean"),

    "HistGradientBoosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
    ]),

    "ExtraTrees": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ]),
}

comparison_results = []

for name, model in models.items():
    print(f"\nCross-validating: {name}")

    scores = -cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    comparison_results.append({
        "model": name,
        "cv_rmse_mean": scores.mean(),
        "cv_rmse_std": scores.std(),
    })

model_comparison = pd.DataFrame(comparison_results).sort_values("cv_rmse_mean")
model_comparison.to_csv(TABLES_DIR / "model_comparison.csv", index=False)

print("\nModel comparison before tuning:")
print(model_comparison)


# ============================================================
# 9. Automatically tune the best model
# ============================================================

best_model_name = model_comparison.iloc[0]["model"]

print("\nBest model before tuning:", best_model_name)

if best_model_name == "HistGradientBoosting":

    print("\nTuning HistGradientBoostingRegressor...")

    tuning_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
    ])

    param_distributions = {
        "model__learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "model__max_iter": [200, 300, 500, 800],
        "model__max_leaf_nodes": [15, 31, 45, 63],
        "model__min_samples_leaf": [10, 20, 30, 50],
        "model__l2_regularization": [0.0, 0.01, 0.1, 1.0],
        "model__max_bins": [128, 255],
    }

elif best_model_name == "ExtraTrees":

    print("\nTuning ExtraTreesRegressor...")

    tuning_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    param_distributions = {
        "model__n_estimators": [300, 500, 800, 1000],
        "model__max_features": ["sqrt", "log2", 0.5, 0.7, 1.0],
        "model__min_samples_leaf": [1, 2, 3, 5],
        "model__min_samples_split": [2, 5, 10],
        "model__max_depth": [None, 10, 20, 30, 50],
    }

else:
    raise ValueError(
        "Baseline Mean was selected as the best model. "
        "This should not normally happen. Check your features and data."
    )


random_search = RandomizedSearchCV(
    estimator=tuning_model,
    param_distributions=param_distributions,
    n_iter=30,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
)

random_search.fit(X_train, y_train)

best_cv_rmse = -random_search.best_score_

print("\nBest model after tuning:")
print(random_search.best_estimator_)

print("\nBest CV RMSE after tuning:")
print(best_cv_rmse)

print("\nBest parameters:")
print(random_search.best_params_)


# ============================================================
# 10. Final tuned model evaluation on test set
# ============================================================

best_tuned_model = random_search.best_estimator_

y_pred_test = best_tuned_model.predict(X_test)

test_mse = mean_squared_error(y_test, y_pred_test)
test_rmse = np.sqrt(test_mse)

print(f"\nFinal tuned model: {best_model_name}")
print(f"Final test RMSE: {test_rmse:.4f}")
print(f"Final test MSE: {test_mse:.4f}")

final_results = pd.DataFrame([
    {
        "baseline_test_rmse": baseline_rmse,
        "best_model_before_tuning": best_model_name,
        "best_cv_rmse_after_tuning": best_cv_rmse,
        "final_test_rmse": test_rmse,
        "final_test_mse": test_mse,
        "best_parameters": str(random_search.best_params_),
    }
])

final_results.to_csv(TABLES_DIR / "final_results.csv", index=False)

print("\nSaved final results table:")
print(final_results)


# Save true/predicted values so make_report_outputs.py can create plots
prediction_diagnostics = pd.DataFrame({
    "true_age": y_test.values,
    "predicted_age": y_pred_test,
    "error": y_pred_test - y_test.values,
})

prediction_diagnostics.to_csv(
    TABLES_DIR / "prediction_diagnostics.csv",
    index=False,
)

print("\nSaved prediction diagnostics table.")


# ============================================================
# 11. Feature group comparison
# ============================================================

audio_cols = [c for c in X.columns if c.startswith("aud_")]
non_audio_cols = [c for c in X.columns if not c.startswith("aud_")]

pitch_voice_cols = [
    c for c in X.columns
    if any(
        key in c.lower()
        for key in ["pitch", "jitter", "shimmer", "f0", "hnr"]
    )
]

groups = [
    cv_rmse_for_columns(
        X_train,
        y_train,
        non_audio_cols,
        "Provided CSV / metadata features only",
        cv,
    ),
    cv_rmse_for_columns(
        X_train,
        y_train,
        audio_cols,
        "Extracted audio features only",
        cv,
    ),
    cv_rmse_for_columns(
        X_train,
        y_train,
        pitch_voice_cols,
        "Pitch / voice-quality features",
        cv,
    ),
    cv_rmse_for_columns(
        X_train,
        y_train,
        X.columns.tolist(),
        "All features",
        cv,
    ),
]

feature_group_results = pd.DataFrame(groups).sort_values("cv_rmse_mean")
feature_group_results.to_csv(
    TABLES_DIR / "feature_group_comparison.csv",
    index=False,
)

print("\nFeature group comparison:")
print(feature_group_results)


# ============================================================
# 12. Feature importance
# ============================================================

print("\nComputing permutation feature importance...")

perm = permutation_importance(
    best_tuned_model,
    X_test,
    y_test,
    scoring="neg_root_mean_squared_error",
    n_repeats=10,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

importance_df = pd.DataFrame({
    "feature": X_test.columns,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean", ascending=False)

importance_df.to_csv(TABLES_DIR / "feature_importance.csv", index=False)

print("\nTop 20 important features:")
print(importance_df.head(20))


# ============================================================
# 13. Load evaluation data
# ============================================================

evaluation_df_raw = pd.read_csv(EVALUATION_CSV)

if "Id" not in evaluation_df_raw.columns:
    raise ValueError("evaluation.csv must contain an Id column.")

evaluation_df_raw["Id"] = pd.to_numeric(
    evaluation_df_raw["Id"],
    errors="coerce",
).astype(int)

print("\nEvaluation data shape:", evaluation_df_raw.shape)
print(evaluation_df_raw.head())


# ============================================================
# 14. Load or extract evaluation audio features
# ============================================================

EVAL_AUDIO_FEATURES_PATH = CACHE_DIR / "audio_features_eval.pkl"

if EVAL_AUDIO_FEATURES_PATH.exists():
    audio_features_eval = joblib.load(EVAL_AUDIO_FEATURES_PATH)
    print("\nLoaded evaluation audio features from cache.")
else:
    print("\nExtracting evaluation audio features...")
    audio_features_eval = extract_audio_features_for_dataframe(
        evaluation_df_raw,
        EVALUATION_AUDIO_DIR,
    )
    joblib.dump(audio_features_eval, EVAL_AUDIO_FEATURES_PATH)
    print("Saved evaluation audio features to cache.")

print("Evaluation audio features shape:", audio_features_eval.shape)


# ============================================================
# 15. Build evaluation feature matrix
# ============================================================

evaluation_df = preprocess_metadata(evaluation_df_raw)

eval_merged = evaluation_df.merge(
    audio_features_eval,
    on="Id",
    how="left",
)

for col in SKEWED_AUDIO_COLS:
    if col in eval_merged.columns:
        eval_merged[col] = pd.to_numeric(eval_merged[col], errors="coerce")
        eval_merged[col] = np.log1p(eval_merged[col].clip(lower=0))

X_eval = build_feature_matrix(eval_merged, has_target=False)

X_eval = X_eval.reindex(columns=X_train.columns, fill_value=np.nan)

print("\nEvaluation feature matrix shape:", X_eval.shape)
print("Evaluation columns match training:", list(X_eval.columns) == list(X_train.columns))


# ============================================================
# 16. Predict evaluation set and save final submission
# ============================================================

y_pred_eval = best_tuned_model.predict(X_eval)

submission = pd.DataFrame({
    "Id": evaluation_df_raw["Id"],
    "Predicted": y_pred_eval,
})

submission_path = OUTPUT_DIR / "final_submission.csv"
submission.to_csv(submission_path, index=False)

print("\nSaved final submission file:", submission_path)
print(submission.head())
print("Submission shape:", submission.shape)


# ============================================================
# 17. Save compact summary results
# ============================================================

summary_results = pd.DataFrame([
    {"metric": "baseline_test_rmse", "value": baseline_rmse},
    {"metric": "best_cv_rmse_after_tuning", "value": best_cv_rmse},
    {"metric": "final_test_rmse", "value": test_rmse},
    {"metric": "final_test_mse", "value": test_mse},
])

summary_results.to_csv(
    TABLES_DIR / "model_summary_results.csv",
    index=False,
)

print("\nAll training results saved successfully.")
print("Tables saved in:", TABLES_DIR)
print("Submission saved in:", submission_path)
print("\nNow run: python make_report_outputs.py")