# ============================================================
# Make report tables and figures
# Run this file AFTER train.py has created outputs/tables/
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    DEVELOPMENT_CSV,
    OUTPUT_DIR,
    TABLES_DIR,
)

# ============================================================
# Paths
# ============================================================

REPORT_DIR = OUTPUT_DIR / "report_artifacts"
REPORT_FIGURES_DIR = REPORT_DIR / "figures"
REPORT_TABLES_DIR = REPORT_DIR / "tables"

REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helper functions
# ============================================================

def save_plot(filename):
    path = REPORT_FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {path}")


def save_table(df, filename):
    path = REPORT_TABLES_DIR / filename
    df.to_csv(path, index=False)
    print(f"Saved table: {path}")


# ============================================================
# 1. Dataset overview table
# ============================================================

df = pd.read_csv(DEVELOPMENT_CSV)

dataset_summary = pd.DataFrame([
    {"item": "Number of samples", "value": len(df)},
    {"item": "Number of columns", "value": df.shape[1]},
    {"item": "Minimum age", "value": df["age"].min()},
    {"item": "Maximum age", "value": df["age"].max()},
    {"item": "Mean age", "value": round(df["age"].mean(), 2)},
    {"item": "Median age", "value": round(df["age"].median(), 2)},
    {"item": "Age standard deviation", "value": round(df["age"].std(), 2)},
])

save_table(dataset_summary, "dataset_summary.csv")

print("\nDataset summary:")
print(dataset_summary)


# ============================================================
# 2. Age distribution plot
# ============================================================

plt.figure(figsize=(8, 5))
sns.histplot(df["age"], bins=30, kde=True)
plt.xlabel("Age")
plt.ylabel("Number of samples")
plt.title("Distribution of Speaker Age")
save_plot("age_distribution.png")


# ============================================================
# 3. Gender vs age plot
# ============================================================

if "gender" in df.columns:
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="gender", y="age")
    plt.xlabel("Gender")
    plt.ylabel("Age")
    plt.title("Age Distribution by Gender")
    save_plot("age_by_gender.png")
else:
    print("gender column not found. Skipping age_by_gender.png")


# ============================================================
# 4. Correlation heatmap
# ============================================================

corr_cols = [
    "age",
    "mean_pitch",
    "max_pitch",
    "min_pitch",
    "jitter",
    "shimmer",
    "energy",
    "zcr_mean",
    "spectral_centroid_mean",
    "tempo",
    "hnr",
    "num_words",
    "num_characters",
    "num_pauses",
    "silence_duration",
]

corr_cols = [c for c in corr_cols if c in df.columns]

print("\nColumns used for heatmap:")
print(corr_cols)

if len(corr_cols) > 1:
    corr_df = df[corr_cols].copy()

    for col in corr_df.columns:
        corr_df[col] = pd.to_numeric(corr_df[col], errors="coerce")

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        corr_df.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
    )
    plt.title("Correlation Matrix of Main Numerical Features")
    save_plot("correlation_heatmap.png")
else:
    print("Heatmap not created: not enough numeric columns found.")


# ============================================================
# 5. Model comparison table and graph
# ============================================================

model_comparison_path = TABLES_DIR / "model_comparison.csv"

if model_comparison_path.exists():
    model_comparison = pd.read_csv(model_comparison_path)

    save_table(model_comparison, "model_comparison_report.csv")

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=model_comparison,
        x="cv_rmse_mean",
        y="model"
    )
    plt.xlabel("Mean CV RMSE")
    plt.ylabel("Model")
    plt.title("Model Comparison using Cross-Validation RMSE")
    save_plot("model_comparison_rmse.png")

    print("\nModel comparison:")
    print(model_comparison)
else:
    print("model_comparison.csv not found. Run train.py first.")


# ============================================================
# 6. Feature group comparison table and graph
# ============================================================

feature_group_path = TABLES_DIR / "feature_group_comparison.csv"

if feature_group_path.exists():
    feature_group_results = pd.read_csv(feature_group_path)

    save_table(feature_group_results, "feature_group_comparison_report.csv")

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=feature_group_results,
        x="cv_rmse_mean",
        y="feature_group"
    )
    plt.xlabel("Mean CV RMSE")
    plt.ylabel("Feature Group")
    plt.title("Feature Group Comparison")
    save_plot("feature_group_comparison.png")

    print("\nFeature group comparison:")
    print(feature_group_results)
else:
    print("feature_group_comparison.csv not found. Run train.py first.")


# ============================================================
# 7. Feature importance table and graph
# ============================================================

feature_importance_path = TABLES_DIR / "feature_importance.csv"

if feature_importance_path.exists():
    importance_df = pd.read_csv(feature_importance_path)

    top20_importance = importance_df.head(20)

    save_table(top20_importance, "top20_feature_importance_report.csv")

    plt.figure(figsize=(10, 7))
    top_imp = top20_importance.iloc[::-1]

    plt.barh(top_imp["feature"], top_imp["importance_mean"])
    plt.xlabel("Permutation Importance")
    plt.title("Top 20 Feature Importances")
    save_plot("feature_importance_top20.png")

    print("\nTop 20 feature importances:")
    print(top20_importance)
else:
    print("feature_importance.csv not found. Run train.py first.")


# ============================================================
# 8. True vs predicted + error distribution
# ============================================================

prediction_path = TABLES_DIR / "prediction_diagnostics.csv"

if prediction_path.exists():
    pred_df = pd.read_csv(prediction_path)

    save_table(pred_df, "prediction_diagnostics_report.csv")

    plt.figure(figsize=(6, 6))
    plt.scatter(pred_df["true_age"], pred_df["predicted_age"], alpha=0.6)

    min_age = min(pred_df["true_age"].min(), pred_df["predicted_age"].min())
    max_age = max(pred_df["true_age"].max(), pred_df["predicted_age"].max())

    plt.plot(
        [min_age, max_age],
        [min_age, max_age],
        linestyle="--"
    )

    plt.xlabel("True Age")
    plt.ylabel("Predicted Age")
    plt.title("True vs Predicted Age")
    save_plot("true_vs_predicted_age.png")

    plt.figure(figsize=(8, 5))
    sns.histplot(pred_df["error"], kde=True)
    plt.xlabel("Prediction Error: Predicted - True")
    plt.ylabel("Number of samples")
    plt.title("Prediction Error Distribution")
    save_plot("error_distribution.png")

    print("\nPrediction diagnostics:")
    print(pred_df.head())
else:
    print(
        "prediction_diagnostics.csv not found. "
        "Add prediction_diagnostics saving in train.py and rerun train.py."
    )


# ============================================================
# 9. Final results table
# ============================================================

final_results_path = TABLES_DIR / "final_results.csv"

if final_results_path.exists():
    final_results = pd.read_csv(final_results_path)

    save_table(final_results, "final_results_report.csv")

    print("\nFinal results:")
    print(final_results)
else:
    print("final_results.csv not found. Run train.py first.")


# ============================================================
# 10. Model summary table
# ============================================================

model_summary_path = TABLES_DIR / "model_summary_results.csv"

if model_summary_path.exists():
    model_summary = pd.read_csv(model_summary_path)

    save_table(model_summary, "model_summary_results_report.csv")

    print("\nModel summary:")
    print(model_summary)
else:
    print("model_summary_results.csv not found. Run train.py first.")


# ============================================================
# 11. Final message
# ============================================================

print("\nReport artifacts created successfully.")
print("All figures saved in:", REPORT_FIGURES_DIR)
print("All tables saved in:", REPORT_TABLES_DIR)

print("\nExpected figures:")
print("- age_distribution.png")
print("- age_by_gender.png")
print("- correlation_heatmap.png")
print("- model_comparison_rmse.png")
print("- feature_group_comparison.png")
print("- feature_importance_top20.png")
print("- true_vs_predicted_age.png")
print("- error_distribution.png")