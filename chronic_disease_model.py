"""
=============================================================================
  Chronic Disease Prediction — End-to-End ML Pipeline
  Author : Senior ML Engineer
  Target : HasChronicDisease  (Yes / No)
  Dataset: chronic_disease_prediction_dataset.csv  (1500 rows x 15 cols)
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. IMPORTING THE DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                        # Non-interactive backend for file saving
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Reproducibility seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Output directory for artefacts
ARTEFACT_DIR = "artefacts"
os.makedirs(ARTEFACT_DIR, exist_ok=True)

print("=" * 65)
print("  CHRONIC DISEASE PREDICTION — FULL ML PIPELINE")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/12] Loading & Cleaning Data …")

df = pd.read_csv("chronic_disease_prediction_dataset.csv")

print(f"  Raw shape       : {df.shape}")
print(f"  Duplicate rows  : {df.duplicated().sum()}")
print(f"  Null values     :\n{df.isnull().sum()}")

# Drop irrelevant identifier column
df.drop(columns=["Patient_ID"], inplace=True)

# Remove duplicate rows (if any)
df.drop_duplicates(inplace=True)

# Impute missing numerics with median, categoricals with mode
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_cols_features = [c for c in cat_cols if c != "HasChronicDisease"]

for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Correct data types — strip whitespace from string columns
for col in cat_cols:
    df[col] = df[col].str.strip()

# Encode binary target: Yes → 1, No → 0
df["HasChronicDisease"] = (df["HasChronicDisease"] == "Yes").astype(int)

print(f"  Cleaned shape   : {df.shape}")
print(f"  Target balance  :\n{df['HasChronicDisease'].value_counts()}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/12] Generating Visualisations …")

PALETTE = {"0": "#4C72B0", "1": "#DD8452"}   # Blue = No, Orange = Yes

# --- 3a. Target class distribution ---
fig, ax = plt.subplots(figsize=(5, 4))
counts = df["HasChronicDisease"].value_counts()
bars = ax.bar(["No Disease (0)", "Has Disease (1)"],
              counts.values,
              color=["#4C72B0", "#DD8452"],
              edgecolor="white", linewidth=1.2, width=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10, str(val),
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_title("Target Class Distribution", fontsize=13, fontweight="bold")
ax.set_ylabel("Count")
ax.set_ylim(0, counts.max() * 1.15)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{ARTEFACT_DIR}/01_target_distribution.png", dpi=150)
plt.close()

# --- 3b. Numeric feature distributions ---
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    axes[i].hist(df[col], bins=25, color="#4C72B0",
                 edgecolor="white", alpha=0.85)
    axes[i].set_title(col, fontsize=10, fontweight="bold")
    axes[i].spines[["top", "right"]].set_visible(False)
# Hide unused subplots
for j in range(len(num_cols), len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Numeric Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{ARTEFACT_DIR}/02_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

# --- 3c. Correlation heatmap (numeric features only) ---
fig, ax = plt.subplots(figsize=(10, 7))
corr_df = df[num_cols + ["HasChronicDisease"]].corr()
mask = np.triu(np.ones_like(corr_df, dtype=bool))
sns.heatmap(corr_df, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, linewidths=0.5,
            annot_kws={"size": 8}, ax=ax)
ax.set_title("Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{ARTEFACT_DIR}/03_correlation_heatmap.png", dpi=150)
plt.close()

print("  Saved: 01_target_distribution, 02_feature_distributions, 03_correlation_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/12] EDA — Feature Relationship Insights …")

# --- 4a. Numeric features vs target (box plots) ---
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    group0 = df[df["HasChronicDisease"] == 0][col]
    group1 = df[df["HasChronicDisease"] == 1][col]
    axes[i].boxplot([group0, group1],
                    patch_artist=True,
                    boxprops=dict(facecolor="#4C72B0", alpha=0.6),
                    medianprops=dict(color="black", linewidth=2))
    # Overlay second group colour
    bp = axes[i].boxplot([group1], patch_artist=True,
                          boxprops=dict(facecolor="#DD8452", alpha=0.7),
                          medianprops=dict(color="black", linewidth=2),
                          positions=[2])
    axes[i].set_xticks([1, 2])
    axes[i].set_xticklabels(["No Disease", "Has Disease"])
    axes[i].set_title(col, fontsize=10, fontweight="bold")
    axes[i].spines[["top", "right"]].set_visible(False)
for j in range(len(num_cols), len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Feature vs. Chronic Disease (Box Plots)", fontsize=14,
             fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{ARTEFACT_DIR}/04_eda_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()

# --- 4b. Categorical features vs target (stacked bar) ---
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
axes = axes.flatten()
for i, col in enumerate(cat_cols_features):
    ct = pd.crosstab(df[col], df["HasChronicDisease"], normalize="index") * 100
    ct.columns = ["No Disease", "Has Disease"]
    ct.plot(kind="bar", stacked=True, ax=axes[i],
            color=["#4C72B0", "#DD8452"],
            edgecolor="white", width=0.6)
    axes[i].set_title(col, fontsize=10, fontweight="bold")
    axes[i].set_xlabel("")
    axes[i].set_ylabel("Percentage (%)")
    axes[i].legend(fontsize=8)
    axes[i].tick_params(axis="x", rotation=30)
    axes[i].spines[["top", "right"]].set_visible(False)
for j in range(len(cat_cols_features), len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Categorical Features vs. Chronic Disease", fontsize=14,
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{ARTEFACT_DIR}/05_eda_categorical.png", dpi=150, bbox_inches="tight")
plt.close()

print("  Saved: 04_eda_boxplots, 05_eda_categorical")


# ─────────────────────────────────────────────────────────────────────────────
# 5. DATA MODELLING — Algorithm Choice
# ─────────────────────────────────────────────────────────────────────────────
# GradientBoostingClassifier is chosen because:
#  • Handles mixed feature types well after encoding
#  • Naturally resistant to outliers via tree splitting
#  • Produces well-calibrated probabilities (important for healthcare)
#  • Consistently outperforms linear models on tabular health data
#  • sklearn-native → no extra dependencies needed

print("\n[4/12] Model Strategy: Gradient Boosting Classifier (GBM) selected")


# ─────────────────────────────────────────────────────────────────────────────
# 6. SPLITTING DATA INTO FEATURES (X) AND TARGET (y)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/12] Feature / Target Split …")

# One-hot encode categoricals for the base model path
df_encoded = pd.get_dummies(df, columns=cat_cols_features, drop_first=False)

X = df_encoded.drop(columns=["HasChronicDisease"])
y = df_encoded["HasChronicDisease"]

print(f"  Feature matrix X : {X.shape}")
print(f"  Target vector  y : {y.shape}  |  Positive rate: {y.mean():.2%}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. TRAIN / TEST SPLIT  (80 – 20, stratified)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/12] Train-Test Split (80 / 20, stratified) …")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)
print(f"  Train : {X_train.shape}  |  Positive: {y_train.mean():.2%}")
print(f"  Test  : {X_test.shape}   |  Positive: {y_test.mean():.2%}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/12] Training Gradient Boosting Classifier …")

gbm = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    min_samples_leaf=10,
    random_state=RANDOM_STATE
)
gbm.fit(X_train, y_train)
print("  Training complete.")


# ─────────────────────────────────────────────────────────────────────────────
# 9. MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8/12] Model Evaluation …")

y_pred  = gbm.predict(X_test)
y_proba = gbm.predict_proba(X_test)[:, 1]

acc     = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"\n  ── Test-set Metrics ──────────────────────────────")
print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f} %)")
print(f"  ROC-AUC  : {roc_auc:.4f}")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Has Disease"]))

# Confusion Matrix plot
fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Has Disease"])
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix — GBM", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{ARTEFACT_DIR}/06_confusion_matrix.png", dpi=150)
plt.close()
print("  Saved: 06_confusion_matrix")


# ─────────────────────────────────────────────────────────────────────────────
# 10. COMMUNICATION & VISUALISATION — ROC Curve
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9/12] ROC Curve …")

fpr, tpr, thresholds = roc_curve(y_test, y_proba)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="#DD8452", lw=2.5,
        label=f"GBM  (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], color="grey", lw=1.2, linestyle="--", label="Random Classifier")
ax.fill_between(fpr, tpr, alpha=0.08, color="#DD8452")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curve — Chronic Disease Prediction", fontsize=12, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{ARTEFACT_DIR}/07_roc_curve.png", dpi=150)
plt.close()
print("  Saved: 07_roc_curve")


# ─────────────────────────────────────────────────────────────────────────────
# 11. MODEL SAVING  (.sav — Joblib format)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[10/12] Saving Trained Model …")

MODEL_PATH = f"{ARTEFACT_DIR}/chronic_disease_gbm_model.sav"
joblib.dump(gbm, MODEL_PATH)
print(f"  Model saved → {MODEL_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# 12. AUTOMATED PIPELINE (Scikit-learn Pipeline — scaling + preprocessing)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[11/12] Building & Training Automated Preprocessing Pipeline …")

# Use the RAW (un-encoded) feature set for the Pipeline path so that
# ColumnTransformer handles encoding internally — production-ready pattern.
X_raw = df.drop(columns=["HasChronicDisease"])
y_raw = df["HasChronicDisease"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_raw, y_raw,
    test_size=0.20,
    stratify=y_raw,
    random_state=RANDOM_STATE
)

# Identify column groups inside the raw DataFrame
numeric_features     = X_raw.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_raw.select_dtypes(include="object").columns.tolist()

# Numeric sub-pipeline: impute → scale
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])

# Categorical sub-pipeline: impute → ordinal encode
#   OrdinalEncoder is used here; for tree models it is equivalent to
#   one-hot encoding and keeps dimensionality low.
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

# Combine into a ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer,     numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Full end-to-end Pipeline
full_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier",   GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=RANDOM_STATE
    ))
])

full_pipeline.fit(X_train_r, y_train_r)
pipe_acc = accuracy_score(y_test_r, full_pipeline.predict(X_test_r))
print(f"  Pipeline test accuracy : {pipe_acc:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 13. PIPELINE SAVING (compressed format)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[12/12] Saving Pipeline (compressed) …")

PIPELINE_PATH = f"{ARTEFACT_DIR}/chronic_disease_pipeline.pkl.gz"
joblib.dump(full_pipeline, PIPELINE_PATH, compress=("gzip", 3))
print(f"  Pipeline saved → {PIPELINE_PATH}")


# =============================================================================
# ADVANCED FEATURE 1 — Cross-Validation (Stratified K-Fold, k=5)
# =============================================================================
print("\n──────────────────────────────────────────────────────────")
print("  ADVANCED 1 — Stratified 5-Fold Cross-Validation")
print("──────────────────────────────────────────────────────────")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cv_acc = cross_val_score(full_pipeline, X_raw, y_raw,
                         cv=cv, scoring="accuracy", n_jobs=-1)
cv_auc = cross_val_score(full_pipeline, X_raw, y_raw,
                         cv=cv, scoring="roc_auc", n_jobs=-1)

print(f"  CV Accuracy  : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}  "
      f"(folds: {np.round(cv_acc, 4)})")
print(f"  CV ROC-AUC   : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}  "
      f"(folds: {np.round(cv_auc, 4)})")

# CV score visualisation
fig, ax = plt.subplots(figsize=(7, 4))
folds = [f"Fold {i+1}" for i in range(5)]
x = np.arange(5)
width = 0.35
bars1 = ax.bar(x - width/2, cv_acc, width, label="Accuracy", color="#4C72B0", alpha=0.85)
bars2 = ax.bar(x + width/2, cv_auc, width, label="ROC-AUC",  color="#DD8452", alpha=0.85)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.005, f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.005, f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(folds)
ax.set_ylim(0.5, 1.05)
ax.set_title("5-Fold Cross-Validation — Accuracy & ROC-AUC", fontsize=12, fontweight="bold")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{ARTEFACT_DIR}/08_cross_validation.png", dpi=150)
plt.close()
print("  Saved: 08_cross_validation")


# =============================================================================
# ADVANCED FEATURE 2 — Hyperparameter Tuning (RandomizedSearchCV)
# =============================================================================
print("\n──────────────────────────────────────────────────────────")
print("  ADVANCED 2 — Hyperparameter Tuning (RandomizedSearchCV)")
print("──────────────────────────────────────────────────────────")

param_dist = {
    "classifier__n_estimators"  : [100, 150, 200, 300],
    "classifier__learning_rate" : [0.01, 0.05, 0.1, 0.15],
    "classifier__max_depth"     : [3, 4, 5],
    "classifier__subsample"     : [0.7, 0.8, 0.9, 1.0],
    "classifier__min_samples_leaf": [5, 10, 15, 20]
}

random_search = RandomizedSearchCV(
    full_pipeline,
    param_distributions=param_dist,
    n_iter=20,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring="roc_auc",
    n_jobs=-1,
    verbose=0,
    random_state=RANDOM_STATE
)
random_search.fit(X_train_r, y_train_r)

best_params = random_search.best_params_
best_cv_auc = random_search.best_score_

print(f"  Best CV ROC-AUC  : {best_cv_auc:.4f}")
print("  Best Parameters  :")
for k, v in best_params.items():
    print(f"    {k:<45} = {v}")

best_pipeline = random_search.best_estimator_
tuned_acc = accuracy_score(y_test_r, best_pipeline.predict(X_test_r))
tuned_auc = roc_auc_score(y_test_r, best_pipeline.predict_proba(X_test_r)[:, 1])
print(f"\n  Tuned Model — Test Accuracy : {tuned_acc:.4f}")
print(f"  Tuned Model — Test ROC-AUC  : {tuned_auc:.4f}")

# Save the tuned best pipeline (overwrite with best version)
BEST_PIPELINE_PATH = f"{ARTEFACT_DIR}/chronic_disease_best_pipeline.pkl.gz"
joblib.dump(best_pipeline, BEST_PIPELINE_PATH, compress=("gzip", 3))
print(f"  Best pipeline saved → {BEST_PIPELINE_PATH}")


# =============================================================================
# ADVANCED FEATURE 3 — Feature Importance Analysis
# =============================================================================
print("\n──────────────────────────────────────────────────────────")
print("  ADVANCED 3 — Feature Importance Analysis")
print("──────────────────────────────────────────────────────────")

# Extract feature names after ColumnTransformer
num_feat_names = numeric_features
cat_feat_names = list(
    best_pipeline.named_steps["preprocessor"]
    .transformers_[1][1]
    .named_steps["encoder"]
    .get_feature_names_out(categorical_features)
)
all_feature_names = num_feat_names + cat_feat_names

importances = best_pipeline.named_steps["classifier"].feature_importances_
feat_imp_df = pd.DataFrame({
    "Feature"   : all_feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False).reset_index(drop=True)

print("\n  Top 10 Most Important Features:")
print(feat_imp_df.head(10).to_string(index=False))

# Feature importance bar chart
top_n = 15
top_feats = feat_imp_df.head(top_n)
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(top_feats["Feature"][::-1],
               top_feats["Importance"][::-1],
               color="#4C72B0", edgecolor="white", alpha=0.9)
for bar, val in zip(bars, top_feats["Importance"][::-1]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=8)
ax.set_xlabel("Importance Score (Mean Decrease in Impurity)", fontsize=10)
ax.set_title(f"Top {top_n} Feature Importances — GBM", fontsize=12, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{ARTEFACT_DIR}/09_feature_importance.png", dpi=150)
plt.close()
print("  Saved: 09_feature_importance")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("  PIPELINE COMPLETE — SUMMARY")
print("=" * 65)
print(f"  Base  GBM  Accuracy  : {acc:.4f}  |  AUC : {roc_auc:.4f}")
print(f"  Tuned GBM  Accuracy  : {tuned_acc:.4f}  |  AUC : {tuned_auc:.4f}")
print(f"  CV    Mean Accuracy  : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
print(f"  CV    Mean ROC-AUC   : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
print(f"\n  Artefacts saved in  : ./{ARTEFACT_DIR}/")
print("=" * 65)
