#!/usr/bin/env python3
import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from types import SimpleNamespace

import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAVE_SGKF = True
except ImportError:
    HAVE_SGKF = False

# -------------------------
# SETTINGS
# -------------------------
CONFIG = {
    "mode": "tabular",
    "input": "../output/openpose_dataset_windows.csv", 
    "out": "../output/models_output",
    "tag": "openpose_fitness_v2",
}

RANDOM_STATE = 42327
CV_SPLITS = 5
N_ITER_SEARCH = 15 # Reduced slightly for faster local execution
SCORING = "f1_weighted"
N_JOBS = -1

# Models that require normalized data
NEED_SCALING = {"SVM", "KNN", "MLP_NeuralNet"}

# -------------------------
# Model Configurations
# -------------------------
def make_base_models():
    return {
        "RandomForest": RandomForestClassifier(
            random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "KNN": KNeighborsClassifier(),
        "MLP_NeuralNet": MLPClassifier(
            max_iter=1000,
            random_state=RANDOM_STATE,
            early_stopping=True,
        ),
    }

def make_param_distributions(model_name):
    # Simplified search spaces to avoid "total space smaller than n_iter" warnings
    if model_name == "RandomForest":
        return {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [None, 10, 20],
        }
    if model_name == "XGBoost":
        return {
            "clf__n_estimators": [100, 300],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [3, 6],
        }
    if model_name == "SVM":
        return {
            "clf__C": [1.0, 10.0],
            "clf__gamma": ["scale", 0.1],
        }
    if model_name == "KNN":
        return {
            "clf__n_neighbors": [3, 5, 9],
            "clf__weights": ["uniform", "distance"],
        }
    if model_name == "MLP_NeuralNet":
        return {
            "clf__hidden_layer_sizes": [(64, 32), (128, 64)],
            "clf__alpha": [1e-4, 1e-3],
        }
    return {}

# -------------------------
# Helpers
# -------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_confusion_matrix(cm, class_names, out_path, title):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# -------------------------
# Data Loading
# -------------------------
def load_tabular(csv_path: str):
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("CSV must contain 'label' column.")

    # Use video_id for grouping to prevent data leakage
    if "video_id" in df.columns:
        groups = df["video_id"].astype(str)
    else:
        groups = df["file"].astype(str) if "file" in df.columns else np.arange(len(df))

    # Meta columns to exclude from training
    non_feature = {
        "label", "video_id", "file", "json_path", 
        "start_frame", "end_frame", "valid_frac",
        "fps_target", "fps_assumed", "stride"
    }
    feature_cols = [c for c in df.columns if c not in non_feature and pd.api.types.is_numeric_dtype(df[c])]

    X = df[feature_cols].copy()
    y = df["label"].astype(str).copy()
    return df, X, y, groups, feature_cols

# -------------------------
# Training logic with Imputer
# -------------------------
def run_benchmark(X, y_encoded, groups, class_names, feature_names, out_dir, tag):
    ensure_dir(out_dir)
    
    # Stratified split ensures all exercises are represented in both sets
    if HAVE_SGKF:
        cv_split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    else:
        cv_split = GroupKFold(n_splits=5)
        
    train_idx, test_idx = next(cv_split.split(X, y_encoded, groups=groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    groups_train = groups.iloc[train_idx]

    models = make_base_models()
    results = []

    for name, estimator in models.items():
        print(f"\n>>> Processing Model: {name}")
        
        # PIPELINE: 1. Impute NaNs -> 2. Scale (if needed) -> 3. Classify
        # We use median strategy as it's more robust to outliers in pose data
        steps = [("imputer", SimpleImputer(strategy="median"))]
        
        if name in NEED_SCALING:
            steps.append(("scaler", StandardScaler()))
            
        steps.append(("clf", estimator))
        pipe = Pipeline(steps)
        
        # Hyperparameter Search
        search = RandomizedSearchCV(
            pipe, make_param_distributions(name), 
            n_iter=N_ITER_SEARCH,
            cv=GroupKFold(n_splits=3), 
            scoring=SCORING, 
            n_jobs=N_JOBS, 
            random_state=RANDOM_STATE,
            error_score='raise' # Helpful for debugging
        )
        
        try:
            if name == "XGBoost":
                w = compute_sample_weight(class_weight="balanced", y=y_train)
                search.fit(X_train, y_train, groups=groups_train, clf__sample_weight=w)
            else:
                search.fit(X_train, y_train, groups=groups_train)

            best_model = search.best_estimator_
            preds = best_model.predict(X_test)
            
            acc = accuracy_score(y_test, preds)
            _, _, f1_w, _ = precision_recall_fscore_support(y_test, preds, average="weighted", zero_division=0)
            
            print(f"Success! Accuracy: {acc:.2%} | F1-Score: {f1_w:.4f}")
            
            # Save Model and Confusion Matrix
            joblib.dump(best_model, os.path.join(out_dir, f"{tag}_{name.lower()}.pkl"))
            cm = confusion_matrix(y_test, preds)
            save_confusion_matrix(cm, class_names, os.path.join(out_dir, f"cm_{name.lower()}.png"), f"{name} Confusion Matrix")
            
            results.append({"Model": name, "Accuracy": acc, "F1_Weighted": f1_w})
            
        except Exception as e:
            print(f"FAILED to train {name}: {e}")

    # Output Leaderboard
    if results:
        res_df = pd.DataFrame(results).sort_values("F1_Weighted", ascending=False)
        res_df.to_csv(os.path.join(out_dir, "leaderboard.csv"), index=False)
        print("\n--- FINAL LEADERBOARD ---")
        print(res_df.to_string(index=False))

# -------------------------
# Main Execution
# -------------------------
def main():
    args = SimpleNamespace(**CONFIG)
    
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        return

    df, X, y, groups, feat_names = load_tabular(args.input)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = list(le.classes_)
    
    out_path = os.path.join(args.out, args.tag)
    ensure_dir(out_path)
    joblib.dump(le, os.path.join(out_path, "label_encoder.pkl"))
    
    run_benchmark(X, y_encoded, groups, class_names, feat_names, out_path, args.tag)

if __name__ == "__main__":
    main()