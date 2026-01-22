#!/usr/bin/env python3
import os
import time
import json
import argparse
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAVE_SGKF = True
except Exception:
    HAVE_SGKF = False

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

# -------------------------
# SETTINGS (edit here)
# -------------------------
CONFIG = {
    "mode": "tabular",  # "tabular" | "sequence"

    # tabular
    "input": "../data/mediapipe_dataset.csv",

    # sequence
    "npz": "../output/mediapipe_pose_npz/dataset_windows.npz",
    "meta": "../output/mediapipe_pose_npz/dataset_windows_meta.csv",
    "labels": "../output/mediapipe_pose_npz/labels.csv",

    # outputs
    "out": "../output/models_output",
    "tag": None,  # e.g. "features_pp_v1"
}

# -------------------------
# Defaults
# -------------------------
RANDOM_STATE = 42327

CV_SPLITS = 5
N_ITER_SEARCH = 25
SCORING = "f1_weighted"
N_JOBS = -1

NEED_SCALING = {"SVM", "KNN", "MLP_NeuralNet"}

# -------------------------
# Models + Search spaces
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
            max_iter=2000,
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1,
        ),
    }


def make_param_distributions(model_name):
    if model_name == "RandomForest":
        return {
            "clf__n_estimators": [200, 400, 800],
            "clf__max_depth": [None, 5, 10, 20],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", "log2", None],
        }
    if model_name == "XGBoost":
        return {
            "clf__n_estimators": [200, 400, 800],
            "clf__learning_rate": [0.03, 0.05, 0.1, 0.2],
            "clf__max_depth": [3, 5, 7, 9],
            "clf__subsample": [0.7, 0.85, 1.0],
            "clf__colsample_bytree": [0.7, 0.85, 1.0],
            "clf__reg_lambda": [0.5, 1.0, 2.0, 5.0],
        }
    if model_name == "SVM":
        return {
            "clf__C": [0.5, 1.0, 3.0, 10.0, 30.0],
            "clf__gamma": ["scale", 0.01, 0.03, 0.1, 0.3],
        }
    if model_name == "KNN":
        return {
            "clf__n_neighbors": [3, 5, 7, 11, 15, 21],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],
        }
    if model_name == "MLP_NeuralNet":
        return {
            "clf__hidden_layer_sizes": [(128, 64), (128, 64, 32), (256, 128, 64)],
            "clf__alpha": [1e-5, 1e-4, 1e-3],
            "clf__learning_rate_init": [1e-4, 5e-4, 1e-3],
            "clf__activation": ["relu", "tanh"],
        }
    raise ValueError(f"Unknown model: {model_name}")


def make_pipeline(model_name, estimator):
    if model_name in NEED_SCALING:
        return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    return Pipeline([("clf", estimator)])


def fit_with_optional_sample_weight(model_name, search_obj, X_train, y_train, groups_train):
    # Group-aware CV uses groups passed to fit()
    if model_name == "XGBoost":
        w = compute_sample_weight(class_weight="balanced", y=y_train)
        search_obj.fit(X_train, y_train, groups=groups_train, clf__sample_weight=w)
    else:
        search_obj.fit(X_train, y_train, groups=groups_train)


# -------------------------
# Helpers
# -------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def safe_group_kfold(groups_train, desired_splits: int):
    n_groups = pd.Series(groups_train).nunique()
    max_splits = max(2, min(desired_splits, n_groups))
    return GroupKFold(n_splits=max_splits), max_splits


def save_confusion_matrix(cm, class_names, out_path, title):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=90)
    plt.yticks(ticks=np.arange(len(class_names)), labels=class_names)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def per_class_group_counts(y_lbl, groups_series):
    return (
        pd.DataFrame({"label": y_lbl, "group": groups_series.values})
        .groupby("label")["group"]
        .nunique()
        .sort_values(ascending=False)
    )


def split_train_test_group_stratified(X, y_enc, groups, seed=RANDOM_STATE):
    """
    Returns train_idx, test_idx with zero group overlap.
    Prefers StratifiedGroupKFold if available; otherwise falls back to GroupShuffleSplit-like heuristic.
    """
    if HAVE_SGKF:
        print(f"[INFO] Using StratifiedGroupKFold for train/test split")
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        train_idx, test_idx = next(sgkf.split(X, y_enc, groups=groups))
        return train_idx, test_idx

    # Fallback: approximate stratification by sampling groups per class
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"y": y_enc, "g": groups.values})
    # choose ~20% of groups per class as test (min 1)
    test_groups = set()
    for cls, gdf in df.groupby("y"):
        unique_groups = gdf["g"].unique()
        rng.shuffle(unique_groups)
        k = max(1, int(round(0.2 * len(unique_groups))))
        test_groups.update(unique_groups[:k])

    mask_test = groups.isin(test_groups).values
    test_idx = np.where(mask_test)[0]
    train_idx = np.where(~mask_test)[0]
    return train_idx, test_idx


# -------------------------
# Data loading
# -------------------------
def load_tabular(csv_path: str):
    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        raise ValueError(f"CSV must contain 'label'. Found columns: {df.columns.tolist()}")

    # group column preference: npz_source (video-level) > video_path > fallback
    if "npz_source" in df.columns:
        groups = df["npz_source"].astype(str)
    elif "video_path" in df.columns:
        groups = df["video_path"].astype(str)
    else:
        # last resort: if nothing else exists, this risks leakage
        raise ValueError("Need a group column to avoid leakage. Add 'npz_source' or 'video_path' to your CSV.")

    # drop non-features
    non_feature = {"label", "npz_source", "video_path", "start_frame", "end_frame", "valid_frac"}
    feature_cols = [c for c in df.columns if c not in non_feature]

    if not feature_cols:
        raise ValueError("No feature columns found after dropping metadata columns.")

    X = df[feature_cols].copy()
    y = df["label"].astype(str).copy()
    return df, X, y, groups, feature_cols


def load_sequence(npz_path: str, meta_csv: str, labels_csv: str):
    data = np.load(npz_path, allow_pickle=True)
    if "X" not in data or "y" not in data:
        raise ValueError(f"{npz_path} must contain arrays X and y.")

    Xseq = data["X"]  # (N, WIN, 33, 4)
    y = data["y"].astype(int)  # label_id already

    meta = pd.read_csv(meta_csv)
    if "npz_source" not in meta.columns:
        raise ValueError(f"{meta_csv} must contain 'npz_source' for grouping.")
    if len(meta) != len(y):
        raise ValueError(f"Meta rows ({len(meta)}) must match N in dataset ({len(y)}).")

    groups = meta["npz_source"].astype(str)

    labels = pd.read_csv(labels_csv)
    if not {"label", "label_id"}.issubset(labels.columns):
        raise ValueError(f"{labels_csv} must contain columns label,label_id")

    # Flatten sequences for classical models
    X = Xseq.reshape(Xseq.shape[0], -1).astype(np.float32)
    feature_cols = [f"x{i}" for i in range(X.shape[1])]

    # Build label encoder compatible object (so you still get class names)
    # Here y is already ids. We'll create class_names list by sorting label_id.
    labels_sorted = labels.sort_values("label_id")
    class_names = labels_sorted["label"].astype(str).tolist()

    return meta, X, y, groups, feature_cols, class_names


# -------------------------
# Training core
# -------------------------
def run_training(
    X,
    y_encoded,
    groups,
    class_names,
    feature_names,
    output_dir,
    dataset_tag,
):
    ensure_dir(output_dir)

    # Save feature names + class names
    joblib.dump(feature_names, os.path.join(output_dir, "feature_names.pkl"))
    with open(os.path.join(output_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    # Holdout split (grouped, stratified if possible)
    train_idx, test_idx = split_train_test_group_stratified(X, y_encoded, groups)

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

    overlap = set(groups_train.unique()) & set(groups_test.unique())

    print(f"\nSplit sizes (samples): train={len(train_idx)} test={len(test_idx)}")
    print(f"Split sizes (groups/videos): train={groups_train.nunique()} test={groups_test.nunique()}")
    print(f"Group overlap (should be 0): {len(overlap)}")

    # Diagnostics: groups per class
    y_train_lbl = [class_names[i] for i in y_train]
    y_test_lbl = [class_names[i] for i in y_test]

    print("\n#Groups per class (train):")
    print(per_class_group_counts(y_train_lbl, groups_train).to_string())
    print("\n#Groups per class (test):")
    print(per_class_group_counts(y_test_lbl, groups_test).to_string())

    # CV strategy (group-aware, auto-adjust splits)
    cv, actual_splits = safe_group_kfold(groups_train, CV_SPLITS)
    print(f"\n[INFO] GroupKFold splits: requested={CV_SPLITS} using={actual_splits}")

    base_models = make_base_models()
    results = []

    print("\nStarting Hyperparameter Search + CV + Holdout Evaluation...")

    for name, estimator in base_models.items():
        print(f"\n>> Model: {name}")

        pipe = make_pipeline(name, estimator)
        param_dist = make_param_distributions(name)

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            n_iter=N_ITER_SEARCH,
            scoring=SCORING,
            cv=cv,
            verbose=1,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            refit=True,
        )

        t0 = time.perf_counter()
        fit_with_optional_sample_weight(name, search, X_train, y_train, groups_train)
        train_time_s = time.perf_counter() - t0

        best_model = search.best_estimator_
        best_params = search.best_params_
        best_cv_score = float(search.best_score_)

        print(f"   Best CV {SCORING}: {best_cv_score:.4f}")
        print(f"   Best params: {best_params}")

        # Holdout inference timing
        t1 = time.perf_counter()
        preds = best_model.predict(X_test)
        t2 = time.perf_counter()

        per_sample_ms = ((t2 - t1) * 1000) / max(1, len(X_test))

        acc = accuracy_score(y_test, preds)
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            y_test, preds, average="weighted", zero_division=0
        )
        precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
            y_test, preds, average="macro", zero_division=0
        )

        print(f"   Holdout Accuracy: {acc:.2%}")
        print(f"   Holdout F1 weighted: {f1_w:.4f} | F1 macro: {f1_m:.4f}")
        print(f"   Inference per sample: {per_sample_ms:.4f} ms")

        # Save model
        model_filename = f"{dataset_tag}_classifier_{name.lower()}_best.pkl"
        joblib.dump(best_model, os.path.join(output_dir, model_filename))

        # Save label encoder-like info (ids -> names)
        with open(os.path.join(output_dir, "id_to_label.json"), "w") as f:
            json.dump({i: cn for i, cn in enumerate(class_names)}, f, indent=2)

        # Save report
        report_txt = classification_report(
            y_test, preds, target_names=class_names, zero_division=0
        )
        with open(os.path.join(output_dir, f"classification_report_{name.lower()}.txt"), "w") as f:
            f.write(f"DatasetTag: {dataset_tag}\n")
            f.write(f"Groups train={groups_train.nunique()} test={groups_test.nunique()} overlap={len(overlap)}\n")
            f.write(f"CV: GroupKFold n_splits={actual_splits} | Best CV ({SCORING})={best_cv_score:.6f}\n")
            f.write(f"Best params: {best_params}\n\n")
            f.write(report_txt)

        # Confusion matrix
        cm = confusion_matrix(y_test, preds, labels=np.arange(len(class_names)))
        save_confusion_matrix(
            cm,
            class_names,
            os.path.join(output_dir, f"confusion_matrix_{name.lower()}.png"),
            title=f"Confusion Matrix (Holdout) - {name} - {dataset_tag}",
        )

        results.append(
            {
                "Model": name,
                "Train_samples": len(X_train),
                "Test_samples": len(X_test),
                "Train_groups": int(groups_train.nunique()),
                "Test_groups": int(groups_test.nunique()),
                "Group_overlap": int(len(overlap)),
                "CV_splits": int(actual_splits),
                "CV_F1_weighted": best_cv_score,
                "Holdout_Accuracy": acc,
                "Holdout_F1_weighted": f1_w,
                "Holdout_F1_macro": f1_m,
                "Inference_Time_ms": per_sample_ms,
                "Training_Time_s": train_time_s,
                "Best_Params": str(best_params),
                "Model_File": model_filename,
            }
        )

    results_df = pd.DataFrame(results).sort_values(by="Holdout_F1_weighted", ascending=False)
    results_df.to_csv(os.path.join(output_dir, "benchmark_results_with_cv.csv"), index=False)

    print(f"\n--- FINAL LEADERBOARD ({dataset_tag}) ---")
    print(results_df.to_string(index=False))

    return results_df


# -------------------------
# Main
# -------------------------
def main():
    args = SimpleNamespace(**CONFIG)

    if args.mode not in ("tabular", "sequence"):
        raise ValueError(f"CONFIG['mode'] must be 'tabular' or 'sequence', got: {args.mode}")

    ensure_dir(args.out)

    if args.mode == "tabular":
        dataset_tag = args.tag or PathSafeTag(args.input)
        print(f"[TABULAR] Loading: {args.input}")
        df, X, y, groups, feature_names = load_tabular(args.input)

        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        class_names = list(le.classes_)
        joblib.dump(le, os.path.join(args.out, "label_encoder.pkl"))

        out_dir = os.path.join(args.out, dataset_tag)
        ensure_dir(out_dir)
        joblib.dump(feature_names, os.path.join(out_dir, "feature_names.pkl"))
        joblib.dump(le, os.path.join(out_dir, "label_encoder.pkl"))

        run_training(
            X=X,
            y_encoded=y_encoded,
            groups=groups,
            class_names=class_names,
            feature_names=feature_names,
            output_dir=out_dir,
            dataset_tag=dataset_tag,
        )

    else:
        dataset_tag = args.tag or "windows_flat"
        print(f"[SEQUENCE] Loading: {args.npz}")
        meta, X_np, y_ids, groups, feature_names, class_names = load_sequence(
            args.npz, args.meta, args.labels
        )

        # In sequence mode, y is already numeric ids aligned with labels.csv order.
        # Ensure it starts at 0..C-1; if not, remap.
        uniq = np.unique(y_ids)
        if not np.array_equal(uniq, np.arange(len(uniq))):
            remap = {old: i for i, old in enumerate(sorted(uniq))}
            y_mapped = np.array([remap[v] for v in y_ids], dtype=int)
        else:
            y_mapped = y_ids

        X = pd.DataFrame(X_np, columns=feature_names)

        out_dir = os.path.join(args.out, dataset_tag)
        ensure_dir(out_dir)

        run_training(
            X=X,
            y_encoded=y_mapped,
            groups=groups,
            class_names=class_names,
            feature_names=feature_names,
            output_dir=out_dir,
            dataset_tag=dataset_tag,
        )


def PathSafeTag(p: str) -> str:
    base = os.path.basename(p)
    tag = os.path.splitext(base)[0]
    tag = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in tag])
    return tag


if __name__ == "__main__":
    main()
