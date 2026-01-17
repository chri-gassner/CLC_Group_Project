import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Config
DATASETS = {
    'OpenPose': 'Project/openpose_pose_extraction/output/openpose_fitness_dataset.csv',
    'MediaPipe': 'Project/mediapipe_pose_extraction/output/mediapipe_pose_dataset.csv'
}
BASE_OUTPUT_DIR = 'Project/models_output/'

# Model Definitions
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1),
    "SVM": SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    "XGBoost": xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1)
}

# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
for dataset_name, csv_path in DATASETS.items():
    print(f"\n{'='*40}")
    print(f"PROCESSING DATASET: {dataset_name}")
    print(f"{'='*40}")

    # Setup specific output dir
    output_dir = os.path.join(BASE_OUTPUT_DIR, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load Data
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}. Skipping.")
        continue

    # Features and Labels
    X = df.drop(['label', 'file'], axis=1)
    y = df['label']

    # Encode Labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_
    print(f"Classes: {class_names}")

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save Preprocessors
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    joblib.dump(le, os.path.join(output_dir, 'label_encoder.pkl'))

    results = []

    # Training Loop
    print("\nStarting Training & Evaluation Loop...")
    for name, model in models.items():
        print(f"\n--- Training {name} ({dataset_name}) ---")
        
        if name == "SVM":
            X_train_curr, X_test_curr = X_train_scaled, X_test_scaled
        else:
            X_train_curr, X_test_curr = X_train, X_test 

        # Train
        train_start = time.perf_counter()
        model.fit(X_train_curr, y_train)
        train_time = time.perf_counter() - train_start

        # Benchmark
        inf_start = time.perf_counter()
        preds_encoded = model.predict(X_test_curr)
        inf_end = time.perf_counter()
        
        acc = accuracy_score(y_test, preds_encoded)
        total_inf_time_ms = (inf_end - inf_start) * 1000
        per_sample_ms = total_inf_time_ms / len(X_test)
        
        print(f"  -> Accuracy: {acc:.4f}")
        print(f"  -> Inference per sample: {per_sample_ms:.4f} ms")
        
        # Save Model
        joblib.dump(model, os.path.join(output_dir, f'fitness_classifier_{name.lower()}.pkl'))
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Inference_Time_ms": per_sample_ms,
            "Training_Time_s": train_time
        })
        
        print(classification_report(y_test, preds_encoded, target_names=class_names))

    # Comparison & Visualization
    results_df = pd.DataFrame(results)
    print(f"\n--- FINAL LEADERBOARD ({dataset_name}) ---")
    print(results_df.sort_values(by="Accuracy", ascending=False).to_string(index=False))

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.barplot(x='Model', y='Accuracy', data=results_df, palette='viridis', hue='Accuracy')
    plt.ylim(0, 1.1)
    plt.title(f'Model Accuracy ({dataset_name})')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    sns.barplot(x='Model', y='Inference_Time_ms', data=results_df, palette='magma', hue='Inference_Time_ms')
    plt.title(f'Inference Speed ({dataset_name})')
    plt.ylabel('Time per Sample (ms)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_benchmark_comparison.png'))
    print(f"Plot saved to {output_dir}")
    plt.close() # Close to prevent overlap

    # Feature Importance (RF only)
    rf_model = models["RandomForest"]
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='coolwarm', hue='Importance')
    plt.title(f'Top 15 Features (RF - {dataset_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance_rf.png'))
    plt.close()