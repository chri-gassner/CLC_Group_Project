import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import xgboost as xgb
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATASETS = {
    'OpenPose': 'src/openpose_pose_extraction/output/openpose_fitness_dataset.csv',
    'MediaPipe': 'src/mediapipe_pose_extraction/output/mediapipe_pose_dataset.csv'
}
BASE_OUTPUT_DIR = 'src/models_output/'

# Model Definitions
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42
    ),
    
    "XGBoost": xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=5, 
        use_label_encoder=False, eval_metric='mlogloss', random_state=42
    ),
    
    "SVM": SVC(
        kernel='rbf', C=1.0, probability=True, random_state=42
    ),
    
    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),
    
    "MLP_NeuralNet": MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), 
        activation='relu', 
        solver='adam', 
        alpha=0.0001, 
        batch_size='auto', 
        learning_rate='constant', 
        max_iter=1000,
        random_state=42
    )
}

# Models that require scaled data (StandardScaler)
NEED_SCALING = ["SVM", "KNN", "MLP_NeuralNet"]

# Main Processing Loop
for dataset_name, csv_path in DATASETS.items():
    print(f"\n{'='*60}")
    print(f"PROCESSING DATASET: {dataset_name}")
    print(f"{'='*60}")

    # Setup Output Directory
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
    class_names = list(le.classes_) # Convert to list for Seaborn/Matplotlib
    print(f"Classes found: {class_names}")

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Scaling (Important for SVM, KNN, MLP)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save Preprocessors
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    joblib.dump(le, os.path.join(output_dir, 'label_encoder.pkl'))
    print("Scaler & LabelEncoder saved.")

    results = []

    # Training Loop
    print("\nStarting Training & Evaluation Loop...")
    for name, model in models.items():
        print(f"\n>> Training {name}...")
        
        # Choose right data (Scaled vs. Raw)
        if name in NEED_SCALING:
            X_train_curr, X_test_curr = X_train_scaled, X_test_scaled
            print(f"   (Using Scaled Data for {name})")
        else:
            X_train_curr, X_test_curr = X_train, X_test 
            print(f"   (Using Raw Data for {name})")

        # Measure Training Time
        train_start = time.perf_counter()
        model.fit(X_train_curr, y_train)
        train_time = time.perf_counter() - train_start

        # Measure Inference Time (Simulating Edge Performance)
        # We measure the time for ALL test data and divide by the number of samples
        inf_start = time.perf_counter()
        preds_encoded = model.predict(X_test_curr)
        inf_end = time.perf_counter()
        
        acc = accuracy_score(y_test, preds_encoded)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds_encoded, average='weighted'
        )
        total_inf_time_ms = (inf_end - inf_start) * 1000
        per_sample_ms = total_inf_time_ms / len(X_test)
        
        print(f"   -> Accuracy: {acc:.2%}")
        print(f"   -> Inference per sample: {per_sample_ms:.4f} ms")
        
        # Save Model
        model_filename = f'fitness_classifier_{name.lower()}.pkl'
        joblib.dump(model, os.path.join(output_dir, model_filename))
        
        results.append({
            "Model": name,
            "N": n_test,
            "Accuracy": acc,
            "Precision_w": precision,
            "Recall_w": recall,
            "F1_w": f1,
            "Inference_Time_ms": per_sample_ms,
            "Training_Time_s": train_time
        })

    # Visualization & Reporting
    results_df = pd.DataFrame(results)
    
    print(f"\n--- FINAL LEADERBOARD ({dataset_name}) ---")
    print(results_df.sort_values(by="Accuracy", ascending=False).to_string(index=False))
    
    # Save CSV Report
    results_df.to_csv(os.path.join(output_dir, 'benchmark_results.csv'), index=False)

    # Create Plots
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Accuracy
    plt.subplot(1, 2, 1)
    sns.barplot(x='Model', y='Accuracy', data=results_df, palette='viridis', hue='Model', legend=False)
    plt.ylim(0.5, 1.05) # Focus on the upper half since accuracy is usually > 50%
    plt.title(f'Model Accuracy Comparison ({dataset_name})')
    plt.ylabel('Accuracy (0-1)')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Plot 2: Speed
    plt.subplot(1, 2, 2)
    sns.barplot(x='Model', y='Inference_Time_ms', data=results_df, palette='magma', hue='Model', legend=False)
    plt.title(f'Inference Speed Comparison ({dataset_name})')
    plt.ylabel('Time per Sample (ms) - Lower is Better')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'model_benchmark_comparison.png')
    plt.savefig(plot_path)
    print(f"\nBenchmark Plot saved to: {plot_path}")
    plt.close()

    # Feature Importance (Only for Random Forest, as it is the easiest to interpret)
    if "RandomForest" in models:
        rf_model = models["RandomForest"]
        importances = rf_model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(15)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='coolwarm', hue='Feature', legend=False)
        plt.title(f'Top 15 Features determining the Exercise (RF - {dataset_name})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance_rf.png'))
        plt.close()
        print("Feature Importance Plot saved.")

print("\nDone.")
