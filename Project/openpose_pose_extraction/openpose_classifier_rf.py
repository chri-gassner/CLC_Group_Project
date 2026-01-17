import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Daten laden
df = pd.read_csv('Project/openpose_pose_extraction/output/openpose_fitness_dataset.csv')

# Features und Labels
X = df.drop(['label', 'file'], axis=1)
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Modell trainieren
print("Trainiere Modell...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Performance Benchmark
# Wir simulieren hier, wie schnell das Modell für EINE einzelne Vorhersage ist (wichtig für Edge)
# Um Messfehler zu glätten, lassen wir es über das ganze Testset laufen und teilen durch die Anzahl.
start_time = time.perf_counter() # Präziser als time.time()
_ = model.predict(X_test)
end_time = time.perf_counter()

total_time_ms = (end_time - start_time) * 1000
samples_count = len(X_test)
inference_time_per_sample = total_time_ms / samples_count

print(f"\n--- PERFORMANCE ---")
print(f"Total time for {samples_count} samples: {total_time_ms:.2f} ms")
print(f"Average Inference Time per Sample: {inference_time_per_sample:.4f} ms")
print(f"Theoretical Max FPS (nur ML-Teil): {1000/inference_time_per_sample:.2f} FPS")


# Feature Importance
# Welche Winkel/Koordinaten waren für die Entscheidung am wichtigsten?
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\n--- TOP 10 WICHTIGSTE FEATURES ---")
print(feature_importance_df.head(10))

# Optional: Plot speichern
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15))
plt.title('Top 15 Most Important Features for Exercise Classification')
plt.tight_layout()
plt.savefig('Project/openpose_pose_extraction/output/feature_importance.png')
print("Plot gespeichert als 'feature_importance.png'")


# Evaluieren (Metriken)
preds = model.predict(X_test)
print(f"\n--- QUALITÄT ---")
print(f"Model Accuracy: {accuracy_score(y_test, preds):.4f}")
# print(f"Confusion Matrix:\n{confusion_matrix(y_test, preds)}") # Erstmal auskommentiert für Übersicht
print(f"Classification Report:\n{classification_report(y_test, preds)}")

# Model Speichern
joblib.dump(model, 'Project/openpose_pose_extraction/output/fitness_classifier_openpose.pkl')
print("\nModell gespeichert.")