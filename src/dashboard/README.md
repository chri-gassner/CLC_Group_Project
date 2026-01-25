### 3. `dashboard/README.md` (Verfeinert)
*Habe `pandas` ergänzt (das wird im Code benutzt).*

# Dashboard – Streamlit Visualization

The presentation layer of the project. It connects directly to **Google Cloud Firestore** to visualize the benchmarking results in near real-time.

---

## 1. Installation

This can run in the same environment as the backend or a separate one.

```bash
pip install streamlit pandas google-cloud-firestore
```

## 2. Configuration
Ensure you have authenticated locally so the dashboard can access the database:

```bash
gcloud auth application-default login
gcloud config set project clc-group-vision-2026
```

## 3. Run
Run the Streamlit app:

```bash
streamlit run src/dashboard/app.py
```
Open your browser at http://localhost:8501.