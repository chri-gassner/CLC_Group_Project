from pathlib import Path
import pandas as pd

ROOT = Path("/app/src/data")  # oder Path("../data") wenn du bewusst relativ willst
OUT = Path("manifest.csv")

rows = []
for class_dir in sorted([p for p in ROOT.iterdir() if p.is_dir()]):
    label = class_dir.name
    for vid in class_dir.rglob("*.mp4"):
        rows.append({"video_path": str(vid), "label": label})

df = pd.DataFrame(rows).sort_values(["label", "video_path"]).reset_index(drop=True)
print("Videos:", len(df), "Classes:", df["label"].nunique())

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print("Wrote", OUT)
