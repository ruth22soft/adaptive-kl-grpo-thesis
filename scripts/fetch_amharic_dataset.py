#!/usr/bin/env python3
import os
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'amharic_text_generation')
os.makedirs(OUT_DIR, exist_ok=True)

candidates = [
    ("oscar", "unshuffled_deduplicated_am"),
    ("mc4", "am"),
    ("wiki40b", "am"),
    ("wikipedia", "20220301.am"),
]

ds = None
used = None
for repo, cfg in candidates:
    try:
        print(f"Trying {repo}:{cfg}...")
        ds = load_dataset(repo, cfg, split="train")
        used = f"{repo}:{cfg}"
        print(f"Loaded {used}, examples={len(ds)}")
        break
    except Exception as e:
        print(f"Failed {repo}:{cfg} -> {e}")

if ds is None:
    raise SystemExit("No candidate Amharic dataset could be loaded. Abort.")

# Convert to plain text column name heuristics
text_col = None
for col in ["text", "content", "body", "sentence", "article"]:
    if col in ds.column_names:
        text_col = col
        break
if text_col is None:
    # fallback to first column
    text_col = ds.column_names[0]

print(f"Using text column: {text_col}")

# Limit number of examples to avoid huge downloads (user can re-run to get more)
MAX_EXAMPLES = min(100000, len(ds))
print(f"Taking up to {MAX_EXAMPLES} examples")
if MAX_EXAMPLES < len(ds):
    ds = ds.select(range(MAX_EXAMPLES))

texts = [t for t in ds[text_col] if isinstance(t, str) and len(t.strip())>20]
print(f"Filtered to {len(texts)} usable texts")

# Build prompt/ground_truth pairs: simple instruction + target text
rows = []
for t in texts:
    prompt = "Write a short Amharic passage or continue the following text:" 
    rows.append({
        "prompt": prompt,
        "ground_truth": t,
        "data_source": used,
    })

# Split train/val
train_rows, val_rows = train_test_split(rows, test_size=0.05, random_state=42)

train_df = pd.DataFrame(train_rows)
val_df = pd.DataFrame(val_rows)

train_path = os.path.join(OUT_DIR, "train.parquet")
val_path = os.path.join(OUT_DIR, "val.parquet")

print(f"Writing train ({len(train_df)}) -> {train_path}")
train_df.to_parquet(train_path, index=False)
print(f"Writing val ({len(val_df)}) -> {val_path}")
val_df.to_parquet(val_path, index=False)

print("Done. Wrote parquets to:")
print(train_path)
print(val_path)
