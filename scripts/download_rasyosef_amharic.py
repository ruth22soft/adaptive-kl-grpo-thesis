#!/usr/bin/env python3
import os
import sys
import requests
from tqdm import tqdm

REPO = "rasyosef/amharic-sentences-corpus"
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'amharic_text_generation')
os.makedirs(OUT_DIR, exist_ok=True)

FILES = [
    # common shard names observed for this dataset
    'data/train-00000-of-00003.parquet',
    'data/train-00000-of-00001.parquet',
    'data/valid-00000-of-00001.parquet',
    'data/valid-00000-of-00003.parquet',
    'data/validation-00000-of-00001.parquet',
    'data/test-00000-of-00001.parquet',
]

HEADERS = {}
HF_TOKEN = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
if HF_TOKEN:
    HEADERS['Authorization'] = f'Bearer {HF_TOKEN}'

def download_file(repo, remote_path, local_path):
    url = f'https://huggingface.co/datasets/{repo}/resolve/main/{remote_path}'
    print(f"Downloading {url} -> {local_path}")
    with requests.get(url, headers=HEADERS, stream=True) as r:
        if r.status_code != 200:
            print(f"Failed to download {remote_path}: {r.status_code}")
            return False
        total = int(r.headers.get('content-length', 0))
        with open(local_path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=os.path.basename(local_path)) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return True

def main():
    # Find train and val files among candidates
    found_train = None
    found_val = None
    for fname in FILES:
        try:
            url = f'https://huggingface.co/datasets/{REPO}/resolve/main/{fname}'
            r = requests.head(url, headers=HEADERS)
            if r.status_code == 200:
                if 'train' in fname and not found_train:
                    found_train = fname
                if ('valid' in fname or 'validation' in fname or 'val' in fname) and not found_val:
                    found_val = fname
            else:
                # continue
                pass
        except Exception:
            pass

    if not found_train:
        print("Could not find a train shard in candidate list. Exiting.")
        print("If the dataset is gated, set HF_TOKEN environment variable and retry.")
        sys.exit(2)

    if not found_val:
        # fallback: use test shard as val
        for fname in FILES:
            if 'test' in fname:
                url = f'https://huggingface.co/datasets/{REPO}/resolve/main/{fname}'
                r = requests.head(url, headers=HEADERS)
                if r.status_code == 200:
                    found_val = fname
                    break

    if not found_val:
        print("Could not find a validation shard; continuing with train only (will also write small val by splitting later).")

    train_dst = os.path.join(OUT_DIR, 'train.parquet')
    val_dst = os.path.join(OUT_DIR, 'val.parquet')

    if found_train:
        ok = download_file(REPO, found_train, train_dst)
        if not ok:
            print("Train download failed. Exiting.")
            sys.exit(3)

    if found_val:
        ok = download_file(REPO, found_val, val_dst)
        if not ok:
            print("Val download failed. Exiting.")
            sys.exit(4)
    else:
        # create a tiny val split by taking first 5% of train if train exists
        import pandas as pd
        print("Creating a small val split from train (5%)")
        df = pd.read_parquet(train_dst)
        n = max(1, int(len(df) * 0.05))
        val_df = df.sample(n=n, random_state=42)
        val_df.to_parquet(val_dst, index=False)

    print("Download complete. Files written:")
    print(train_dst)
    print(val_dst)

if __name__ == '__main__':
    main()
