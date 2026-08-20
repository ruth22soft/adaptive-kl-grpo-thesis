#!/usr/bin/env python3
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

SRC_DIR='/home/ai-server-02/R_projects/final_thesis/data/simplelr_qwen_level3to5'
OUT_DIR=os.path.join(os.path.dirname(__file__),'..','data','amharic_text_generation')
os.makedirs(OUT_DIR, exist_ok=True)

paths=[]
for name in ['train.parquet','test.parquet']:
    p=os.path.join(SRC_DIR,name)
    if os.path.exists(p):
        paths.append(p)

if not paths:
    print('No source parquets found in',SRC_DIR)
    raise SystemExit(1)

dfs=[pd.read_parquet(p) for p in paths]
df=pd.concat(dfs, ignore_index=True)
print('Loaded rows:',len(df))

# helper to extract prompt text
import numpy as np

def extract_prompt(row):
    # prefer extra_info.question
    ei=row.get('extra_info')
    if isinstance(ei, dict) and 'question' in ei and ei['question']:
        return ei['question']
    # prompt field may be array of dicts
    pr=row.get('prompt')
    if isinstance(pr, (list,tuple)):
        # find dict with role 'user'
        for d in pr:
            try:
                if d.get('role')=='user' and d.get('content'):
                    return d.get('content')
            except Exception:
                continue
        # fallback to first content
        for d in pr:
            if isinstance(d, dict) and d.get('content'):
                return d.get('content')
        return None
    # sometimes prompt is numpy array
    if isinstance(pr, (np.ndarray,)):
        for d in pr.tolist():
            if isinstance(d, dict) and d.get('role')=='user' and d.get('content'):
                return d.get('content')
        for d in pr.tolist():
            if isinstance(d, dict) and d.get('content'):
                return d.get('content')
    # otherwise, if string
    if isinstance(pr,str):
        return pr
    return None


def extract_ground_truth(row):
    rm=row.get('reward_model')
    if isinstance(rm, dict) and 'ground_truth' in rm and rm['ground_truth']:
        return rm['ground_truth']
    ei=row.get('extra_info')
    if isinstance(ei, dict) and 'answer' in ei and ei['answer']:
        return ei['answer']
    sol=row.get('solution')
    if isinstance(sol,str) and len(sol.strip())>0:
        return sol
    return None

records=[]
for _,r in df.iterrows():
    prompt=extract_prompt(r)
    gt=extract_ground_truth(r)
    if not prompt or not gt:
        continue
    records.append({'prompt':prompt,'ground_truth':gt,'data_source': r.get('data_source','simplelr_qwen')})

print('Extracted records:',len(records))
if len(records)==0:
    raise SystemExit('No records extracted')

train, val = train_test_split(records, test_size=0.05, random_state=42)

train_df=pd.DataFrame(train)
val_df=pd.DataFrame(val)

train_path=os.path.join(OUT_DIR,'train.parquet')
val_path=os.path.join(OUT_DIR,'val.parquet')
train_df.to_parquet(train_path, index=False)
val_df.to_parquet(val_path, index=False)
print('Wrote:',train_path,val_path)
