# Amharic Training Run Report

## Summary
- Goal: run Amharic training using `rasyosef/Llama-3.2-400M-Amharic-Instruct-Poems-Stories-Wikipedia` in the `adaptive-kl-grpo-thesis` repo, with dataset mounted at `/data/amharic_text_generation` and checkpoints at `/ckpts/amharic_text_generation_ctx512`.

## Actions performed
- Downloaded dataset shards from `rasyosef/amharic-sentences-corpus` and wrote `train.parquet` and `val.parquet` under the repo `data/amharic_text_generation` (already present).
- Added downloader utility (if present) at `scripts/download_rasyosef_amharic.py`.
- Launched the container using `run_amharic_text_generation.sh` and directly via `docker run` with Hydra overrides.
- Iteratively resolved missing Python runtime dependencies inside the container so trainer could start (installed: `pandas`, `tensordict`, `ray[rllib]`, `codetiming`, `omegaconf`, `transformers`, `accelerate`, `word2number`, and others). These installs were performed inside the running container to unblock the run.

## Issues found
- Missing Python packages in the base Docker image caused immediate crashes on import (first `pandas`, then `tensordict`, `ray`, `codetiming`, `omegaconf`, `transformers`, `word2number`, etc.).
- `run_amharic_text_generation.sh` applies runtime overrides: `trainer.total_epochs=1` and `trainer.save_freq=50` which caused the job to finish very quickly when the dataset was small.
- Dataset size: after filtering the dataset reported `Size of train dataloader: 1`, so a single training batch ran (step 1) and the process exited normally before any checkpoint (save_freq was 50 in the launched run).
- Installing packages inside the container works but is ephemeral; re-running the container will require reinstallation or building a custom image / adding a `requirements.txt` / Dockerfile step.

## What completed successfully
- Dataset files placed at `data/amharic_text_generation/train.parquet` and `val.parquet`.
- Trainer launched to the point where it executed step 1 and logged metrics to `/ckpts/amharic_text_generation_ctx512/train.log`.

## Remaining / next steps
1. Persist dependency installation by adding a `requirements.txt` or building a custom Docker image that includes the Python packages used by the repo. This avoids repeated container pip installs.
2. Adjust the runtime overrides so training actually runs the intended number of steps/epochs. Example choices:
   - Use `trainer.total_epochs=10` and `trainer.total_training_steps=1566` (your requested settings), and ensure `trainer.save_freq=200` and `trainer.remove_previous_ckpt=True` are set.
   - For quick verification, set `trainer.save_freq=1` temporarily to produce a checkpoint immediately.
3. Investigate dataset size/format and verify `prompt`/`ground_truth` columns are present and that data sharding provides enough samples for training.

## Proposed architecture / deployment recommendations
- Keep Hydra-driven configs (`verl.trainer.config/...`) for reproducible experiments.
- Build a small `Dockerfile` (derived from `docker.io/haibinlin/verl:v0.0.5-th2.4.0-cu124-base`) that: installs the repo-level `requirements.txt`, copies the codebase, and exposes a small entrypoint script for launching with Hydra overrides. This ensures consistent runtime environments.
- Persist checkpoints to a named Docker volume (currently `simplerl_ckpts`) or host path; include a `latest_checkpointed_iteration.txt` tracker for resume logic.

## Quick commands to commit and push these changes
Run these from the repo root (`adaptive-kl-grpo-thesis` parent):

```bash
git add adaptive-kl-grpo-thesis/RUN_REPORT_AMHARIC.md
git add adaptive-kl-grpo-thesis/scripts/download_rasyosef_amharic.py  # if you added it
git commit -m "docs: add Amharic training run report and dataset downloader"
git push origin HEAD
```

If you prefer a single commit for all changes in the repo (careful—this stages everything):

```bash
git add -A
git commit -m "chore: record Amharic training run, add downloader and run notes"
git push origin HEAD
```

---
If you want, I can: build a `requirements.txt`, create a Dockerfile with the pinned dependencies, or open a PR with these changes. Tell me which you prefer next.
