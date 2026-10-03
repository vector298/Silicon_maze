# GRC Restitution Docket: entity resolution

`pipeline.py` is one self-contained script: blocking, then pair features, then LightGBM, then average-linkage clustering, then `submission.csv`.
It trains everything from scratch (TF-IDF + gradient boosting). It uses no pretrained weights and makes no external calls.

## Run on Kaggle
1. Kaggle → **Datasets → New Dataset**. Upload the competition zip (64 MB is fine) and keep it private.
   If the competition is hosted on Kaggle itself, use **Add Input → Competition** instead.
2. In your notebook: **Add Input →** that dataset. Settings → Accelerator: none (CPU is fine).
3. New cell: paste the full contents of `pipeline.py`, then run it.
   The script finds `train/`, `validation/` and `test/` `observations.tsv` under `/kaggle/input` on its own.
   Run with `GRC_QUICK=1` (`os.environ["GRC_QUICK"]="1"` before the paste) to do a 15% smoke test first.
4. Outputs in `/kaggle/working`: `submission.csv` for upload and `val_predictions.csv` for error analysis.

## What to paste back for help
Only the training/validation log lines: column map, block counts, pair recall, the threshold grid and the per-regime scores.
**Never paste test rows anywhere** (competition rule).
