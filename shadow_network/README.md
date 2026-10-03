# Avengers: Shadow Network — solution

```
pip install pandas scikit-learn lightgbm networkx scipy
cd shadow_network
python run.py --data ../data/upload_structure --out submission          # main
python run.py --variant m_only   --out submission_m_only                 # task-2 alternative
python run.py --variant ato_only --out submission_ato_only               # task-2 alternative
python validate1.py                                                      # task-1 train->validation check
```

Each output dir has `task1_predictions.csv`, `task2_predictions.csv`, `task3_predictions.csv`
and `submission.csv` (the combined layout of `sample_submission.csv`).
Variants differ only in task 2.

## What the data turned out to be

* **Messages are templated**: `<opener> <topic> <stock ending> <filler words>`.
  There are 40 topics and ~45 openers. 13 openers (plus "Update:" without
  "Monitor closely.") are *covert* templates that never appear on TRUSTWORTHY
  messages.
* **Accounts have a latent type**, shared by all accounts of the same person:
  | type | # | labels |
  |---|---|---|
  | clean (N) | 2023 | benign template → TRUSTWORTHY, covert template → SUSPICIOUS (100% pure) |
  | S | 193 | everything SUSPICIOUS (covert mostly MALICIOUS) |
  | MIX | 208 | "social/operational" openers ("Appreciated.", "Executing as planned.", …) → SUSPICIOUS, the rest MALICIOUS |
  | M | 123 | everything MALICIOUS |
* **Takeovers**: only two clean accounts flip to M-type behaviour in the labelled
  period (A001564 from 2025-05-29, A000797 from 2025-08-09).
* **Campaign cells**: M-type accounts message each other ~5x more than chance;
  their communication graph has modularity 0.81 (0.49 for a degree-matched
  random graph). MIX/S accounts show no device/location/recipient/timing
  structure beyond what clean accounts show.
* `sample_submission.csv` is a weak baseline (~73% agreement with this model on
  task 1, compromise timestamp = account's first test message), not ground truth.

## Approach

* **Task 1** (`task1.py`, `features.py`): LightGBM on leave-one-out sender/recipient
  label history, per-template-group sender history (covert / social / directive),
  template purity maps, recent-history window, behavioural context
  (hours vs. normal window, device ownership/novelty, location), and OOF TF-IDF
  logistic-regression probabilities. Validation macro-F1 (train → validation
  split): **0.989** argmax / 0.9975 with tuned class weights; ~99.3% agreement
  with the pure rule-based reading on test. Messages from takeover accounts after
  takeover are set to MALICIOUS.
* **Task 2**: compromised = M + MIX hostile accounts (the ones sending MALICIOUS
  traffic) + detected takeovers. Timestamp = first non-trustworthy message
  (first MALICIOUS for takeovers). The ground-truth definition of "compromised"
  is not recoverable from the data, hence the `m_only` and `ato_only` variants.
* **Task 3**: Louvain communities on the M-type communication graph (12 cells,
  125 accounts); MIX, S and clean accounts get one cluster each.
