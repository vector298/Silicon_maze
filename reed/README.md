# Reed Richards' Last Defense

Predict 12 structural properties of each computational graph from its runtime fingerprint
(`candidate_runtime`, `c03`, `c14` per config).

- `features.py` — per-graph features: log-runtime distribution, knob (`c03`, `c14`) stats,
  runtime-vs-`c14` fits, config-id space size/gaps, model name (categorical), batch size.
- `cv.py <data_dir>` — 5-fold CV grouped by kernel hash (test kernels are almost all unseen).
- `train_predict.py <data_dir> [out]` — 3 LightGBM configs x 3 seeds per target; test graphs whose
  kernel hash appears in train get the exact train labels (identical hash => identical structure).

Grouped CV: E ≈ 6.86 → score ≈ 0.50 (k = 10).

Train column → submission column: `num_edges`→`operation_connectivity`,
`degree_skewness`→`connectivity_imbalance`, `degree_kurtosis`→`connectivity_extremity`,
`degree_95_percentile`→`high_end_threshold`; the others keep their names.
