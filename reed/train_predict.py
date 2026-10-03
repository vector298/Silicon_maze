"""Train LightGBM ensemble on per-graph fingerprint features and write submission.csv.

Usage: python train_predict.py <data_dir> [out_csv] [variant]
variant: full   - all features (best grouped-by-hash CV, weaker under train/test shift)
         robust - drops model name, batch, config-id and config-order features
         blend  - 50/50 average of full and robust (default)
"""
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from features import TARGETS, SUB_COLS, graph_features

D = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else 'submission.csv'
variant = sys.argv[3] if len(sys.argv) > 3 else 'blend'
tr = pd.read_csv(f'{D}/train_data.csv'); te = pd.read_csv(f'{D}/test_data.csv')
Ftr = graph_features(tr); Fte = graph_features(te)
lab = tr.groupby('graph_id')[TARGETS].first()
Y = lab.loc[Ftr.graph_id].values
cats = pd.Categorical(pd.concat([Ftr.model, Fte.model])).categories
for F in (Ftr, Fte): F['model_c'] = pd.Categorical(F.model, categories=cats)
feats = [c for c in Ftr.columns if c not in ('graph_id', 'model', 'hash')]
CFG_ID = {'min_cfg_id', 'max_cfg_id', 'gap_mean', 'gap_max', 'cid_ratio', 'batch', 'n_cfg', 'model_c'}
is_order = lambda c: c[:4] in ('rel_', 'c14_', 'c03_') and c.split('_')[1].isdigit()
robust = [c for c in feats if c not in CFG_ID and not is_order(c)]
sets = {'full': [feats], 'robust': [robust], 'blend': [feats, robust]}[variant]

base = dict(learning_rate=0.03, subsample=0.8, subsample_freq=1, verbose=-1, n_jobs=4)
configs = [
    dict(reg_lambda=5, min_child_samples=30, num_leaves=63, n_estimators=800, colsample_bytree=0.5),
    dict(reg_lambda=5, min_child_samples=20, num_leaves=127, n_estimators=1000, learning_rate=0.02, colsample_bytree=0.4),
    dict(reg_lambda=10, min_child_samples=40, num_leaves=63, n_estimators=1200, colsample_bytree=0.3),
]
pred = np.zeros((len(Fte), len(TARGETS)))
for fs in sets:
    p = np.zeros_like(pred)
    n = 0
    for cfg in configs:
        for seed in range(2):
            for j in range(len(TARGETS)):
                m = lgb.LGBMRegressor(**{**base, **cfg, 'random_state': seed}).fit(Ftr[fs], Y[:, j])
                p[:, j] += m.predict(Fte[fs])
            n += 1
    pred += p / n / len(sets)

# Graphs with the same kernel hash have identical labels: average predictions within each hash.
pred = pd.DataFrame(pred).groupby(Fte.hash.values).transform('mean').to_numpy(copy=True)

# Graphs whose kernel hash was seen in training have identical structure: copy labels exactly.
hash_lab = lab.assign(hash=[i.rsplit('_', 1)[1] for i in lab.index]).groupby('hash')[TARGETS].first()
hit = Fte.hash.isin(hash_lab.index).values
pred[hit] = hash_lab.loc[Fte.hash[hit]].values
print(f'exact hash matches: {hit.sum()} / {len(Fte)}')

# Clip to the valid training range; integer targets keep fractional values (better RMSE).
pred = np.clip(pred, Y.min(0), Y.max(0))
sub = pd.DataFrame(pred, columns=SUB_COLS)
sub.insert(0, 'graph_id', Fte.graph_id.values)
sub.to_csv(out, index=False)
print('wrote', out, sub.shape)
