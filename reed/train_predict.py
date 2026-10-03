"""Train LightGBM ensemble on per-graph fingerprint features and write submission.csv.

Usage: python train_predict.py <data_dir> [out_csv]
"""
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from features import TARGETS, SUB_COLS, graph_features

D = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else 'submission.csv'
tr = pd.read_csv(f'{D}/train_data.csv'); te = pd.read_csv(f'{D}/test_data.csv')
Ftr = graph_features(tr); Fte = graph_features(te)
lab = tr.groupby('graph_id')[TARGETS].first()
Y = lab.loc[Ftr.graph_id].values
cats = pd.Categorical(pd.concat([Ftr.model, Fte.model])).categories
for F in (Ftr, Fte): F['model_c'] = pd.Categorical(F.model, categories=cats)
feats = [c for c in Ftr.columns if c not in ('graph_id', 'model', 'hash')]

base = dict(learning_rate=0.03, subsample=0.8, subsample_freq=1, verbose=-1, n_jobs=4)
configs = [
    dict(reg_lambda=5, min_child_samples=30, num_leaves=63, n_estimators=800, colsample_bytree=0.5),
    dict(reg_lambda=5, min_child_samples=20, num_leaves=127, n_estimators=1000, learning_rate=0.02, colsample_bytree=0.4),
    dict(reg_lambda=10, min_child_samples=40, num_leaves=63, n_estimators=1200, colsample_bytree=0.3),
]
pred = np.zeros((len(Fte), len(TARGETS)))
n = 0
for cfg in configs:
    for seed in range(3):
        for j in range(len(TARGETS)):
            m = lgb.LGBMRegressor(**{**base, **cfg, 'random_state': seed}).fit(Ftr[feats], Y[:, j])
            pred[:, j] += m.predict(Fte[feats])
        n += 1
pred /= n

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
