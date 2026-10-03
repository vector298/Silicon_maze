import sys, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
sys.path.insert(0, '/home/user/Silicon_maze/reed')
from features import TARGETS, graph_features
D = sys.argv[1]
tr = pd.read_csv(f'{D}/train_data.csv'); te = pd.read_csv(f'{D}/test_data.csv')
Ftr = graph_features(tr); Fte = graph_features(te)
Y = tr.groupby('graph_id')[TARGETS].first().loc[Ftr.graph_id].values
Ftr.to_pickle(f'{D}/Ftr.pkl'); Fte.to_pickle(f'{D}/Fte.pkl'); np.save(f'{D}/Y.npy', Y)
cats = pd.Categorical(pd.concat([Ftr.model, Fte.model])).categories
for F in (Ftr, Fte): F['model_c'] = pd.Categorical(F.model, categories=cats)
feats = [c for c in Ftr.columns if c not in ('graph_id', 'model', 'hash')]
X = Ftr[feats]
P = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=10,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1)
folds = list(GroupKFold(5).split(X, groups=Ftr.hash))
def run(X):
    oof = np.zeros_like(Y)
    for tri, vai in folds:
        for j in range(len(TARGETS)):
            oof[vai, j] = lgb.LGBMRegressor(**P).fit(X.iloc[tri], Y[tri, j]).predict(X.iloc[vai])
    nr = np.sqrt(((oof - Y) ** 2).mean(0)) / Y.std(0)
    print(' '.join(f'{v:.3f}' for v in nr), 'E %.3f score %.4f' % (nr.sum(), np.exp(-nr.sum() / 10)), flush=True)
    return oof
oof = run(X)
np.save(f'{D}/oof_lgb2.npy', oof)
