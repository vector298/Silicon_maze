"""Task 1: message trust classification.

`build(fit_df, pred_dfs, all_msgs, meta)` trains on `fit_df` (labelled) and
returns class probabilities for each frame in `pred_dfs`.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from features import LABELS, behaviour_features, history_features, text_oof

PARAMS = dict(
    objective="multiclass", num_class=3, learning_rate=0.03, num_leaves=31,
    min_child_samples=20, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=1, lambda_l2=1.0, verbose=-1,
)


def make_matrix(df, hist, all_msgs, meta, loo, text_p):
    X = pd.concat(
        [history_features(df, hist, loo), behaviour_features(df, all_msgs, meta, loo=True)],
        axis=1,
    )
    for i, lab in enumerate(LABELS):
        X[f"txt_{lab[:3]}"] = text_p[:, i]
    return X


def build(fit_df, pred_dfs, all_msgs, meta, n_rounds=600, seeds=(0, 1, 2)):
    oof_txt, other_txt = text_oof(fit_df.message, fit_df.trust_label, [d.message for d in pred_dfs])
    X = make_matrix(fit_df, fit_df, all_msgs, meta, True, oof_txt)
    y = pd.Categorical(fit_df.trust_label, categories=LABELS).codes
    Xp = [make_matrix(d, fit_df, all_msgs, meta, False, t) for d, t in zip(pred_dfs, other_txt)]
    for x in Xp:
        for c in X.columns:
            if str(X[c].dtype) == "category":
                x[c] = pd.Categorical(x[c], categories=X[c].cat.categories)
    probs = [np.zeros((len(d), 3)) for d in pred_dfs]
    for s in seeds:
        m = lgb.train({**PARAMS, "seed": s}, lgb.Dataset(X, y), n_rounds)
        for i, x in enumerate(Xp):
            probs[i] += m.predict(x[X.columns]) / len(seeds)
    return probs, m, X.columns


def decide(p, w=(1.0, 1.0, 1.0)):
    """Argmax with per-class weights (tuned for macro-F1)."""
    return np.array(LABELS)[np.argmax(p * np.array(w), axis=1)]
