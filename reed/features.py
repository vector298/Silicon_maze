import re
import warnings
import numpy as np
import pandas as pd

TARGETS = ['num_edges', 'degree_std', 'max_degree', 'num_sink_nodes', 'num_unique_opcodes',
           'opcode_entropy', 'degree_skewness', 'degree_kurtosis', 'degree_95_percentile',
           'in_degree_std', 'out_degree_std', 'num_degree_1_nodes']
# submission column names, in the same order as TARGETS
SUB_COLS = ['operation_connectivity', 'degree_std', 'max_degree', 'num_sink_nodes', 'num_unique_opcodes',
            'opcode_entropy', 'connectivity_imbalance', 'connectivity_extremity', 'high_end_threshold',
            'in_degree_std', 'out_degree_std', 'num_degree_1_nodes']


def split_id(gid):
    model, h = gid.rsplit('_', 1)
    return model, h


def _stats(x, p):
    q = np.quantile(x, [0, .05, .1, .25, .5, .75, .9, .95, 1])
    d = {f'{p}_q{i}': v for i, v in enumerate(q)}
    d[f'{p}_mean'] = x.mean()
    d[f'{p}_std'] = x.std()
    d[f'{p}_nuniq'] = len(np.unique(x))
    return d


def graph_features(df):
    warnings.simplefilter('ignore', np.exceptions.RankWarning)
    rows = []
    for gid, g in df.groupby('graph_id', sort=False):
        g = g.sort_values('config_id')
        rt = g.candidate_runtime.values.astype(float)
        lrt = np.log1p(rt)
        c03 = g.c03.values.astype(float)
        c14 = g.c14.values.astype(float)
        model, h = split_id(gid)
        d = {'graph_id': gid, 'model': model, 'hash': h}
        m = re.search(r'batch_(\d+)', model)
        d['batch'] = float(m.group(1)) if m else np.nan
        d['n_cfg'] = len(g)
        d['max_cfg_id'] = g.config_id.max()
        d['min_cfg_id'] = g.config_id.min()
        d.update(_stats(lrt, 'lrt'))
        d['lrt_range'] = lrt.max() - lrt.min()
        rel = lrt - lrt.min()
        d.update(_stats(rel, 'rel'))
        d.update(_stats(c03, 'c03'))
        d.update(_stats(c14, 'c14'))
        lc14 = np.log1p(c14)
        d['corr_lrt_c14'] = np.corrcoef(lrt, lc14)[0, 1] if lc14.std() > 0 and lrt.std() > 0 else 0
        d['corr_lrt_c03'] = np.corrcoef(lrt, c03)[0, 1] if c03.std() > 0 and lrt.std() > 0 else 0
        if lc14.std() > 0:
            b = np.polyfit(lc14, lrt, 1)
            d['slope_lrt_c14'], d['icpt_lrt_c14'] = b
        # runtime of the best config and its knobs
        i = rt.argmin()
        d['best_c14'] = c14[i]; d['best_c03'] = c03[i]
        j = rt.argmax()
        d['worst_c14'] = c14[j]; d['worst_c03'] = c03[j]
        d['frac_c03_0'] = (c03 == 0).mean()
        d['c14_max_div_min'] = c14.max() / max(c14.min(), 1)
        # first few configs (config ids tend to be generated in a fixed order)
        for k in range(5):
            d[f'c14_{k}'] = c14[k] if k < len(c14) else np.nan
            d[f'c03_{k}'] = c03[k] if k < len(c03) else np.nan
            d[f'rel_{k}'] = rel[k] if k < len(rel) else np.nan
        d['lrt_x_c14'] = np.mean(lrt + lc14)
        d['n_dup_rt'] = len(rt) - len(np.unique(rt))
        w = lrt + lc14  # runtime * tile-ish knob ~ total work
        d.update(_stats(w, 'work'))
        if lc14.std() > 0:
            d['fit_resid_std'] = np.std(lrt - np.polyval(np.polyfit(lc14, lrt, 1), lc14))
            b2 = np.polyfit(lc14, lrt, 2)
            d['q2a'], d['q2b'], d['q2c'] = b2
        for v in (0, 1, 2, 4, 8):
            mk = c03 == v
            d[f'n_c03_{v}'] = mk.mean()
            d[f'lrt_c03_{v}'] = lrt[mk].min() if mk.any() else np.nan
            d[f'work_c03_{v}'] = w[mk].mean() if mk.any() else np.nan
        cid = g.config_id.values
        gaps = np.diff(cid)
        d['gap_mean'] = gaps.mean() if len(gaps) else np.nan
        d['gap_max'] = gaps.max() if len(gaps) else np.nan
        d['cid_ratio'] = len(cid) / (cid.max() + 1)
        d['n_c14_le4'] = (c14 <= 4).mean()
        d['c14_lmean'] = lc14.mean()
        uu = np.unique(c14)
        d['c14_gcd'] = np.gcd.reduce(uu.astype(int)) if len(uu) else 0
        rows.append(d)
    return pd.DataFrame(rows)
