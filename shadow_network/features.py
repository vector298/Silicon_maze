"""Feature construction for the Avengers: Shadow Network tasks."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from topics import template_parts

LABELS = ["TRUSTWORTHY", "SUSPICIOUS", "MALICIOUS"]


def load(data_dir):
    r = lambda p: pd.read_csv(f"{data_dir}/{p}")
    tr = r("train/messages.csv").merge(r("train/message_labels.csv"))
    va = r("validation/messages.csv").merge(r("validation/message_labels.csv"))
    te = r("test/messages.csv")
    va.index = va.index + len(tr)
    te.index = te.index + len(tr) + len(va)
    meta = dict(
        accounts=r("test/accounts.csv"),
        devices=r("test/devices.csv"),
        locations=r("test/locations.csv"),
        orgs=r("test/organizations.csv"),
    )
    for d in (tr, va, te):
        d["ts"] = pd.to_datetime(d.timestamp)
    add_template_keys([tr, va, te])
    return tr, va, te, meta


def add_template_keys(frames):
    """Template key = opener with topic slot + the stock phrase after the topic
    (rare trailing phrases collapsed to '*')."""
    parts = [d.message.map(template_parts) for d in frames]
    post = pd.concat([p.str[1] for p in parts])
    # Stock template endings ("Monitor closely.") are kept; comma-terminated
    # filler transitions ("Therefore,") are random and collapse to '*'.
    common = set(post.value_counts().loc[lambda c: c >= 100].index)
    common = {c for c in common if not c.endswith(",")}
    for d, p in zip(frames, parts):
        d["tmpl_pre"] = p.str[0]
        d["tmpl_key"] = p.str[0] + "|" + p.str[1].where(p.str[1].isin(common), "*")


def template_maps(hist):
    """Template-level label behaviour learnt from labelled history:
    - key_trust: P(TRUSTWORTHY | key) on accounts that are mostly clean
      (low => covert template; clean accounts only send these as SUSPICIOUS),
    - key_mixmal: P(MALICIOUS | key) on hostile senders for non-covert keys
      (separates 'social/operational' openers from directive-style ones)."""
    bad = hist.trust_label != "TRUSTWORTHY"
    sb = bad.groupby(hist.sender_id).transform("mean")
    clean = hist[sb < 0.5]
    key_trust = (clean.trust_label == "TRUSTWORTHY").groupby(clean.tmpl_key).mean()
    host = hist[(sb >= 0.6) & (hist.tmpl_key.map(key_trust).fillna(1) >= 0.5)]
    # Only accounts that use both labels carry information about the key.
    am = (host.trust_label == "MALICIOUS").groupby(host.sender_id).transform("mean")
    mix = host[(am > 0.2) & (am < 0.8)]
    key_mixmal = (mix.trust_label == "MALICIOUS").groupby(mix.tmpl_key).agg(["sum", "size"])
    key_mixmal = (key_mixmal["sum"] + 0.5) / (key_mixmal["size"] + 1)
    return key_trust, key_mixmal


def template_group(df, maps):
    key_trust, key_mixmal = maps
    # Unseen keys fall back to the opener's '*' variant.
    fb = df.tmpl_pre + "|*"
    kt = df.tmpl_key.map(key_trust).fillna(fb.map(key_trust)).fillna(1.0)
    km = df.tmpl_key.map(key_mixmal).fillna(fb.map(key_mixmal)).fillna(0.5)
    grp = np.where(kt < 0.5, "cov", np.where(km < 0.5, "soc", "dir"))
    return pd.Series(grp, index=df.index), kt, km


def _group_stats(hist, key, prefix):
    """Per-key label counts from labelled history."""
    h = hist.assign(
        _bad=(hist.trust_label != "TRUSTWORTHY").astype(float),
        _mal=(hist.trust_label == "MALICIOUS").astype(float),
        _sus=(hist.trust_label == "SUSPICIOUS").astype(float),
        _n=1.0,
    )
    return h.groupby(key)[["_n", "_bad", "_mal", "_sus"]].sum().add_prefix(prefix)


def history_features(df, hist, loo):
    """Sender/recipient label history. If loo, `df` rows are part of `hist`
    and their own label is subtracted (leave-one-out)."""
    out = pd.DataFrame(index=df.index)
    prior = 2.0
    base = {
        "bad": (hist.trust_label != "TRUSTWORTHY").mean(),
        "mal": (hist.trust_label == "MALICIOUS").mean(),
        "sus": (hist.trust_label == "SUSPICIOUS").mean(),
    }
    for key, p in [("sender_id", "s"), ("recipient_id", "r")]:
        st = _group_stats(hist, key, p)
        j = df[[key]].join(st, on=key).fillna(0.0)
        n = j[f"{p}_n"].values.copy()
        cnt = {k: j[f"{p}_{k}"].values.copy() for k in ("bad", "mal", "sus")}
        if loo:
            n -= 1
            cnt["bad"] -= (df.trust_label != "TRUSTWORTHY").values
            cnt["mal"] -= (df.trust_label == "MALICIOUS").values
            cnt["sus"] -= (df.trust_label == "SUSPICIOUS").values
        out[f"{p}_hist_n"] = n
        for k in cnt:
            out[f"{p}_hist_{k}"] = (cnt[k] + prior * base[k]) / (n + prior)
        out[f"{p}_hist_malshare"] = (cnt["mal"] + 0.5) / (cnt["bad"] + 1.0)
    # Same stats for the sender acting as a recipient (and vice versa).
    st = _group_stats(hist, "recipient_id", "x")
    j = df[["sender_id"]].join(st, on="sender_id").fillna(0.0)
    out["s_as_recv_bad"] = (j.x_bad + prior * base["bad"]) / (j.x_n + prior)
    out = out.join(recent_features(df, hist))
    out = out.join(template_history_features(df, hist, loo))
    return out


def template_history_features(df, hist, loo):
    """Sender label rates conditioned on template group (LOO when loo)."""
    maps = template_maps(hist)
    gd, kt, km = template_group(df, maps)
    gh, _, _ = template_group(hist, maps)
    out = pd.DataFrame(index=df.index)
    out["key_trust"], out["key_mixmal"] = kt.values, km.values
    out["tmpl_group"] = pd.Categorical(gd, categories=["cov", "soc", "dir"])
    h = hist.assign(_g=gh.values, _bad=(hist.trust_label != "TRUSTWORTHY").astype(float),
                    _mal=(hist.trust_label == "MALICIOUS").astype(float), _n=1.0)
    st = h.groupby(["sender_id", "_g"])[["_n", "_bad", "_mal"]].sum().unstack("_g").fillna(0)
    own_bad = (df.trust_label != "TRUSTWORTHY").values if loo else 0
    own_mal = (df.trust_label == "MALICIOUS").values if loo else 0
    for g in ["cov", "soc", "dir"]:
        cols = [(c, g) for c in ["_n", "_bad", "_mal"]]
        j = pd.DataFrame(0.0, index=df.index, columns=["n", "bad", "mal"])
        if all(c in st.columns for c in cols):
            j[:] = st.reindex(df.sender_id)[cols].fillna(0).values
        if loo:
            same = (gd == g).values
            j["n"] -= same
            j["bad"] -= same * own_bad
            j["mal"] -= same * own_mal
        out[f"s_{g}_n"] = j.n.values
        out[f"s_{g}_bad"] = ((j.bad + 0.2) / (j.n + 1)).values
        out[f"s_{g}_mal"] = ((j.mal + 0.2) / (j.n + 1)).values
        out[f"s_{g}_malshare"] = ((j.mal + 0.5) / (j.bad + 1)).values
    return out


def recent_features(df, hist, k=3):
    """Label mix of the sender's last k labelled messages strictly before each
    row. Captures accounts whose behaviour changed (compromise switchers)."""
    h = hist[["sender_id", "ts", "trust_label"]].sort_values("ts")
    h = h.assign(_bad=(h.trust_label != "TRUSTWORTHY").astype(float),
                 _mal=(h.trust_label == "MALICIOUS").astype(float))
    out = pd.DataFrame(index=df.index, columns=["s_recent_bad", "s_recent_mal", "s_recent_gap_days"], dtype=float)
    groups = {s: g for s, g in h.groupby("sender_id")}
    for s, rows in df.groupby("sender_id"):
        g = groups.get(s)
        if g is None:
            continue
        t = g.ts.values
        pos = np.searchsorted(t, rows.ts.values, side="left")
        for idx, p, ts in zip(rows.index, pos, rows.ts.values):
            if p == 0:
                continue
            w = g.iloc[max(0, p - k):p]
            out.at[idx, "s_recent_bad"] = w._bad.mean()
            out.at[idx, "s_recent_mal"] = w._mal.mean()
            out.at[idx, "s_recent_gap_days"] = (ts - t[p - 1]) / np.timedelta64(1, "D")
    return out


def behaviour_features(df, ref, meta, loo):
    """Per-message behavioural context relative to the sender's reference
    traffic (`ref` = all known messages incl. unlabelled test)."""
    acc = meta["accounts"].set_index("account_id")
    dev = meta["devices"].set_index("device_id")
    loc = meta["locations"].set_index("location_id")
    out = pd.DataFrame(index=df.index)
    hr = df.ts.dt.hour + df.ts.dt.minute / 60
    ws = df.sender_id.map(acc.normal_window_start)
    we = df.sender_id.map(acc.normal_window_end)
    inwin = np.where(ws <= we, (hr >= ws) & (hr <= we), (hr >= ws) | (hr <= we))
    out["in_window"] = inwin.astype(int)
    mid = np.where(ws <= we, (ws + we) / 2, ((ws + we + 24) / 2) % 24)
    dh = np.abs(hr - mid)
    out["hour_dist"] = np.minimum(dh, 24 - dh)
    out["hour"] = hr
    out["dow"] = df.ts.dt.dayofweek
    sp = df.sender_id.map(acc.person_id)
    out["own_device"] = (df.device_id.map(dev.owner_person_id) == sp).astype(int)
    out["dev_age_days"] = df.device_id.map(dev.first_seen_offset_days)
    out["acct_age_days"] = df.sender_id.map(acc.creation_offset_days)
    out["same_org_recip"] = (
        df.sender_id.map(acc.organization_id) == df.recipient_id.map(acc.organization_id)
    ).astype(int)
    out["lat"] = df.location_id.map(loc.latitude)
    out["lon"] = df.location_id.map(loc.longitude)
    out["len_chars"] = df.message.str.len()
    out["n_sent"] = df.message.str.count(r"[.!?]")
    out["excl"] = df.message.str.contains("!").astype(int)

    # Frequency of (sender, x) pairs in reference traffic, excluding self.
    for col in ["device_id", "location_id", "recipient_id"]:
        pair = ref.groupby(["sender_id", col]).size().rename("c")
        tot = ref.groupby("sender_id").size().rename("t")
        j = df[["sender_id", col]].join(pair, on=["sender_id", col]).join(tot, on="sender_id")
        c, t = j.c.fillna(0).values, j.t.fillna(0).values
        if loo:
            c, t = c - 1, t - 1
        out[f"pair_freq_{col}"] = c / np.maximum(t, 1)
        out[f"pair_new_{col}"] = (c <= 0).astype(int)
    # How many distinct accounts use this device; how many devices this sender uses.
    out["dev_n_accts"] = df.device_id.map(ref.groupby("device_id").sender_id.nunique())
    out["snd_n_devs"] = df.sender_id.map(ref.groupby("sender_id").device_id.nunique())
    out["snd_n_locs"] = df.sender_id.map(ref.groupby("sender_id").location_id.nunique())
    out["snd_n_msgs"] = df.sender_id.map(ref.groupby("sender_id").size())
    out["rcv_n_msgs"] = df.recipient_id.map(ref.groupby("recipient_id").size())
    out["rcv_n_senders"] = df.recipient_id.map(ref.groupby("recipient_id").sender_id.nunique())

    for c, s in [("message_category", df.message_category),
                 ("device_type", df.device_id.map(dev.device_type)),
                 ("location_type", df.location_id.map(loc.location_type)),
                 ("account_type", df.sender_id.map(acc.account_type)),
                 ("role", df.sender_id.map(acc.role)),
                 ("r_role", df.recipient_id.map(acc.role))]:
        out[c] = s.astype("category")
    return out


def opener(msgs):
    """The templated leading phrase up to the first ':' or '.'."""
    return msgs.str.extract(r"^([^:.!?]{0,40}[:.!?])")[0].fillna("")


def text_vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200000)


def text_oof(train_text, train_y, other_texts, seed=0, C=2.0):
    """OOF class probabilities on train; full-fit probabilities on others."""
    vec = text_vectorizer()
    X = vec.fit_transform(train_text)
    oof = np.zeros((X.shape[0], 3))
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    y = pd.Categorical(train_y, categories=LABELS).codes
    for a, b in skf.split(X, y):
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(X[a], y[a])
        oof[b] = m.predict_proba(X[b])
    m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(X, y)
    others = [m.predict_proba(vec.transform(t)) for t in other_texts]
    return oof, others
