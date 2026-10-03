#!/usr/bin/env python3
"""GRC Restitution Docket - entity resolution pipeline (v1 baseline).

Runs end to end on Kaggle (or locally):
  1. finds the train / validation / test worlds under DATA_ROOT
  2. normalises every field (accents, Devanagari/Tamil transliteration, phones, domains)
  3. blocking: builds candidate pairs from shared phone / domain / rare name token /
     house number + street / report_id keys
  4. pair features (char + word TF-IDF cosines, phone hamming, house number, years ...)
  5. LightGBM pair classifier trained from scratch on the training world
  6. average-linkage clustering, threshold tuned on validation with the exact
     regime-weighted Companion F0.5
  7. refits on train + validation and writes submission.csv for the test world

Everything is trained from scratch on the provided data (TF-IDF + gradient boosting);
no pretrained weights, no external lookups. Test records never leave the notebook.

Usage on Kaggle: paste this file into one cell (or `%run pipeline.py`).
Env overrides: GRC_DATA (input root), GRC_OUT (output dir), GRC_QUICK=1 (subsample).
"""
import csv
import gc
import glob
import heapq
import math
import os
import re
import sys
import time
import unicodedata

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb

DATA_ROOT = os.environ.get("GRC_DATA", "/kaggle/input")
OUT_DIR = os.environ.get("GRC_OUT", "/kaggle/working" if os.path.isdir("/kaggle/working") else ".")
QUICK = os.environ.get("GRC_QUICK", "0") == "1"   # subsample worlds for a fast smoke run
REFIT_ON_VAL = True                               # final model trained on train + validation
MAX_TRAIN_PAIRS = 12_000_000
REGIME_W = {"A": 0.20, "B": 0.15, "C": 0.15, "D": 0.15, "E": 0.15, "F": 0.20}
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ----------------------------------------------------------------------------- loading
def read_table(path):
    sep = "\t" if path.endswith(".tsv") else ","
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False,
                       quoting=csv.QUOTE_NONE if sep == "\t" else csv.QUOTE_MINIMAL,
                       on_bad_lines="warn", encoding="utf-8")


def find_worlds(root):
    worlds = {}
    for p in sorted(glob.glob(os.path.join(root, "**", "observations.*"), recursive=True)):
        parts = [x.lower() for x in p.split(os.sep)[-4:-1]]
        for key, pats in (("train", ("train",)), ("val", ("valid", "val", "dev")), ("test", ("test",))):
            if any(any(q in part for q in pats) for part in parts) and key not in worlds:
                worlds[key] = p
                break
    return worlds


LABEL_PAT = r"entity|business|cluster|label|truth|gold|biz"


def load_world(path):
    df = read_table(path)
    d = os.path.dirname(path)
    others = [f for f in glob.glob(os.path.join(d, "*.tsv")) + glob.glob(os.path.join(d, "*.csv"))
              if os.path.abspath(f) != os.path.abspath(path) and "sample_submission" not in f]
    pending = []
    for f in others:
        t = read_table(f)
        if "obs_id" in t.columns:
            new = [c for c in t.columns if c not in df.columns]
            if new:
                df = df.merge(t[["obs_id"] + new], on="obs_id", how="left")
                log(f"  merged {os.path.basename(f)} -> {new}")
        else:
            pending.append((f, t))
    for f, t in pending:  # business-level files (e.g. business_id -> regime)
        common = [c for c in t.columns if c in df.columns]
        if len(common) == 1:
            new = [c for c in t.columns if c not in df.columns]
            df = df.merge(t.drop_duplicates(common), on=common, how="left")
            log(f"  merged {os.path.basename(f)} on {common} -> {new}")
    df = df.fillna("")
    lab = [c for c in df.columns if c != "obs_id" and re.search(LABEL_PAT, c.lower())
           and c.lower() not in ("name", "business_name")]
    reg = [c for c in df.columns if "regime" in c.lower()]
    return df, (lab[0] if lab else None), (reg[0] if reg else None)


def pick(cols, pats, used):
    for p in pats:
        for c in cols:
            if c not in used and re.search(p, c.lower()):
                return c
    return None


def map_columns(df, label_col, regime_col):
    used = {"obs_id", label_col, regime_col}
    m = {}
    for key, pats in [("source", [r"^source$", r"network", r"^src"]),
                      ("report", [r"report", r"case"]),
                      ("year", [r"^year$", r"year", r"date"]),
                      ("email", [r"e.?mail"]),
                      ("website", [r"web", r"site", r"url", r"domain"]),
                      ("phone", [r"phone", r"tel", r"mobile"]),
                      ("postcode", [r"post", r"zip", r"^pin"]),
                      ("city", [r"city", r"town", r"locality"]),
                      ("address", [r"addr", r"street", r"location"]),
                      ("name", [r"^name$", r"name", r"title"])]:
        c = pick(df.columns, pats, used)
        if c is not None:
            m[key] = c
            used.add(c)
    m["extra"] = [c for c in df.columns if c not in used]
    return m


# ----------------------------------------------------------------------------- normalisation
_INDIC_CLEAN = re.compile(r"\b(candra|short|prishthamatra|vocalic)\b ?")
INDIC = ("DEVANAGARI", "TAMIL", "BENGALI", "GURMUKHI", "GUJARATI", "TELUGU", "KANNADA", "MALAYALAM", "ORIYA")


def translit(s):
    """Rough romanisation of Indic scripts via Unicode character names (no external data)."""
    if s.isascii():
        return s
    out = []
    for ch in s:
        if ord(ch) < 128:
            out.append(ch)
            continue
        try:
            nm = unicodedata.name(ch)
        except ValueError:
            out.append(" ")
            continue
        if not nm.startswith(INDIC):
            out.append(ch)
            continue
        dg = unicodedata.digit(ch, None)
        if dg is not None:
            out.append(str(dg))
        elif " LETTER " in nm:
            out.append(_INDIC_CLEAN.sub("", nm.split(" LETTER ", 1)[1].lower()).replace(" ", ""))
        elif "VOWEL SIGN" in nm:
            if out and out[-1].endswith("a"):
                out[-1] = out[-1][:-1]
            out.append(_INDIC_CLEAN.sub("", nm.split("VOWEL SIGN ", 1)[1].lower()).replace(" ", ""))
        elif "VIRAMA" in nm:
            if out and out[-1].endswith("a"):
                out[-1] = out[-1][:-1]
        elif "ANUSVARA" in nm or "CANDRABINDU" in nm:
            out.append("n")
        elif "VISARGA" in nm:
            out.append("h")
        elif "NUKTA" in nm:
            pass
        else:
            out.append(" ")
    return "".join(out)


_NONALNUM = re.compile(r"[^a-z0-9]+")


def norm(s):
    s = translit(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.casefold().replace("&", " and ").replace("'", "")
    # non-ASCII digits -> ASCII
    s = "".join(str(unicodedata.digit(c)) if (not c.isascii() and unicodedata.digit(c, None) is not None) else c
                for c in s)
    return _NONALNUM.sub(" ", s).strip()


_FOLD = [(re.compile(p), r) for p, r in [(r"ph", "f"), (r"ck", "k"), (r"q", "k"), (r"w", "v"), (r"z", "s"),
                                           (r"th", "t"), (r"sh", "s"), (r"(.)\1+", r"\1")]]


def fold(s):
    for p, r in _FOLD:
        s = p.sub(r, s)
    return s


LEGAL = set("llc inc ltd pvt private limited co corp corporation company the sarl sas sa eurl sasu "
            "and of plc llp group enterprises enterprise".split())
ADDR_MAP = {"road": "rd", "street": "st", "avenue": "ave", "av": "ave", "boulevard": "blvd", "bd": "blvd",
            "drive": "dr", "lane": "ln", "place": "pl", "court": "ct", "highway": "hwy", "suite": "ste",
            "north": "n", "south": "s", "east": "e", "west": "w", "floor": "fl", "building": "bldg",
            "nagar": "ngr", "marg": "mg", "saint": "st", "sainte": "ste"}
ADDR_STOP = set("rd st ave blvd dr ln pl ct hwy ste n s e w fl bldg rue de du la le les des near by "
                "the and of opp behind po box no".split())
GENERIC_TLD = re.compile(r"\.(com|net|org|co|in|fr|us|io|biz|info|wk|wa)(\.[a-z]{2})?$")


def domain_of(s):
    s = s.strip().lower()
    if not s:
        return ""
    if "@" in s:
        s = s.split("@", 1)[1]
    s = re.sub(r"^[a-z]+://", "", s).split("/")[0].split("?")[0]
    s = re.sub(r"^www\d?\.", "", s)
    return s if "." in s else ""


def digits(s):
    return "".join(str(unicodedata.digit(c)) for c in s if unicodedata.digit(c, None) is not None)


def prepare(df, m):
    g = lambda k: df[m[k]].astype(str) if k in m else pd.Series([""] * len(df), index=df.index)
    R = pd.DataFrame(index=df.index)
    R["obs_id"] = df["obs_id"].values
    src = g("source")
    R["source"] = np.where(src.str.len() > 0, src.str.upper(), df["obs_id"].str.split("-").str[0].str.upper())
    R["report"] = g("report").values
    yr = g("year").str.extract(r"((?:19|20)\d\d)")[0]
    R["year"] = pd.to_numeric(yr, errors="coerce").fillna(-1).astype(int).values

    name = g("name").map(norm)
    R["name"] = name.values
    R["name_fold"] = name.map(fold).values
    core = name.map(lambda s: " ".join(t for t in s.split() if t not in LEGAL))
    R["core"] = core.values
    R["core_compact"] = core.str.replace(" ", "", regex=False).values

    city_raw = g("city").map(norm)
    addr = (g("address")).map(norm).map(lambda s: " ".join(ADDR_MAP.get(t, t) for t in s.split()))
    R["addr"] = addr.values
    pc_col = g("postcode").map(digits)
    pc_addr = addr.str.extract(r"\b(\d{5,6})\b")[0].fillna("")
    R["postcode"] = np.where(pc_col.str.len() >= 4, pc_col, pc_addr)
    R["housenum"] = addr.str.extract(r"^(?:[a-z]{0,3}\s)?(\d{1,5}[a-z]?)\b")[0].fillna("").values
    R["housenum"] = np.where(R["housenum"] == R["postcode"], "", R["housenum"])
    R["street_toks"] = addr.map(lambda s: [t for t in s.split() if len(t) >= 3 and not t.isdigit()
                                           and t not in ADDR_STOP]).values
    R["city"] = city_raw.values

    ph = g("phone").map(digits)
    R["p9"] = ph.map(lambda d: d[-9:] if len(d) >= 7 else "").values
    R["p7"] = ph.map(lambda d: d[-7:] if len(d) >= 7 else "").values
    em = g("email").str.strip().str.lower()
    R["email_dom"] = em.map(domain_of).values
    R["email_local"] = em.map(lambda s: s.split("@")[0] if "@" in s else "").values
    R["web_dom"] = g("website").map(domain_of).values
    # emails sometimes land in the website column and vice versa
    R["doms"] = [sorted({d for d in (a, b) if d}) for a, b in zip(R["email_dom"], R["web_dom"])]
    stems = R["doms"].map(lambda ds: " ".join(GENERIC_TLD.sub("", d).replace(".", " ").replace("-", "")
                                                for d in ds))
    R["brand"] = (R["core"] + " " + stems).str.strip().values

    extra = [norm(" ".join(v)) for v in zip(*[df[c].astype(str) for c in m["extra"]])] if m["extra"] else [""] * len(df)
    R["alltext"] = (R["name"] + " " + R["addr"] + " " + R["city"] + " " + pd.Series(extra, index=df.index)
                    + " " + stems).str.split().str.join(" ").values
    loc = R["postcode"].where(R["postcode"] != "", R["city"].str.split().str[-1].fillna(""))
    loc = loc.where(loc != "", addr.str.split().str[-1].fillna(""))
    R["loc"] = loc.values
    R["n_fields"] = ((R["name"] != "").astype(int) + (R["addr"] != "") + (R["p9"] != "") +
                     (R["doms"].str.len() > 0) + (R["city"] != "") + (R["year"] > 0))
    return R.reset_index(drop=True)


# ----------------------------------------------------------------------------- blocking
def _pairs_from_keys(keys, idx, n, maxb):
    d = pd.DataFrame({"k": keys, "i": idx})
    d = d[d["k"].str.len() > 0].drop_duplicates()
    if d.empty:
        return np.zeros(0, np.int64)
    d["s"] = d.groupby("k")["i"].transform("size")
    d = d[(d["s"] >= 2) & (d["s"] <= maxb)].sort_values(["s", "k", "i"])
    out = []
    for s, gdf in d.groupby("s", sort=False):
        M = gdf["i"].to_numpy().reshape(-1, s)
        r, c = np.triu_indices(s, 1)
        a, b = M[:, r].ravel(), M[:, c].ravel()
        out.append(np.minimum(a, b).astype(np.int64) * n + np.maximum(a, b))
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def explode_keys(lists):
    idx = np.repeat(np.arange(len(lists)), [len(x) for x in lists])
    keys = np.array([k for x in lists for k in x], dtype=object)
    return keys, idx


def block(R):
    n = len(R)
    allp = []

    def add(name, keys, idx, maxb):
        p = _pairs_from_keys(pd.Series(keys, dtype=object).astype(str).values, idx, n, maxb)
        log(f"    block {name:<14} {len(p):>10,} pairs")
        allp.append(p)

    ar = np.arange(n)
    add("phone9", R["p9"].values, ar, 60)
    add("phone7+loc", np.where(R["p7"] != "", R["p7"] + "|" + R["loc"], ""), ar, 30)
    k, i = explode_keys(R["doms"].tolist())
    add("domain", k, i, 60)
    add("email_local", np.where(R["email_local"].str.len() >= 4, R["email_local"] + "|" + R["loc"], ""), ar, 30)

    name_toks = R["core"].map(lambda s: sorted({t for t in s.split() if len(t) >= 3})).tolist()
    k, i = explode_keys(name_toks)
    add("name_tok", k, i, 40)
    k2 = np.array([a + "|" + R["loc"].iat[b] for a, b in zip(k, i)], dtype=object) if len(k) else k
    add("name_tok+loc", k2, i, 60)
    fk = R["name_fold"].map(lambda s: sorted({t for t in s.split() if len(t) >= 4})).tolist()
    k, i = explode_keys(fk)
    add("fold_tok", k, i, 40)
    add("prefix6+loc", np.where(R["core_compact"].str.len() >= 4,
                                R["core_compact"].str[:6] + "|" + R["loc"], ""), ar, 60)
    st = [[h + "|" + t for t in toks] if h else [] for h, toks in zip(R["housenum"], R["street_toks"])]
    k, i = explode_keys(st)
    add("house+street", k, i, 40)
    add("house+pc", np.where((R["housenum"] != "") & (R["postcode"] != ""),
                             R["housenum"] + "|" + R["postcode"], ""), ar, 40)
    add("report", R["report"].values, ar, 25)
    at = R["alltext"].map(lambda s: sorted({t for t in s.split() if len(t) >= 4})).tolist()
    k, i = explode_keys(at)
    add("rare_text", k, i, 6)

    allp = np.concatenate(allp)
    keys, cnt = np.unique(allp, return_counts=True)
    i, j = keys // n, keys % n
    log(f"  candidates: {len(keys):,}")
    return i.astype(np.int64), j.astype(np.int64), cnt.astype(np.float32)


# ----------------------------------------------------------------------------- features
def tfidf_mat(texts, **kw):
    v = TfidfVectorizer(dtype=np.float32, sublinear_tf=True, min_df=1, **kw)
    try:
        return v.fit_transform(texts)
    except ValueError:  # empty vocabulary
        return sp.csr_matrix((len(texts), 1), dtype=np.float32)


def rowdot(X, i, j, chunk=1_000_000):
    out = np.empty(len(i), np.float32)
    for s in range(0, len(i), chunk):
        a, b = i[s:s + chunk], j[s:s + chunk]
        out[s:s + chunk] = np.asarray(X[a].multiply(X[b]).sum(1)).ravel()
    return out


class World:
    """Per-world matrices built only from that world's own records."""

    def __init__(self, R):
        self.R = R
        self.n = len(R)
        log("  tf-idf matrices")
        self.M = {
            "name_c": tfidf_mat(R["name"], analyzer="char_wb", ngram_range=(2, 4)),
            "name_w": tfidf_mat(R["name"], analyzer="word", token_pattern=r"\S+"),
            "fold_c": tfidf_mat(R["name_fold"], analyzer="char_wb", ngram_range=(2, 3)),
            "brand_c": tfidf_mat(R["brand"], analyzer="char_wb", ngram_range=(3, 4)),
            "addr_c": tfidf_mat(R["addr"], analyzer="char_wb", ngram_range=(2, 4)),
            "addr_w": tfidf_mat(R["addr"], analyzer="word", token_pattern=r"\S+"),
            "all_c": tfidf_mat(R["alltext"], analyzer="char_wb", ngram_range=(3, 4)),
            "all_w": tfidf_mat(R["alltext"], analyzer="word", token_pattern=r"\S+"),
        }
        p9 = R["p9"].str.zfill(9).where(R["p9"] != "", "")
        self.P = np.array([[ord(c) - 48 for c in s] if s else [-1] * 9 for s in p9], np.int8)
        self.has_p = (R["p9"] != "").to_numpy()
        codes = lambda col: pd.factorize(R[col])[0].astype(np.int64)
        self.code = {c: codes(c) for c in ["housenum", "postcode", "report", "email_local", "city",
                                            "core_compact", "source"]}
        self.empty = {c: (R[c] == "").to_numpy() for c in ["housenum", "postcode", "report", "email_local",
                                                            "city", "core_compact", "name", "addr"]}
        dc = pd.Series([d for ds in R["doms"] for d in ds]).value_counts()
        self.dom_freq = dc
        self.d1 = R["doms"].map(lambda x: x[0] if x else "").to_numpy(dtype=object)
        self.d2 = R["doms"].map(lambda x: x[-1] if x else "").to_numpy(dtype=object)
        self.year = R["year"].to_numpy()
        self.nf = R["n_fields"].to_numpy()
        self.nlen = R["name"].str.len().to_numpy()
        self.sources = R["source"].to_numpy(dtype=object)

    def cheap(self, i, j):
        s = np.maximum.reduce([rowdot(self.M["name_c"], i, j), rowdot(self.M["addr_c"], i, j),
                               rowdot(self.M["all_c"], i, j)])
        return s

    def features(self, i, j, nkeys, known_sources):
        F = {}
        for k, X in self.M.items():
            F[k] = rowdot(X, i, j)
        both_p = self.has_p[i] & self.has_p[j]
        ham = (self.P[i] != self.P[j]).sum(1).astype(np.float32)
        F["phone_both"] = both_p.astype(np.float32)
        F["phone_ham"] = np.where(both_p, ham, -1)
        F["phone_any"] = (self.has_p[i] | self.has_p[j]).astype(np.float32)
        for c in ["housenum", "postcode", "report", "email_local", "city", "core_compact"]:
            both = ~self.empty[c][i] & ~self.empty[c][j]
            eq = self.code[c][i] == self.code[c][j]
            F[c + "_eq"] = np.where(both, eq.astype(np.float32), -1)
        for c in ["name", "addr"]:
            F[c + "_present"] = (~self.empty[c][i]).astype(np.float32) + (~self.empty[c][j])
        d1i, d2i, d1j, d2j = self.d1[i], self.d2[i], self.d1[j], self.d2[j]
        dm = ((d1i == d1j) | (d1i == d2j) | (d2i == d1j) | (d2i == d2j)) & (d1i != "") & (d1j != "")
        F["dom_eq"] = np.where((d1i != "") & (d1j != ""), dm.astype(np.float32), -1)
        fq = self.dom_freq.reindex(d1i).fillna(0).to_numpy()
        F["dom_freq"] = np.where(dm, np.log1p(fq), -1).astype(np.float32)
        yi, yj = self.year[i], self.year[j]
        by = (yi > 0) & (yj > 0)
        F["year_diff"] = np.where(by, np.abs(yi - yj), -1).astype(np.float32)
        F["year_min"] = np.where(by, np.minimum(yi, yj), -1).astype(np.float32)
        F["year_max"] = np.where(by, np.maximum(yi, yj), -1).astype(np.float32)
        F["blip_span"] = (by & (np.minimum(yi, yj) < 2018) & (np.maximum(yi, yj) >= 2023)).astype(np.float32)
        F["nf_min"] = np.minimum(self.nf[i], self.nf[j]).astype(np.float32)
        F["nf_max"] = np.maximum(self.nf[i], self.nf[j]).astype(np.float32)
        F["nlen_ratio"] = (np.minimum(self.nlen[i], self.nlen[j]) /
                           np.maximum(1, np.maximum(self.nlen[i], self.nlen[j]))).astype(np.float32)
        cc = self.R["core_compact"].to_numpy(dtype=object)
        a, b = cc[i], cc[j]
        F["prefix"] = np.fromiter(((len(x) >= 3 and len(y) >= 3 and (x.startswith(y) or y.startswith(x)))
                                   for x, y in zip(a, b)), np.float32, len(i))
        F["same_source"] = (self.code["source"][i] == self.code["source"][j]).astype(np.float32)
        si, sj = self.sources[i], self.sources[j]
        for s in known_sources:
            F["src_" + s] = ((si == s).astype(np.float32) + (sj == s))
        F["nkeys"] = nkeys
        deg = np.bincount(i, minlength=self.n) + np.bincount(j, minlength=self.n)
        F["deg_min"] = np.minimum(deg[i], deg[j]).astype(np.float32)
        F["deg_max"] = np.maximum(deg[i], deg[j]).astype(np.float32)
        return pd.DataFrame(F)


def build_pairs(R, labels=None, known_sources=(), cap=None, seed=0):
    W = World(R)
    log("  blocking")
    i, j, nk = block(R)
    cs = W.cheap(i, j)
    p_eq = W.has_p[i] & W.has_p[j] & ((W.P[i] == W.P[j]).all(1))
    keep = (cs >= 0.25) | (nk >= 2) | p_eq
    i, j, nk = i[keep], j[keep], nk[keep]
    log(f"  after cheap filter: {len(i):,}")
    if labels is not None:
        y = (labels[i] == labels[j]).astype(np.int8)
        tot = pd.Series(labels).value_counts()
        tot_pairs = int((tot * (tot - 1) // 2).sum())
        log(f"  pair recall of candidates: {y.sum() / max(1, tot_pairs):.4f}  ({y.sum():,}/{tot_pairs:,}), "
            f"positive rate {y.mean():.4f}")
        if cap and len(i) > cap:
            rng = np.random.default_rng(seed)
            sel = np.sort(rng.choice(len(i), cap, replace=False))
            i, j, nk, y = i[sel], j[sel], nk[sel], y[sel]
    else:
        y = None
    log("  features")
    X = W.features(i, j, nk, known_sources)
    return W, i, j, X, y


# ----------------------------------------------------------------------------- clustering
def cluster(n, i, j, p, thr, miss):
    """Greedy average linkage. Unscored pairs between two clusters count as `miss`
    (miss=None: average over scored pairs only)."""
    parent = np.arange(n)
    size = np.ones(n, np.int64)
    adj = {}
    keep = p >= 0.02
    for a, b, q in zip(i[keep].tolist(), j[keep].tolist(), p[keep].tolist()):
        adj.setdefault(a, {})[b] = [q, 1]
        adj.setdefault(b, {})[a] = [q, 1]

    def score(a, b, sc):
        s, c = sc
        if miss is None:
            return s / c
        return (s + miss * (size[a] * size[b] - c)) / (size[a] * size[b])

    heap = [(-q, a, b) for a, b, q in zip(i.tolist(), j.tolist(), p.tolist()) if q >= thr]
    heapq.heapify(heap)
    while heap:
        negq, a, b = heapq.heappop(heap)
        if parent[a] != a or parent[b] != b:
            continue
        sc = adj.get(a, {}).get(b)
        if sc is None:
            continue
        cur = score(a, b, sc)
        if abs(cur + negq) > 1e-9:  # stale entry
            if cur >= thr:
                heapq.heappush(heap, (-cur, a, b))
            continue
        if cur < thr:
            break
        if len(adj.get(a, {})) < len(adj.get(b, {})):
            a, b = b, a
        parent[b] = a
        size[a] += size[b]
        Ab = adj.pop(b, {})
        Aa = adj.setdefault(a, {})
        Aa.pop(b, None)
        for k, (s, c) in Ab.items():
            if k == a:
                continue
            adj[k].pop(b, None)
            e = Aa.get(k)
            if e is None:
                e = [0.0, 0]
                Aa[k] = e
                adj[k][a] = e
            e[0] += s
            e[1] += c
        for k, e in Aa.items():
            q = score(a, k, e)
            if q >= thr:
                heapq.heappush(heap, (-q, a, k))
    # resolve roots
    root = parent.copy()
    while True:
        nr = root[root]
        if (nr == root).all():
            break
        root = nr
    return root


# ----------------------------------------------------------------------------- metric
def companion_f05(true, pred, regime=None, verbose=True):
    d = pd.DataFrame({"t": np.asarray(true), "p": np.asarray(pred)})
    ts = d.groupby("t")["t"].transform("size").to_numpy()
    ps = d.groupby("p")["p"].transform("size").to_numpy()
    it = d.groupby(["t", "p"])["t"].transform("size").to_numpy()
    tp, fp, fn = it - 1, ps - it, ts - it
    den = 1.25 * tp + 0.25 * fn + fp
    s = np.where(den == 0, 1.0, 1.25 * tp / np.maximum(den, 1e-12))
    if regime is None:
        return float(s.mean()), {}
    by = pd.Series(s).groupby(np.asarray(regime)).mean().to_dict()
    ws = {k: REGIME_W.get(str(k).strip()[:1].upper(), 0) for k in by}
    tot = sum(ws.values())
    score = sum(by[k] * ws[k] for k in by) / tot if tot else float(s.mean())
    return float(score), by


# ----------------------------------------------------------------------------- main
def main():
    worlds = find_worlds(DATA_ROOT)
    log("worlds:", worlds)
    for k in ("train", "val", "test"):
        if k not in worlds:
            print(f"!! could not find the {k} world under {DATA_ROOT}. Files seen:")
            for p in glob.glob(os.path.join(DATA_ROOT, "**", "*.*"), recursive=True)[:50]:
                print("   ", p)
            if k != "test":
                sys.exit(1)

    data = {}
    for k in ("train", "val"):
        log(f"loading {k}")
        df, lab, reg = load_world(worlds[k])
        if lab is None:
            print(f"!! no label column found in {k}; columns are {list(df.columns)}. Set LABEL_PAT."); sys.exit(1)
        if QUICK:
            keep = df[lab].drop_duplicates().sample(frac=0.15, random_state=0)
            df = df[df[lab].isin(keep)].reset_index(drop=True)
        m = map_columns(df, lab, reg)
        log(f"  {len(df):,} records, label={lab}, regime={reg}, map={m}")
        data[k] = (df, lab, reg, m)

    src_train = prepare(data["train"][0], data["train"][3])
    known_sources = sorted(src_train["source"].value_counts().loc[lambda s: s > 100].index)
    log("sources:", src_train["source"].value_counts().to_dict())

    def world_pairs(k, R, cap=None):
        df, lab, reg, m = data[k]
        labels = pd.factorize(df[lab])[0]
        return (*build_pairs(R, labels, known_sources, cap=cap), labels)

    log("== train world")
    Wt, it, jt, Xt, yt, _ = world_pairs("train", src_train, cap=MAX_TRAIN_PAIRS)
    log("== validation world")
    Rv = prepare(data["val"][0], data["val"][3])
    Wv, iv, jv, Xv, yv, lv = world_pairs("val", Rv)

    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, num_threads=os.cpu_count())
    log("training LightGBM")
    dtr = lgb.Dataset(Xt, yt, free_raw_data=False)
    dva = lgb.Dataset(Xv, yv, reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    best_iter = model.best_iteration
    imp = pd.Series(model.feature_importance("gain"), index=Xt.columns).sort_values(ascending=False)
    log("top features:\n" + imp.head(20).round(0).to_string())

    pv = model.predict(Xv, num_iteration=best_iter).astype(np.float32)
    reg_v = data["val"][0][data["val"][2]].to_numpy() if data["val"][2] else None
    log(f"validation: singleton baseline = {companion_f05(lv, np.arange(len(lv)), reg_v)[0]:.4f}, "
        f"oracle-on-candidates ceiling uses pair recall above")
    best = (-1, None, None)
    for miss in (None, 0.0, 0.2, 0.4):
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            lab_pred = cluster(Wv.n, iv, jv, pv, thr, miss)
            sc, by = companion_f05(lv, lab_pred, reg_v)
            log(f"  miss={miss} thr={thr}: {sc:.4f}  " + " ".join(f"{k}={v:.3f}" for k, v in sorted(by.items())))
            if sc > best[0]:
                best = (sc, thr, miss)
    log(f"BEST validation score {best[0]:.4f} at thr={best[1]} miss={best[2]}")

    # error dump for analysis (validation only - never test)
    lab_pred = cluster(Wv.n, iv, jv, pv, best[1], best[2])
    Rv["pred"] = lab_pred
    Rv["true"] = lv
    Rv["regime"] = reg_v if reg_v is not None else ""
    Rv[["obs_id", "source", "name", "addr", "city", "p9", "doms", "year", "report", "true", "pred", "regime"]] \
        .to_csv(os.path.join(OUT_DIR, "val_predictions.csv"), index=False)

    if "test" not in worlds:
        return
    if REFIT_ON_VAL:
        log("refitting on train + validation")
        Xa = pd.concat([Xt, Xv], ignore_index=True)
        ya = np.concatenate([yt, yv])
        del dtr, dva
        gc.collect()
        model = lgb.train(params, lgb.Dataset(Xa, ya), num_boost_round=int(best_iter * 1.1))
        best_iter = None
        del Xa, ya
    del Xt, Xv, Wt, Wv
    gc.collect()

    log("== test world")
    dft = read_table(worlds["test"])
    mt = map_columns(dft, None, None)
    log(f"  {len(dft):,} records, map={mt}")
    Rt = prepare(dft, mt)
    log("  test sources: " + str(Rt["source"].value_counts().to_dict()))
    Wte, ie, je, Xe, _ = build_pairs(Rt, None, known_sources)
    pe = model.predict(Xe, num_iteration=best_iter).astype(np.float32)
    roots = cluster(Wte.n, ie, je, pe, best[1], best[2])
    sub = pd.DataFrame({"obs_id": dft["obs_id"].values, "cluster": ["c" + str(r) for r in roots]})
    assert sub["obs_id"].is_unique and len(sub) == len(dft)
    path = os.path.join(OUT_DIR, "submission.csv")
    sub.to_csv(path, index=False)
    log(f"wrote {path}: {len(sub):,} rows, {sub['cluster'].nunique():,} clusters")


if __name__ == "__main__":
    main()
