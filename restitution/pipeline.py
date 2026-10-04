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
STAGE2 = os.environ.get("GRC_STAGE2", "1") == "1"  # graph-context second model + 2-hop candidates
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
                      ("country", [r"country"]),
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
    s = re.sub(r"^www\d?\.", "", s).replace("-", "").strip(".")
    return s if "." in s else ""


def domains_of(s):
    """All domains in a field that may hold several values or junk ("a-b.com,ab")."""
    return [d for d in (domain_of(x) for x in re.split(r"[,;|\s]+", s)) if d]


def skeleton(core):
    """S.W.O.R.D. style abbreviation: drop inner vowels of words with 4+ letters
    ("shakti food" -> "shkt fd", "great enterprises" -> "grt entrprss")."""
    out = []
    for t in core.split():
        if len(t) >= 4 and t.isalpha():
            t = t[0] + re.sub(r"[aeiou]", "", t[1:])
        out.append(re.sub(r"(.)\1+", r"\1", t))
    return " ".join(out)


COUNTRY = {"in": "in", "ind": "in", "india": "in", "bharat": "in", "us": "us", "usa": "us", "u s": "us",
           "u s a": "us", "united states": "us", "united states of america": "us", "america": "us",
           "fr": "fr", "fra": "fr", "france": "fr", "republique francaise": "fr"}
_STREETY = re.compile(r"\b(st|street|rd|road|ave|avenue|blvd|lane|ln|dr|drive|nagar|marg|rue|floor|flr|plot|"
                      r"sector|block|house|h no|po box|suite|apt)\b")
_JUNK = re.compile(r"\b(null|none|nan|n a|na|unknown)\b")


def addr_like(s):
    return bool(re.search(r"\d", s)) and bool(_STREETY.search(s))


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

    name = g("name").map(norm).str.replace(_JUNK, " ", regex=True).str.split().str.join(" ")
    addr0 = g("address").map(norm).str.replace(_JUNK, " ", regex=True).str.split().str.join(" ")
    # name and address sometimes arrive swapped
    swap = name.map(addr_like) & ~addr0.map(addr_like)
    name, addr0 = name.where(~swap, addr0), addr0.where(~swap, name)
    log(f"  swapped name/address on {int(swap.sum()):,} records")
    R["name"] = name.values
    R["name_fold"] = name.map(fold).values
    legal_skel = {skeleton(t) for t in LEGAL if len(skeleton(t)) >= 4}
    core = name.map(lambda s: " ".join(t for t in s.split() if t not in LEGAL and skeleton(t) not in legal_skel))
    R["core"] = core.values
    R["core_compact"] = core.str.replace(" ", "", regex=False).values
    R["skel"] = core.map(skeleton).values
    R["skel_compact"] = R["skel"].str.replace(" ", "", regex=False).values

    city_raw = g("city").map(norm)
    addr = addr0.map(lambda s: " ".join(ADDR_MAP.get(t, t) for t in s.split()))
    R["addr"] = addr.values
    pc_col = g("postcode").map(digits)
    pc_addr = addr.str.extract(r"\b(\d{5,6})\b")[0].fillna("")
    R["postcode"] = np.where(pc_col.str.len() >= 4, pc_col, pc_addr)
    R["housenum"] = addr.str.extract(r"\b(\d{1,5}[a-z]?)\b")[0].fillna("").values
    R["addr_nums"] = addr.map(lambda s: frozenset(re.findall(r"\d+", s))).values
    R["housenum"] = np.where(R["housenum"] == R["postcode"], "", R["housenum"])
    R["street_toks"] = addr.map(lambda s: [t for t in s.split() if len(t) >= 3 and not t.isdigit()
                                           and t not in ADDR_STOP]).values
    R["city"] = city_raw.values
    R["country"] = g("country").map(norm).map(lambda c: COUNTRY.get(c, c)).values

    ph = g("phone").map(digits)
    R["p9"] = ph.map(lambda d: d[-9:] if len(d) >= 7 else "").values
    R["p7"] = ph.map(lambda d: d[-7:] if len(d) >= 7 else "").values
    em = g("email").str.strip().str.lower()
    R["email_dom"] = em.map(lambda s: domain_of(s.split(",")[0])).values
    R["email_local"] = em.map(lambda s: s.split("@")[0] if "@" in s else "").values
    webs = g("website").map(domains_of)
    R["web_dom"] = webs.map(lambda x: x[0] if x else "").values
    # emails sometimes land in the website column and vice versa
    R["doms"] = [sorted(set(w) | set(domains_of(e))) for w, e in zip(webs, em)]
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


def knn_pairs(X, k=8, thr=0.5, max_df=1000, chunk=2000):
    """Top-k cosine neighbours on a TF-IDF matrix, ignoring very common n-grams."""
    n = X.shape[0]
    df = np.diff(X.tocsc().indptr)
    keep = np.flatnonzero((df >= 2) & (df <= max_df))
    X = X[:, keep].tocsr().astype(np.float32)
    nr = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
    X = sp.diags(1 / np.maximum(nr, 1e-9)).astype(np.float32) @ X
    XT = X.T.tocsr()
    out = []
    for s0 in range(0, n, chunk):
        S = (X[s0:s0 + chunk] @ XT).tocoo()
        m = (S.data >= thr) & (S.row + s0 != S.col)
        r, c, v = S.row[m] + s0, S.col[m], S.data[m]
        if not len(r):
            continue
        o = np.lexsort((-v, r))
        r, c = r[o], c[o]
        first = np.searchsorted(r, r, side="left")
        sel = (np.arange(len(r)) - first) < k
        r, c = r[sel].astype(np.int64), c[sel].astype(np.int64)
        out.append(np.minimum(r, c) * n + np.maximum(r, c))
    return np.unique(np.concatenate(out)) if out else np.zeros(0, np.int64)


def snm_pairs(keys, n, w=3):
    """Sorted neighbourhood: pair each non-empty key with its next w neighbours in sort order."""
    idx = np.flatnonzero(keys != "")
    if len(idx) < 2:
        return np.zeros(0, np.int64)
    o = idx[np.argsort(keys[idx], kind="stable")]
    out = []
    for d in range(1, w + 1):
        a, b = o[:-d], o[d:]
        out.append(np.minimum(a, b).astype(np.int64) * n + np.maximum(a, b))
    return np.concatenate(out)


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
    if os.environ.get("GRC_NEWBLOCKS", "1") != "1":
        allp = np.concatenate(allp)
        keys, cnt = np.unique(allp, return_counts=True)
        return (keys // n).astype(np.int64), (keys % n).astype(np.int64), cnt.astype(np.float32)
    add("core_exact", np.where(R["core_compact"].str.len() >= 4, R["core_compact"], ""), ar, 80)
    add("skel_exact", np.where(R["skel_compact"].str.len() >= 3, R["skel_compact"], ""), ar, 80)
    sk = R["skel"].map(lambda s: sorted({t for t in s.split() if len(t) >= 3})).tolist()
    k, i = explode_keys(sk)
    add("skel_tok", k, i, 40)
    k2 = np.array([a + "|" + R["loc"].iat[b] for a, b in zip(k, i)], dtype=object) if len(k) else k
    add("skel_tok+loc", k2, i, 60)
    skc = R["skel_compact"].to_numpy(dtype=object)
    p = np.concatenate([snm_pairs(skc, n, 3), snm_pairs(np.array([x[::-1] for x in skc], dtype=object), n, 3)])
    log(f"    block {'skel_snm':<14} {len(p):>10,} pairs")
    allp.append(p)
    p = knn_pairs(tfidf_mat(R["skel"], analyzer="char_wb", ngram_range=(3, 3)), k=10, thr=0.45)
    log(f"    block {'skel_knn':<14} {len(p):>10,} pairs")
    allp.append(p)
    p = knn_pairs(tfidf_mat(R["addr"], analyzer="char_wb", ngram_range=(3, 3)), k=8, thr=0.55)
    log(f"    block {'addr_knn':<14} {len(p):>10,} pairs")
    allp.append(p)
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
            "skel_c": tfidf_mat(R["skel"], analyzer="char_wb", ngram_range=(2, 3)),
            "skel_w": tfidf_mat(R["skel"], analyzer="word", token_pattern=r"\S+"),
        }
        cc = R["core_compact"].where(R["core_compact"] != "", None)
        per = 1e5 / max(1, self.n)  # counts as a rate per 100k records, so worlds of any size compare
        self.core_cnt = np.log1p(per * cc.map(cc.value_counts()).fillna(0).to_numpy(np.float32))
        cl = (R["core_compact"] + "|" + R["loc"]).where(R["core_compact"] != "", None)
        self.coreloc_cnt = np.log1p(per * cl.map(cl.value_counts()).fillna(0).to_numpy(np.float32))
        sc = R["skel_compact"].where(R["skel_compact"] != "", None)
        self.skel_cnt = np.log1p(per * sc.map(sc.value_counts()).fillna(0).to_numpy(np.float32))
        self.nums = R["addr_nums"].to_numpy(dtype=object)
        p9 = R["p9"].str.zfill(9).where(R["p9"] != "", "")
        self.P = np.array([[ord(c) - 48 for c in s] if s else [-1] * 9 for s in p9], np.int8)
        self.has_p = (R["p9"] != "").to_numpy()
        codes = lambda col: pd.factorize(R[col])[0].astype(np.int64)
        self.code = {c: codes(c) for c in ["housenum", "postcode", "report", "email_local", "city",
                                            "core_compact", "source", "country"]}
        self.empty = {c: (R[c] == "").to_numpy() for c in ["housenum", "postcode", "report", "email_local",
                                                            "city", "core_compact", "name", "addr", "country"]}
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

    def features(self, i, j, nkeys, known_sources, deg=None):
        F = {}
        for k, X in self.M.items():
            F[k] = rowdot(X, i, j)
        both_p = self.has_p[i] & self.has_p[j]
        ham = (self.P[i] != self.P[j]).sum(1).astype(np.float32)
        F["phone_both"] = both_p.astype(np.float32)
        F["phone_ham"] = np.where(both_p, ham, -1)
        F["phone_any"] = (self.has_p[i] | self.has_p[j]).astype(np.float32)
        for c in ["housenum", "postcode", "report", "email_local", "city", "core_compact", "country"]:
            both = ~self.empty[c][i] & ~self.empty[c][j]
            eq = self.code[c][i] == self.code[c][j]
            F[c + "_eq"] = np.where(both, eq.astype(np.float32), -1)
        for c in ["name", "addr"]:
            F[c + "_present"] = (~self.empty[c][i]).astype(np.float32) + (~self.empty[c][j])
        d1i, d2i, d1j, d2j = self.d1[i], self.d2[i], self.d1[j], self.d2[j]
        dm = ((d1i == d1j) | (d1i == d2j) | (d2i == d1j) | (d2i == d2j)) & (d1i != "") & (d1j != "")
        F["dom_eq"] = np.where((d1i != "") & (d1j != ""), dm.astype(np.float32), -1)
        fq = self.dom_freq.reindex(d1i).fillna(0).to_numpy()
        F["dom_freq"] = np.where(dm, np.log1p(fq * 1e5 / self.n), -1).astype(np.float32)
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
        for nm_, arr in (("core_cnt", self.core_cnt), ("coreloc_cnt", self.coreloc_cnt), ("skel_cnt", self.skel_cnt)):
            F[nm_ + "_min"] = np.minimum(arr[i], arr[j])
            F[nm_ + "_max"] = np.maximum(arr[i], arr[j])
        ni, nj = self.nums[i], self.nums[j]
        inter = np.fromiter((len(a & b) for a, b in zip(ni, nj)), np.float32, len(i))
        uni = np.fromiter((len(a | b) for a, b in zip(ni, nj)), np.float32, len(i))
        both = np.fromiter((bool(a) and bool(b) for a, b in zip(ni, nj)), bool, len(i))
        F["num_jacc"] = np.where(both, inter / np.maximum(uni, 1), -1).astype(np.float32)
        F["num_conflict"] = np.where(both, (inter == 0).astype(np.float32), -1)
        if deg is None:
            deg = np.bincount(i, minlength=self.n) + np.bincount(j, minlength=self.n)
        F["deg_min"] = (1e5 / self.n) * np.minimum(deg[i], deg[j]).astype(np.float32)
        F["deg_max"] = (1e5 / self.n) * np.maximum(deg[i], deg[j]).astype(np.float32)
        return pd.DataFrame(F)


def true_pair_keys(labels, maxb=200):
    n = len(labels)
    return np.unique(_pairs_from_keys(pd.Series(labels).astype(str).values, np.arange(n), n, maxb))


def build_pairs(R, labels=None, known_sources=(), cap=None, seed=0):
    W = World(R)
    log("  blocking")
    i, j, nk = block(R)
    if labels is not None:
        tk = true_pair_keys(labels)
        log(f"  pair recall BEFORE cheap filter: {(labels[i] == labels[j]).sum() / max(1, len(tk)):.4f}")
    cs = W.cheap(i, j)
    p_eq = W.has_p[i] & W.has_p[j] & ((W.P[i] == W.P[j]).all(1))
    keep = (cs >= 0.15) | (nk >= 2) | p_eq
    i, j, nk = i[keep], j[keep], nk[keep]
    log(f"  after cheap filter: {len(i):,}")
    W.deg = (np.bincount(i, minlength=W.n) + np.bincount(j, minlength=W.n)).astype(np.float32)
    if labels is not None:
        y = (labels[i] == labels[j]).astype(np.int8)
        log(f"  pair recall of candidates: {y.sum() / max(1, len(tk)):.4f}  ({y.sum():,}/{len(tk):,}), "
            f"positive rate {y.mean():.4f}")
        if cap and len(i) > cap:
            rng = np.random.default_rng(seed)
            sel = np.sort(rng.choice(len(i), cap, replace=False))
            i, j, nk, y = i[sel], j[sel], nk[sel], y[sel]
    else:
        y = None
    log("  features")
    X = W.features(i, j, nk, known_sources, deg=W.deg)
    return W, i, j, X, y


# ----------------------------------------------------------------------------- stage 2 (graph context)
def sym(n, i, j, v):
    A = sp.coo_matrix((v.astype(np.float32), (i, j)), shape=(n, n))
    return (A + A.T).tocsr()


def expand_pairs(n, i, j, p, thr=0.5, maxdeg=30, cap=3_000_000):
    """New candidates i-k-j where both hops are strong (transitive blocking)."""
    m = p >= thr
    S = sym(n, i[m], j[m], np.ones(m.sum(), np.float32))
    deg = np.diff(S.indptr)
    D = sp.diags((deg <= maxdeg).astype(np.float32))
    C = sp.triu(D @ S @ S @ D, k=1).tocoo()
    keys = C.row.astype(np.int64) * n + C.col
    old = np.sort(i.astype(np.int64) * n + j)
    pos = np.clip(np.searchsorted(old, keys), 0, len(old) - 1)
    new = old[pos] != keys
    keys, cnt = keys[new], C.data[new]
    if len(keys) > cap:
        o = np.argsort(-cnt, kind="stable")[:cap]
        keys, cnt = keys[o], cnt[o]
    return (keys // n).astype(np.int64), (keys % n).astype(np.int64)


def graph_features(n, i, j, p):
    A = sym(n, i, j, p)
    S = A.copy()
    S.data = (S.data >= 0.5).astype(np.float32)
    pmax = np.zeros(n, np.float32)
    np.maximum.at(pmax, i, p)
    np.maximum.at(pmax, j, p)
    psum = np.asarray(A.sum(1)).ravel()
    sdeg = np.asarray(S.sum(1)).ravel()
    near = np.zeros(n, np.float32)  # how many candidates are almost as good as the best one
    np.add.at(near, i, (p >= 0.8 * pmax[i]) & (p > 0.2))
    np.add.at(near, j, (p >= 0.8 * pmax[j]) & (p > 0.2))
    cw = rowdot(A, i, j)
    cs = rowdot(S, i, j)
    G = pd.DataFrame({
        "p1": p,
        "p_rel_min": np.minimum(p / np.maximum(pmax[i], 1e-6), p / np.maximum(pmax[j], 1e-6)),
        "p_rel_max": np.maximum(p / np.maximum(pmax[i], 1e-6), p / np.maximum(pmax[j], 1e-6)),
        "pmax_min": np.minimum(pmax[i], pmax[j]), "pmax_max": np.maximum(pmax[i], pmax[j]),
        "psum_min": np.minimum(psum[i], psum[j]), "psum_max": np.maximum(psum[i], psum[j]),
        "sdeg_min": np.minimum(sdeg[i], sdeg[j]), "sdeg_max": np.maximum(sdeg[i], sdeg[j]),
        "near_min": np.minimum(near[i], near[j]), "near_max": np.maximum(near[i], near[j]),
        "common_w": cw, "common_s": cs,
        "common_w_rel": cw / np.maximum(np.minimum(psum[i] - p, psum[j] - p), 1e-3),
    })
    return G.astype(np.float32)


# ----------------------------------------------------------------------------- diagnostics (validation only)
def _short(R, k):
    r = R.iloc[k]
    return (f"{r['source']}|{r['year']}|n={r['name'][:35]!r}|a={r['addr'][:40]!r}|c={r['city'][:15]!r}"
            f"|p={r['p9']}|d={','.join(r['doms'])[:30]}|rep={str(r['report'])[:12]}")


def diagnostics(R, labels, regime, i, j, p, pred, n_show=12, seed=0):
    from scipy.sparse.csgraph import connected_components
    n = len(R)
    rng = np.random.default_rng(seed)
    y = labels[i] == labels[j]
    _, comp = connected_components(sym(n, i[y], j[y], np.ones(y.sum())), directed=False)
    sc, by = companion_f05(labels, comp, regime)
    log("DIAG oracle ceiling (perfect model on these candidates): "
        f"{sc:.4f}  " + " ".join(f"{k}={v:.3f}" for k, v in sorted(by.items())))
    tk = true_pair_keys(labels)
    ti, tj = tk // n, tk % n
    ck = np.sort(i.astype(np.int64) * n + j)
    pos = np.clip(np.searchsorted(ck, tk), 0, len(ck) - 1)
    found = ck[pos] == tk
    reach = comp[ti] == comp[tj]
    src = R["source"].to_numpy(dtype=object)
    sp_ = np.where(src[ti] <= src[tj], src[ti] + "-" + src[tj], src[tj] + "-" + src[ti])
    t = pd.DataFrame({"pair": sp_, "found": found, "reach": reach})
    if regime is not None:
        t["regime"] = np.asarray(regime)[ti]
    tab = t.groupby("pair").agg(n=("found", "size"), cand_recall=("found", "mean"), reach_recall=("reach", "mean"))
    log("DIAG true pairs by source pair (cand_recall = pair is a candidate, reach_recall = connected via candidates):\n"
        + tab.sort_values("n", ascending=False).round(3).to_string())
    if regime is not None:
        log("DIAG by regime:\n" + t.groupby("regime").agg(n=("found", "size"), cand_recall=("found", "mean"),
                                                         reach_recall=("reach", "mean")).round(3).to_string())
    fill = pd.DataFrame({c: (R[c].astype(str).str.len() > 0) for c in
                         ["name", "addr", "city", "postcode", "housenum", "p9", "email_dom", "web_dom", "country"]})
    fill["year"] = R["year"] > 0
    log("DIAG field fill rate by source:\n" + fill.groupby(R["source"]).mean().round(2).to_string())

    miss = np.flatnonzero(~reach)
    log(f"DIAG examples of true pairs NOT reachable via candidates ({len(miss):,} total):")
    for k in rng.choice(miss, min(n_show, len(miss)), replace=False):
        print("   A", _short(R, ti[k]), "\n   B", _short(R, tj[k]), "\n")
    wrong = np.flatnonzero((pred[i] == pred[j]) & ~y)
    log(f"DIAG examples of FALSE MERGES (different businesses, same predicted cluster; {len(wrong):,} edges):")
    for k in rng.choice(wrong, min(n_show, len(wrong)), replace=False):
        print(f"   p={p[k]:.2f} regimes={regime[i[k]] if regime is not None else ''}/"
              f"{regime[j[k]] if regime is not None else ''}\n   A", _short(R, i[k]), "\n   B", _short(R, j[k]), "\n")
    split = np.flatnonzero((pred[i] != pred[j]) & y)
    log(f"DIAG examples of MISSED candidate links (same business, different clusters; {len(split):,} edges):")
    for k in rng.choice(split, min(n_show, len(split)), replace=False):
        print(f"   p={p[k]:.2f} regime={regime[i[k]] if regime is not None else ''}\n   A", _short(R, i[k]),
              "\n   B", _short(R, j[k]), "\n")


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

    df_tr = data["train"][0]
    log("raw training examples (3 per source):")
    for sname, g in df_tr.groupby(df_tr["obs_id"].str.split("-").str[0]):
        print(g.drop(columns=[data["train"][1]]).head(3).to_string(max_colwidth=45, index=False), "\n")
    src_train = prepare(df_tr, data["train"][3])
    known_sources = sorted(src_train["source"].value_counts().loc[lambda s: s > 100].index)
    log("sources:", src_train["source"].value_counts().to_dict())

    def world_pairs(k, R, cap=None):
        df, lab, reg, m = data[k]
        labels = pd.factorize(df[lab])[0]
        return (*build_pairs(R, labels, known_sources, cap=cap), labels)

    log("== train world")
    Wt, it, jt, Xt, yt, lt = world_pairs("train", src_train, cap=MAX_TRAIN_PAIRS)
    log("== validation world")
    Rv = prepare(data["val"][0], data["val"][3])
    Wv, iv, jv, Xv, yv, lv = world_pairs("val", Rv)
    reg_v = data["val"][0][data["val"][2]].to_numpy() if data["val"][2] else None

    params = dict(objective="binary", learning_rate=0.08, num_leaves=255, min_data_in_leaf=200, max_bin=127,
                  feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, num_threads=os.cpu_count())

    def fit(X, y, Xva=None, yva=None, rounds=3000):
        if Xva is None:
            return lgb.train(params, lgb.Dataset(X, y), num_boost_round=rounds)
        return lgb.train(params, lgb.Dataset(X, y), num_boost_round=rounds,
                         valid_sets=[lgb.Dataset(Xva, yva)],
                         callbacks=[lgb.early_stopping(60), lgb.log_evaluation(200)])

    def tune(n, i, j, p, labels, tag):
        best = (-1, None, None)
        for miss in (None, 0.0, 0.1, 0.2):
            for thr in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
                if miss is not None and miss >= thr:
                    continue
                sc, by = companion_f05(labels, cluster(n, i, j, p, thr, miss), reg_v)
                log(f"  [{tag}] miss={miss} thr={thr}: {sc:.4f}  " +
                    " ".join(f"{k}={v:.3f}" for k, v in sorted(by.items())))
                if sc > best[0]:
                    best = (sc, thr, miss)
        log(f"  [{tag}] BEST {best[0]:.4f} at thr={best[1]} miss={best[2]}")
        return best

    # ---------------- stage 1: pair model on record features
    log("stage 1: training")
    m1 = fit(Xt, yt, Xv, yv)
    r1 = m1.best_iteration
    imp = pd.Series(m1.feature_importance("gain"), index=Xt.columns).sort_values(ascending=False)
    log("stage 1 top features:\n" + imp.head(15).round(0).to_string())
    pv1 = m1.predict(Xv, num_iteration=r1).astype(np.float32)
    log(f"validation singleton baseline = {companion_f05(lv, np.arange(len(lv)), reg_v)[0]:.4f}")
    best1 = tune(Wv.n, iv, jv, pv1, lv, "stage1")

    if not STAGE2:
        m_final, best, pv, final_parts = m1, best1, pv1, None
    else:
        log("stage 1: out-of-fold predictions on train (3 folds by business)")
        fold = lt[it] % 3
        pt1 = np.zeros(len(it), np.float32)
        m1_folds = []
        for f in range(3):
            mf = fit(Xt[fold != f], yt[fold != f], rounds=r1)
            pt1[fold == f] = mf.predict(Xt[fold == f])
            m1_folds.append(mf)

        def stage2_set(W, i, j, X, p1, labels, predict1):
            ni, nj = expand_pairs(W.n, i, j, p1)
            log(f"  2-hop expansion: +{len(ni):,} pairs")
            if len(ni):
                Xn = W.features(ni, nj, np.zeros(len(ni), np.float32), known_sources, deg=W.deg)
                pn = predict1(ni, Xn)
                i, j = np.concatenate([i, ni]), np.concatenate([j, nj])
                X = pd.concat([X, Xn], ignore_index=True)
                p1 = np.concatenate([p1, pn]).astype(np.float32)
            G = graph_features(W.n, i, j, p1)
            X2 = pd.concat([X.reset_index(drop=True), G], axis=1)
            y = None if labels is None else (labels[i] == labels[j]).astype(np.int8)
            if labels is not None:
                log(f"  pair recall after expansion: {y.sum() / max(1, len(true_pair_keys(labels))):.4f}")
            return i, j, X2, y

        def pred_oof(ni, Xn):
            out = np.zeros(len(ni), np.float32)
            fo = lt[ni] % 3
            for f in range(3):
                if (fo == f).any():
                    out[fo == f] = m1_folds[f].predict(Xn[fo == f])
            return out

        log("stage 2: building graph features (train)")
        it2, jt2, Xt2, yt2 = stage2_set(Wt, it, jt, Xt, pt1, lt, pred_oof)
        log("stage 2: building graph features (validation)")
        iv2, jv2, Xv2, yv2 = stage2_set(Wv, iv, jv, Xv, pv1, lv, lambda ni, Xn: m1.predict(Xn, num_iteration=r1))
        log("stage 2: training")
        m2 = fit(Xt2, yt2, Xv2, yv2)
        r2 = m2.best_iteration
        imp = pd.Series(m2.feature_importance("gain"), index=Xt2.columns).sort_values(ascending=False)
        log("stage 2 top features:\n" + imp.head(15).round(0).to_string())
        pv2 = m2.predict(Xv2, num_iteration=r2).astype(np.float32)
        best2 = tune(Wv.n, iv2, jv2, pv2, lv, "stage2")
        iv, jv, pv, best = iv2, jv2, pv2, best2
        if best2[0] < best1[0]:
            log("!! stage 2 did not beat stage 1 on validation")

    # ---------------- diagnostics + error dump (validation only - never test)
    lab_pred = cluster(Wv.n, iv, jv, pv, best[1], best[2])
    diagnostics(Rv, lv, reg_v, iv, jv, pv, lab_pred)
    out = Rv[["obs_id", "source", "name", "addr", "city", "p9", "doms", "year", "report"]].copy()
    out["true"], out["pred"], out["regime"] = lv, lab_pred, reg_v if reg_v is not None else ""
    out.to_csv(os.path.join(OUT_DIR, "val_predictions.csv"), index=False)

    if "test" not in worlds or os.environ.get("GRC_SKIP_TEST") == "1":
        return
    # ---------------- refit on train + validation, then label the test world
    log("refitting on train + validation")
    m1_all = fit(pd.concat([Xt, Xv], ignore_index=True), np.concatenate([yt, yv]), rounds=int(r1 * 1.1))
    if STAGE2:
        m2_all = fit(pd.concat([Xt2, Xv2], ignore_index=True), np.concatenate([yt2, yv2]), rounds=int(r2 * 1.1))
        del Xt2, Xv2
    del Xt, Xv, Wt, Wv
    gc.collect()

    log("== test world")
    dft = read_table(worlds["test"])
    mt = map_columns(dft, None, None)
    log(f"  {len(dft):,} records, map={mt}")
    Rt = prepare(dft, mt)
    log("  test sources: " + str(Rt["source"].value_counts().to_dict()))
    Wte, ie, je, Xe, _ = build_pairs(Rt, None, known_sources)
    pe = m1_all.predict(Xe).astype(np.float32)
    if STAGE2:
        ie, je, Xe2, _ = stage2_set(Wte, ie, je, Xe, pe, None, lambda ni, Xn: m1_all.predict(Xn))
        pe = m2_all.predict(Xe2).astype(np.float32)
    roots = cluster(Wte.n, ie, je, pe, best[1], best[2])
    sub = pd.DataFrame({"obs_id": dft["obs_id"].values, "cluster": ["c" + str(r) for r in roots]})
    assert sub["obs_id"].is_unique and len(sub) == len(dft)
    path = os.path.join(OUT_DIR, "submission.csv")
    sub.to_csv(path, index=False)
    sizes = sub["cluster"].value_counts()
    log(f"wrote {path}: {len(sub):,} rows, {len(sizes):,} clusters, singletons {(sizes == 1).sum():,}, "
        f"largest {sizes.max()}")


if __name__ == "__main__":
    main()
