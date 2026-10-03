"""End-to-end pipeline: writes the three task CSVs plus the combined
sample-format submission.

    python run.py --data ../data/upload_structure --out submission
"""
import argparse
import os

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import louvain_communities

import task1
from features import LABELS, load, template_group, template_maps
from hostile_types import account_types

TS_FMT = "%Y-%m-%d %H:%M:%S"


def detect_switchers(lab):
    """Clean accounts that start sending MALICIOUS (account takeover).
    Clean accounts otherwise only ever send TRUSTWORTHY on benign templates and
    SUSPICIOUS on covert templates, so any deviation marks the takeover."""
    g, _, _ = template_group(lab, template_maps(lab))
    lab = lab.assign(grp=g.values).sort_values("ts")
    out = {}
    for s, x in lab.groupby("sender_id"):
        bad = x.trust_label != "TRUSTWORTHY"
        if bad.mean() >= 0.6:
            continue
        odd = (x.trust_label == "MALICIOUS") | ((x.trust_label == "SUSPICIOUS") & (x.grp != "cov"))
        if odd.any():
            out[s] = x.ts[odd].min()
    return pd.Series(out, dtype="datetime64[ns]")


def detect_test_switchers(te, p, types):
    """Previously clean accounts whose test traffic the model calls MALICIOUS."""
    clean = te.sender_id.map(types).fillna("N").eq("N")
    hit = te[clean & (p[:, 2] > 0.6)]
    return hit.groupby("sender_id").ts.min()


def campaign_clusters(allm, types, extra_members):
    """Hostile M-type accounts message each other ~5x more than chance and form
    tight communities: those communities are the campaign cells."""
    members = set(types[types == "M"].index) | set(extra_members)
    e = allm[allm.sender_id.isin(members) & allm.recipient_id.isin(members)]
    G = nx.Graph()
    # Sorted insertion keeps Louvain deterministic across runs (set order
    # depends on per-process string hashing).
    G.add_nodes_from(sorted(members))
    for a, b in sorted(zip(e.sender_id, e.recipient_id)):
        if a != b:
            w = G.get_edge_data(a, b, {"weight": 0})["weight"]
            G.add_edge(a, b, weight=w + 1)
    comms = louvain_communities(G.subgraph([n for n in G if G.degree(n) > 0]), weight="weight", seed=0)
    comms = sorted(comms, key=len, reverse=True)
    lab = {a: i for i, c in enumerate(comms) for a in c}
    # Isolated members: join the community they share most contacts with.
    contacts = pd.concat([allm[["sender_id", "recipient_id"]],
                          allm[["recipient_id", "sender_id"]].set_axis(["sender_id", "recipient_id"], axis=1)])
    nb = contacts.groupby("sender_id").recipient_id.agg(set)
    next_id = len(comms)
    for a in sorted(members - set(lab)):
        score = {}
        for c in sorted(nb.get(a, set())):
            for d in sorted(nb.get(c, set())):
                if d in lab:
                    score[lab[d]] = score.get(lab[d], 0) + 1
        if score:
            lab[a] = max(score, key=score.get)
        else:
            lab[a], next_id = next_id, next_id + 1
    return lab


def main(data, out, variant):
    os.makedirs(out, exist_ok=True)
    tr, va, te, meta = load(data)
    lab = pd.concat([tr, va])
    allm = pd.concat([tr, va, te])
    accounts = meta["accounts"].account_id

    # ---- Task 1
    (pt,), _, _ = task1.build(lab, [te], allm, meta)
    t1 = pd.DataFrame({"message_id": te.message_id, "trust_label": task1.decide(pt)})

    # ---- Account typing / Task 2
    A = account_types(lab)
    types = A.typ.reindex(accounts).fillna("N")
    sw = detect_switchers(lab)
    sw_test = detect_test_switchers(te, pt, types)
    print(f"types: {types.value_counts().to_dict()}  labelled switchers: {list(sw.index)}  "
          f"test switchers: {list(sw_test.index)}")

    comp_ts = {}
    host_types = {"all": ("M", "MIX"), "m_only": ("M",), "ato_only": ()}[variant]
    for a in A.index[A.typ.isin(host_types)]:
        comp_ts[a] = A.first_bad_ts[a]
    for s in (sw, sw_test):
        for a, ts in s.items():
            comp_ts.setdefault(a, ts)
    # Messages of takeover accounts after the takeover follow M-type behaviour.
    for a, ts in pd.concat([sw, sw_test]).items():
        m = (te.sender_id == a) & (te.ts >= ts)
        t1.loc[m.values, "trust_label"] = "MALICIOUS"

    t2 = pd.DataFrame({"account_id": accounts})
    t2["compromised"] = t2.account_id.isin(comp_ts).astype(int)
    t2["compromise_timestamp"] = t2.account_id.map(
        lambda a: pd.Timestamp(comp_ts[a]).strftime(TS_FMT) if a in comp_ts else "")

    # ---- Task 3
    camp = campaign_clusters(allm, types, list(sw.index) + list(sw_test.index))
    def cid(a):
        if a in camp:
            return f"CAM_CLUSTER_{camp[a]:03d}"
        return {"MIX": "CAM_CLUSTER_MIX", "S": "CAM_CLUSTER_SUS"}.get(types[a], "CAM_CLUSTER_BENIGN")
    t3 = pd.DataFrame({"account_id": accounts, "cluster_id": accounts.map(cid)})

    t1.to_csv(f"{out}/task1_predictions.csv", index=False)
    t2.to_csv(f"{out}/task2_predictions.csv", index=False)
    t3.to_csv(f"{out}/task3_predictions.csv", index=False)

    # Combined file in the sample_submission layout.
    msg = t1.rename(columns={"message_id": "id"}).assign(
        compromised=-1, compromise_timestamp="NONE", cluster_id="NONE")
    acc = t2.rename(columns={"account_id": "id"}).merge(t3.rename(columns={"account_id": "id"}))
    acc["trust_label"] = "NONE"
    acc["compromise_timestamp"] = acc.compromise_timestamp.replace("", "NONE")
    cols = ["id", "trust_label", "compromised", "compromise_timestamp", "cluster_id"]
    pd.concat([msg[cols], acc[cols]]).to_csv(f"{out}/submission.csv", index=False)

    print(t1.trust_label.value_counts().to_dict())
    print("compromised:", int(t2.compromised.sum()),
          " campaign clusters:", t3.cluster_id.str.match(r"CAM_CLUSTER_\d").sum(),
          "accounts in", t3.cluster_id[t3.cluster_id.str.match(r"CAM_CLUSTER_\d")].nunique(), "cells")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/upload_structure")
    ap.add_argument("--out", default="submission")
    ap.add_argument("--variant", default="all", choices=["all", "m_only", "ato_only"],
                    help="which hostile account types count as compromised in task 2")
    a = ap.parse_args()
    main(a.data, a.out, a.variant)
