"""Account-level hostile type inference from labelled history."""
import numpy as np
import pandas as pd

from features import template_group, template_maps


def account_types(lab):
    """N (clean), S (suspicious-only), MIX (template-dependent), M (malicious)."""
    maps = template_maps(lab)
    g, _, _ = template_group(lab, maps)
    lab = lab.assign(grp=g.values, bad=lab.trust_label != "TRUSTWORTHY", mal=lab.trust_label == "MALICIOUS")
    A = lab.groupby("sender_id").agg(n=("bad", "size"), bad=("bad", "mean"), mal=("mal", "mean"),
                                     first_ts=("ts", "min"))
    A["benign_mal"] = lab[lab.grp != "cov"].groupby("sender_id").mal.mean()
    A["soc_mal"] = lab[lab.grp == "soc"].groupby("sender_id").mal.mean()
    A["dir_mal"] = lab[lab.grp == "dir"].groupby("sender_id").mal.mean()
    typ = np.where(A.bad < 0.6, "N",
                   np.where(A.dir_mal.fillna(A.benign_mal) < 0.5, "S",
                            np.where(A.soc_mal.fillna(1) > 0.5, "M", "MIX")))
    A["typ"] = typ
    first_mal = lab[lab.mal].groupby("sender_id").ts.min()
    first_bad = lab[lab.bad].groupby("sender_id").ts.min()
    A["first_mal_ts"], A["first_bad_ts"] = first_mal, first_bad
    return A
