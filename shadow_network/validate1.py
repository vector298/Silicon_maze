import itertools, sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from features import load
import task1
tr, va, te, meta = load("../data/upload_structure")
allm = pd.concat([tr, va, te], ignore_index=True)
(pv,), m, cols = task1.build(tr, [va], allm, meta)
print("argmax", f1_score(va.trust_label, task1.decide(pv), average="macro"))
best = max(((f1_score(va.trust_label, task1.decide(pv, (1, a, b)), average="macro"), a, b)
            for a in np.arange(0.6, 3.01, 0.1) for b in np.arange(0.6, 3.01, 0.1)))
print("best weights", best)
print(confusion_matrix(va.trust_label, task1.decide(pv, (1, best[1], best[2])), labels=task1.LABELS))
imp = pd.Series(m.feature_importance("gain"), index=cols).sort_values(ascending=False)
print(imp.head(25))
va["pred"] = task1.decide(pv)
err = va[va.pred != va.trust_label]
pd.set_option("display.width", 250)
print(err[["message_id", "sender_id", "timestamp", "trust_label", "pred", "tmpl_key"]].to_string())
