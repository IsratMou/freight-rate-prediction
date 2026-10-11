#!/usr/bin/env python3
"""Reproduce the FIXED-mask model-family comparison from Colab cell9_en.py, section A.

    python model_family_audit.py data/train-test.csv

OPTIONAL audit dependency: scikit-learn (the final solution does not use it).
The no-cleaning baselines here test families with the SAME clean-row mask for
comparison; the shipped training-only robust filter is evaluated separately in
validate.py. The mask is used for reporting, never for training a model on the
holdout. Both clean and ALL-row metrics are emitted for every family.
"""
# Extracted, unmodified comparison logic from the audit notebook's Cell 9, section A.
# Includes no rolling-fold or production refit stage.
import os, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

pd.set_option("display.width", 240)
pd.set_option("display.max_rows", 90)

import sys
train_path = sys.argv[1] if len(sys.argv) > 1 else "data/train-test.csv"
df = pd.read_csv(train_path)
df["date"] = pd.to_datetime(df["date"])

T0 = pd.Timestamp("2025-01-01")
CITIES = sorted(df.pickup.unique().tolist())


def med_of(d):
    return {"mkt": float(d.market_index.median()),
            "w":   float(d.weight.abs().median()),
            "q":   float(d.quote_signal.median())}


def design9(d, med, with_time=True):
    ld = np.log(d.distance.values.astype(float))
    cols = [np.ones((len(d), 1)), ld.reshape(-1, 1), (ld ** 2).reshape(-1, 1),
            d.market_index.fillna(med["mkt"]).values.astype(float).reshape(-1, 1),
            d.quote_signal.fillna(med["q"]).values.astype(float).reshape(-1, 1),
            np.log(d.weight.abs().fillna(med["w"]).values.astype(float)).reshape(-1, 1),
            (d.equipment == "Reefer").values.astype(float).reshape(-1, 1),
            (d.equipment == "Flatbed").values.astype(float).reshape(-1, 1)]
    for col in [d.pickup, d.delivery]:
        idx = {c: i for i, c in enumerate(CITIES[1:])}
        M = np.zeros((len(d), len(CITIES) - 1))
        for r, v in enumerate(col.values):
            j = idx.get(v)
            if j is not None:
                M[r, j] = 1.0
        cols.append(M)
    if with_time:
        cols.append((d.date - T0).dt.days.values.astype(float).reshape(-1, 1))
    return np.column_stack(cols)


def fit9(d, with_time=True):
    med = med_of(d)
    b, *_ = np.linalg.lstsq(design9(d, med, with_time), np.log(d.posted_rate.values.astype(float)), rcond=None)
    return b, med


def p9(b, med, d, with_time=True):
    return np.exp(design9(d, med, with_time) @ b)


def metrics9(y, pred, cm):
    """cm = fixed clean-row mask (model-independent) — all models are measured on the same rows"""
    out = {}
    for lbl, m in [("CLEAN", cm), ("ALL", np.ones(len(y), bool))]:
        e = np.abs(y[m] - pred[m])
        out[f"MAE_{lbl}"] = e.mean()
        out[f"MAPE_{lbl}"] = (e / y[m]).mean() * 100
        out[f"R2_{lbl}"] = 1 - ((y[m] - pred[m]) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()
    out["n_clean"] = int(cm.sum())
    return out


def core_mask_of(d):
    """model-independent clean mask: |log resid| of the distance-only fit < 0.4"""
    ld = np.log(d.distance.values.astype(float))
    X = np.column_stack([np.ones(len(d)), ld])
    y = np.log(d.posted_rate.values.astype(float))
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return np.abs(y - X @ b) < 0.4


# ================================================================
print("=" * 118)
print("A. MODEL FAMILY — Sep–Oct holdout (train Jan–Aug), no cleaning")
print("=" * 118)
tr = df[df.date < "2025-09-01"]; te = df[df.date >= "2025-09-01"]
y_tr_log = np.log(tr.posted_rate.values.astype(float))
y_te = te.posted_rate.values.astype(float)
med_tr = med_of(tr)
df["_core"] = core_mask_of(df)
core_te = df.loc[df.date >= "2025-09-01", "_core"].values
print(f"train {len(tr):,} rows | test {len(te):,} rows | fixed clean mask (same for all models): "
      f"{core_te.sum():,} / {len(te):,}\n")
rows = []


def add_row(name, pred):
    m = metrics9(y_te, pred, core_te)
    rows.append({"model": name,
                 "MAE_CLEAN": round(m["MAE_CLEAN"], 2), "MAPE_CLEAN": round(m["MAPE_CLEAN"], 2),
                 "R2_CLEAN": round(m["R2_CLEAN"], 5),
                 "MAE_ALL": round(m["MAE_ALL"], 2), "R2_ALL": round(m["R2_ALL"], 5)})


add_row("mean baseline", np.full(len(te), float(np.exp(y_tr_log.mean()))))

X = np.column_stack([np.ones(len(tr)), np.log(tr.distance.values.astype(float))])
b, *_ = np.linalg.lstsq(X, y_tr_log, rcond=None)
add_row("distance-only (log-log)",
        np.exp(np.column_stack([np.ones(len(te)), np.log(te.distance.values.astype(float))]) @ b))

b, _ = fit9(tr, with_time=False); add_row("linear (no time)", p9(b, med_tr, te, False))
b, _ = fit9(tr, with_time=True);  add_row("linear (+ time)", p9(b, med_tr, te, True))

for with_t in [False, True]:
    tic = time.time()
    g = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=63,
                                      early_stopping=False, random_state=0)
    g.fit(design9(tr, med_tr, with_t), y_tr_log)
    add_row("GBM 300it/63leaf (no time)" if not with_t else "GBM 300it/63leaf (+ time)",
            np.exp(g.predict(design9(te, med_tr, with_t))))
    print(f"  [GBM fit ({'no time' if not with_t else '+ time'}) : {time.time()-tic:5.1f}s]")

comparison = pd.DataFrame(rows)
print(comparison.to_string(index=False))
os.makedirs("reports", exist_ok=True)
comparison.to_csv("reports/model_family_audit.csv", index=False)
print("wrote reports/model_family_audit.csv")

