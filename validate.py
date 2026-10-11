#!/usr/bin/env python3
"""
Reproduces the validation reported in the write-up.

  * 2-month holdout: train Jan-Aug -> test Sep-Oct  (mirrors the real Nov+Dec horizon)
  * 6-fold rolling-origin evaluation
  * baseline comparisons
  * charts written to reports/

Run:  python validate.py
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from freight_rate_model import fit, predict, _design, _base_frame, CONT_FEATS, T0

OUT = "reports"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, GREY, AMBER = "#2563eb", "#dc2626", "#059669", "#94a3b8", "#d97706"


def metrics(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return {"MAE": np.mean(np.abs(y - yhat)),
            "RMSE": np.sqrt(np.mean((y - yhat) ** 2)),
            "MAPE": np.mean(np.abs((y - yhat) / y)) * 100,
            "R2": 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)}


def is_corrupt(y, yhat):
    """rows whose target is polluted by the multiplier corruption"""
    return np.abs(np.log(np.asarray(y, float)) - np.log(np.asarray(yhat, float))) > 0.4


def load():
    for path in ("data/train-test.csv", "data/train_test.csv"):   # assessment file is hyphenated
        if os.path.exists(path):
            return pd.read_csv(path, parse_dates=["date"])
    raise FileNotFoundError("expected data/train-test.csv (or data/train_test.csv)")


def main():
    df = load()
    print(f"data: {len(df):,} rows, {df.date.min().date()} -> {df.date.max().date()}\n")

    # ---------------- 1. headline holdout: train Jan-Aug -> test Sep-Oct ----------------
    print("=" * 96)
    print("1) TWO-MONTH HOLDOUT  (train Jan-Aug -> test Sep-Oct) — mirrors the Nov+Dec task")
    print("=" * 96)
    tr, te = df[df.date < "2025-09-01"], df[df.date >= "2025-09-01"]
    model = fit(tr)
    p = predict(model, te)
    bad = is_corrupt(te.posted_rate, p)
    rows = []
    for lbl, mask in [("all test rows", np.ones(len(te), bool)), ("clean test rows", ~bad)]:
        m = metrics(te.posted_rate.values[mask], p[mask])
        rows.append({"evaluation": lbl, "n": int(mask.sum()), **{k: round(v, 4) for k, v in m.items()}})
    # baselines
    yte = te.posted_rate.values
    dist_only = np.exp(np.polyval(np.polyfit(np.log(tr.distance), np.log(tr.posted_rate), 1),
                                  np.log(te.distance)))
    for lbl, pr in [("baseline: linear log-log on distance", dist_only),
                    ("baseline: predict the mean", np.full(len(te), tr.posted_rate.mean()))]:
        m = metrics(yte, pr)
        rows.append({"evaluation": lbl, "n": len(te), **{k: round(v, 4) for k, v in m.items()}})
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))
    tab.to_csv(f"{OUT}/holdout_metrics.csv", index=False)

    # ---------------- 2. rolling origin ----------------
    print("\n" + "=" * 96)
    print("2) ROLLING-ORIGIN VALIDATION (train up to the 1st, test the following month)")
    print("=" * 96)
    cuts = ["2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01", "2025-10-01"]
    res = []
    for c in cuts:
        a = pd.Timestamp(c)
        b = a + pd.DateOffset(months=1)
        tr_, te_ = df[df.date < a], df[(df.date >= a) & (df.date < b)]
        m_ = fit(tr_)
        p_ = predict(m_, te_)
        bad_ = is_corrupt(te_.posted_rate, p_)
        mm = metrics(te_.posted_rate.values[~bad_], p_[~bad_])
        res.append({"train ends": c[:10], "test month": str(b)[:7], "n_clean": int((~bad_).sum()),
                    **{k: round(v, 3) for k, v in mm.items()}})
    roll = pd.DataFrame(res)
    print(roll.to_string(index=False))
    roll.to_csv(f"{OUT}/rolling_origin.csv", index=False)

    # ---------------- 3. coefficients ----------------
    print("\n" + "=" * 96)
    print("3) FINAL MODEL COEFFICIENTS (trained on all cleaned data — what ships)")
    print("=" * 96)
    full = fit(df)
    names = ["intercept"] + CONT_FEATS + [f"pickup::{c}" for c in full["cities"][1:]] + \
            [f"delivery::{c}" for c in full["cities"][1:]]
    coef = pd.DataFrame({"term": names, "coef": full["coef"]})
    coef["effect_%"] = (np.exp(coef["coef"]) - 1) * 100
    coef.to_csv(f"{OUT}/coefficients.csv", index=False)
    print(coef.head(9).round(5).to_string(index=False))
    print("\nlargest city effects (% vs baseline city):")
    ce = coef[coef.term.str.contains("::")].copy()
    ce["city"] = ce.term.str.split("::").str[1]
    print(ce.reindex(ce["effect_%"].abs().nlargest(10).index)[["term", "effect_%"]].round(2).to_string(index=False))
    print(f"\ntrained on {full['n_clean']:,} clean rows | flagged {full['n_flagged']} corrupted "
          f"({100*full['n_flagged']/full['n_train']:.2f}%) | R2={full['r2']:.5f} | resid_std={full['resid_std']:.4f}")

    # ---------------- 4. charts ----------------
    print("\n" + "=" * 96)
    print("4) CHARTS -> reports/")
    print("=" * 96)

    # (a) predicted vs actual over time
    fit2 = fit(df[df.date < "2025-09-01"])
    p2 = predict(fit2, te)
    daily = pd.DataFrame({"date": te.date.values, "actual": te.posted_rate.values, "pred": p2})
    core = ~is_corrupt(te.posted_rate, p2)
    core = np.asarray(core)
    daily = daily[core]
    g = daily.groupby(daily.date.dt.date).mean()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(pd.to_datetime(g.index), g.actual, color=GREY, lw=1.1, label="actual (daily mean, clean rows)")
    ax.plot(pd.to_datetime(g.index), g.pred, color=BLUE, lw=2, label="model prediction")
    ax.set(ylabel="rate ($)")
    tr_daily = df[df.date < "2025-09-01"].groupby("date").posted_rate.mean()
    ax.plot(tr_daily.index, tr_daily.values, color=GREY, lw=1.1, alpha=.5)
    ax.axvline(pd.Timestamp("2025-09-01"), color=AMBER, ls="--", lw=1.2)
    ax.text(pd.Timestamp("2025-09-02"), ax.get_ylim()[1] * .97, "  hold-out starts", color=AMBER, fontsize=8, va="top")
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("Hold-out period: predicted vs actual daily mean rate", fontsize=10, loc="left", weight="bold")
    fig.savefig(f"{OUT}/holdout_series.png", bbox_inches="tight"); plt.close(fig)

    # (b) residual distribution on hold-out
    res_te = np.log(te.posted_rate.values[core]) - np.log(p2[core])
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.hist(res_te * 100, bins=70, color=BLUE, alpha=.85)
    ax.axvline(0, color=RED, lw=1.2)
    ax.set(xlabel="log error (%)", ylabel="loads")
    ax.set_title(f"Hold-out residuals: median {np.median(res_te)*100:+.2f}%, "
                 f"11-89 pct [{np.percentile(res_te,11)*100:+.1f}%, {np.percentile(res_te,89)*100:+.1f}%]",
                 fontsize=9, loc="left", weight="bold")
    fig.savefig(f"{OUT}/holdout_residuals.png", bbox_inches="tight"); plt.close(fig)

    # (c) predicted vs actual scatter
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(te.posted_rate.values[core], p2[core], s=4, alpha=.15, color=BLUE, linewidths=0)
    lim = [50, 12000]
    ax.plot(lim, lim, color=RED, lw=1)
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="actual ($)", ylabel="predicted ($)")
    ax.set_title("Hold-out, clean rows: predicted vs actual", fontsize=10, loc="left", weight="bold")
    fig.savefig(f"{OUT}/holdout_scatter.png", bbox_inches="tight"); plt.close(fig)

    # (d) the corruption we removed
    resid_full = np.log(df.posted_rate) - np.log(predict(fit(df[df.date < "2025-09-01"]), df))
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.hist(np.clip(resid_full, -3, 3) * 100, bins=160, color=GREY, alpha=.9)
    ax.axvline(0, color=RED, lw=1)
    ax.set(xlabel="log error (%), clipped at ±300%", ylabel="loads")
    ax.set_title(f"All 48,000 targets: a clean core + {(np.abs(resid_full)>0.4).sum()} corrupted rows ({(np.abs(resid_full)>0.4).mean()*100:.1f}%)",
                 fontsize=10, loc="left", weight="bold")
    fig.savefig(f"{OUT}/target_corruption.png", bbox_inches="tight"); plt.close(fig)

    print("  wrote holdout_series.png, holdout_residuals.png, holdout_scatter.png, target_corruption.png")
    print("\ndone.")


if __name__ == "__main__":
    main()
