#!/usr/bin/env python3
"""Compare December chart predictions under two market-signal availability assumptions.

Run from the repository root, after placing the assessment CSVs under data/:
  python december_signal_sensitivity.py

No labels from validation.csv are accessed. This is a sensitivity check, not a second
submission model. The submitted December CSV uses the same-date feature medians;
training-median rates are saved ONLY to reports/ for transparency.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from freight_rate_model import fit, predict


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default="data/train-test.csv")
    ap.add_argument("--validation", default="data/validation.csv")
    ap.add_argument("--december", default="data/december-chart-inputs.csv")
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train, val, dec = (pd.read_csv(p) for p in (args.train, args.validation, args.december))
    model = fit(train)
    # The scenario file has no market_index / quote_signal. Passing same_period=val
    # uses each DATE's median from the supplied validation features (NOT targets).
    per_date = val.assign(_date=pd.to_datetime(val.date)).groupby("_date")[
        ["market_index", "quote_signal"]].median()
    dates = pd.to_datetime(dec.date)
    same = np.round(predict(model, dec, same_period=val), 2)
    # When neither market signal exists, the production model itself substitutes
    # market_med and quote_med learned on the labeled training partition only.
    fallback = np.round(predict(model, dec, same_period=None), 2)
    t = pd.DataFrame({
        "date": dates.dt.strftime("%Y-%m-%d"),
        "same_date_market_index": dates.map(per_date.market_index),
        "same_date_quote_signal": dates.map(per_date.quote_signal),
        "same_period_rate": same,
        "training_median_rate": fallback,
        "delta_dollars": np.round(same - fallback, 2),
        "delta_pct_of_fallback": np.round(100 * (same / fallback - 1), 3),
    })
    csv = out / "december_signal_sensitivity.csv"
    t.to_csv(csv, index=False)
    fig, ax = plt.subplots(figsize=(8.6, 3.3), dpi=155)
    ax.plot(dates, same, color="#2563eb", lw=2.2, marker="o", markersize=2.7,
            label="Submitted: same-date validation-feature medians")
    ax.plot(dates, fallback, color="#d97706", lw=1.9, linestyle="--",
            label="Sensitivity: training-only signal medians")
    ax.set(xlabel="December 2025", ylabel="Predicted rate ($)",
           title="December chart: sensitivity to market-signal availability")
    ax.legend(frameon=False, loc="best", fontsize=8)
    ax.grid(alpha=.2)
    fig.autofmt_xdate(rotation=0)
    fig.tight_layout()
    png = out / "december_signal_sensitivity.png"
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    print(f"signal training medians: market={model['market_med']:.5f}, quote={model['quote_med']:.5f}")
    print(f"same-date coverage: market {t.same_date_market_index.notna().sum()}/{len(t)}, "
          f"quote {t.same_date_quote_signal.notna().sum()}/{len(t)}")
    for name, p in [("submitted same-date", same), ("training-median fallback", fallback)]:
        print(f"{name}: n={len(p)} mean=${p.mean():.2f} min=${p.min():.2f} "
              f"max=${p.max():.2f} peak={t.date.iloc[np.argmax(p)]}")
    print(f"same-date minus fallback: mean=${(same-fallback).mean():+.2f}, "
          f"mean abs=${np.abs(same-fallback).mean():.2f}, max abs=${np.abs(same-fallback).max():.2f}")
    print(f"wrote {csv} and {png}; submitted outputs NOT modified")


if __name__ == "__main__":
    main()
