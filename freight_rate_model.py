#!/usr/bin/env python3
"""
Freight Rate Prediction — final solution.

Model
-----
    log(posted_rate) = b0
                     + b1 * log(distance) + b2 * log(distance)^2
                     + b3 * market_index
                     + b4 * quote_signal
                     + b5 * log(|weight|)
                     + b6 * is_reefer + b7 * is_flatbed
                     + b8 * t                      (t = days since 2025-01-01)
                     + pickup-city fixed effects
                     + delivery-city fixed effects
                     + eps,   eps ~ N(0, ~2.2%)

Why this model (full reasoning in REPORT.md):
  1. Freight rates are multiplicative: rate ~ distance^a * equipment * market * ...
     -> model log(rate), which also removes the skew of the raw target.
  2. `distance` is by far the dominant driver. The log-log relation is slightly
     convex, so a log(distance)^2 term is included.
  3. market_index and weight enter multiplicatively -> log(target) is the right space.
  4. Pickup/delivery city fixed effects are real, large (up to +/-6%) and stable
     over time (cross-period correlation 0.99), so they generalise forward.
  5. A slow linear drift in rates exists on top of market_index. A plain linear
     time trend extrapolates this drift better than the alternatives
     (no trend / last-level offset / recent-window trend) in rolling-origin tests.
  6. The target contains ~1.4% corrupted rows (rate multiplied by a random factor
     ~0.15x-5x). They are unpredictable from features (CV-AUC 0.49), so they are
     detected via a robust MAD threshold and removed from training; leaving them in
     biases every coefficient.

Fit with plain numpy (lstsq) so the solution needs only pandas + numpy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

T0 = pd.Timestamp("2025-01-01")
CONT_FEATS = ["ld", "ld2", "market", "quote", "log_w", "is_reefer", "is_flatbed", "t"]
RECENT_DAYS = 60  # ignored; kept for clarity of the trend specification


# ----------------------------------------------------------------------------- helpers
def _base_frame(d: pd.DataFrame, market_med: float, weight_med: float, with_time: bool,
                quote_med: float | None = None) -> pd.DataFrame:
    """Feature block that does not depend on the city vocabulary.

    market_index / quote_signal may be absent (e.g. the December chart file carries
    only date + load attributes); missing columns and missing values fall back to the
    training medians stored in the model.
    """
    f = pd.DataFrame(index=d.index)
    f["ld"] = np.log(d["distance"].astype(float))
    f["ld2"] = f["ld"] ** 2
    if "market_index" in d.columns:
        f["market"] = pd.to_numeric(d["market_index"], errors="coerce").fillna(market_med)
    else:
        f["market"] = market_med
    if "quote_signal" in d.columns:
        f["quote"] = pd.to_numeric(d["quote_signal"], errors="coerce").fillna(
            quote_med if quote_med is not None else d["quote_signal"].median())
    else:
        f["quote"] = quote_med if quote_med is not None else 0.0
    f["log_w"] = np.log(pd.to_numeric(d["weight"], errors="coerce").abs().fillna(weight_med))
    f["is_reefer"] = (d["equipment"] == "Reefer").astype(float)
    f["is_flatbed"] = (d["equipment"] == "Flatbed").astype(float)
    if with_time:
        f["t"] = (pd.to_datetime(d["date"]) - T0).dt.days.astype(float)
    return f


def _fill_signals_from(d: pd.DataFrame, source: pd.DataFrame | None) -> pd.DataFrame:
    """Fill absent OR null market/quote signals from same-date supplied features.

    source is a feature-only batch (validation.csv for production, or the unlabeled
    test features for a policy-sensitivity backtest). A row's non-null signals are
    preserved. An unmatched date/all-null day falls back to medians in _base_frame.
    No target column is read or used by this helper.
    """
    if source is None:
        return d
    out = d.copy()
    src = source.copy()
    src["_d"] = pd.to_datetime(src["date"])
    dd = pd.to_datetime(out["date"])
    for col in ("market_index", "quote_signal"):
        daily = dd.map(src.groupby("_d")[col].median())
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(daily)
        else:
            out[col] = daily
    return out


def _city_block(d: pd.DataFrame, cities: list[str], prefix: str) -> np.ndarray:
    """One-hot block for a city column (baseline = cities[0], absorbed by intercept).

    Three cases:
      * baseline city       -> all zeros (by design; absorbed by the intercept)
      * any other seen city -> its dummy = 1
      * genuinely unseen city (not in the training vocabulary at all) -> the block's
        prior mean: every dummy column gets 1/K, i.e. the city's contribution is the
        mean fitted city effect (≈ +3.2% here). Validated by leave-one-city-out:
        $66.7 vs $78.2 MAE against the zeros/baseline fallback (oracle $46.6).
    """
    K = len(cities) - 1
    idx = {c: i for i, c in enumerate(cities)}
    fill = 1.0 / K
    out = np.zeros((len(d), K))
    for row, city in enumerate(d.values):
        j = idx.get(city)
        if j is None:          # truly unseen city
            out[row, :] = fill
        elif j > 0:            # seen, non-baseline city
            out[row, j - 1] = 1.0
        # j == 0 -> baseline city, zeros by design
    return out


def _design(d: pd.DataFrame, model: dict) -> np.ndarray:
    f = _base_frame(d, model["market_med"], model["weight_med"], with_time=True,
                    quote_med=model.get("quote_med"))
    parts = [np.ones((len(d), 1)), f[CONT_FEATS].to_numpy(dtype=float)]
    parts.append(_city_block(d["pickup"], model["cities"], "po"))
    parts.append(_city_block(d["delivery"], model["cities"], "do"))
    X = np.hstack(parts)
    if "coef" in model and X.shape[1] != len(model["coef"]):
        raise ValueError(f"design width {X.shape[1]} != coefficients {len(model['coef'])}")
    return X


def _fit_lstsq(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


# ----------------------------------------------------------------------------- api
def fit(train: pd.DataFrame) -> dict:
    """Fit the model and return a self-contained model dict."""
    d = train.copy()
    d["weight"] = pd.to_numeric(d["weight"], errors="coerce").abs()
    cities = sorted(d["pickup"].unique().tolist())
    market_med = float(pd.to_numeric(d["market_index"], errors="coerce").median())
    quote_med = float(pd.to_numeric(d["quote_signal"], errors="coerce").median())
    weight_med = float(d["weight"].median())

    model = {"cities": cities, "market_med": market_med, "quote_med": quote_med,
             "weight_med": weight_med}

    # ---- 1. robust cleaning of the corrupted targets -------------------------
    Xc = np.hstack([np.ones((len(d), 1)),
                    _base_frame(d, market_med, weight_med, with_time=False)[CONT_FEATS[:-1]].to_numpy(float)])
    y = np.log(d["posted_rate"].astype(float).to_numpy())
    w0 = _fit_lstsq(Xc, y)
    resid = y - Xc @ w0
    sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))   # MAD-based robust sigma
    keep = np.abs(resid - np.median(resid)) < 6.0 * sigma
    model["sigma"] = float(sigma)
    model["n_train"] = int(len(d))
    model["n_clean"] = int(keep.sum())
    model["n_flagged"] = int((~keep).sum())

    # ---- 2. full model on the clean rows ------------------------------------
    d_clean = d[keep]
    X = _design(d_clean, model)
    model["coef"] = _fit_lstsq(X, y[keep])
    fitted = X @ model["coef"]
    model["resid_std"] = float(np.std(y[keep] - fitted))
    model["r2"] = float(1 - np.sum((y[keep] - fitted) ** 2) / np.sum((y[keep] - y[keep].mean()) ** 2))
    return model


def predict(model: dict, d: pd.DataFrame, same_period: pd.DataFrame | None = None) -> np.ndarray:
    """Predict posted_rate for new loads.

    `same_period` (optional): feature-only rows available in the same-date batch
    (e.g. the supplied validation file). Missing columns AND null signal values are
    imputed with daily medians of that batch; unmatched dates fall back to stored
    training medians. This is an explicit inference-time availability assumption;
    `posted_rate`/targets are never consulted.
    """
    d = _fill_signals_from(d.copy(), same_period)
    d["weight"] = pd.to_numeric(d["weight"], errors="coerce")
    log_pred = _design(d, model) @ model["coef"]
    rate = np.exp(log_pred)
    # Output-validity guard only. Measured raw prediction range on train + holdout is
    # $186–$7,092 (target range $57–$25.5k), so these bounds never bind on any observed
    # row; they exist purely to protect the scorer output against pathological inputs.
    lo, hi = 25.0, 40_000.0
    return np.clip(rate, lo, hi)


def coefficients_table(model: dict, top_cities: int = 8) -> pd.DataFrame:
    names = ["intercept"] + CONT_FEATS + [f"pickup::{c}" for c in model["cities"][1:]] + \
            [f"delivery::{c}" for c in model["cities"][1:]]
    coef = pd.DataFrame({"term": names, "coef": model["coef"]})
    coef["effect_%"] = (np.exp(coef["coef"]) - 1) * 100
    return coef


# ----------------------------------------------------------------------------- cli
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Train on train_test.csv and predict the required files.")
    ap.add_argument("--train", default="data/train_test.csv")
    ap.add_argument("--validation", default="data/validation.csv")
    ap.add_argument("--template", default="data/validation_predictions_template.csv")
    ap.add_argument("--december", default="data/december_chart_inputs.csv")
    ap.add_argument("--out", default="validation_predictions.csv")
    args = ap.parse_args()

    tr = pd.read_csv(args.train)
    model = fit(tr)
    print(f"trained on {model['n_clean']:,} clean rows (flagged {model['n_flagged']} corrupted, "
          f"{100*model['n_flagged']/model['n_train']:.2f}%) | R2={model['r2']:.5f} | resid_std={model['resid_std']:.4f}")

    import os
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    val = None
    if os.path.exists(args.validation):
        val = pd.read_csv(args.validation)
        # Batch inference: fill the 249 null validation market signals from supplied
        # features on the same date (if available); never from validation targets.
        pred = predict(model, val, same_period=val)
        out = pd.DataFrame({"load_id": val["load_id"], "predicted_rate": np.round(pred, 2)})
        # keep the template's row order when available
        if os.path.exists(args.template):
            tmpl = pd.read_csv(args.template)
            out = tmpl[["load_id"]].merge(out, on="load_id", how="left")
            assert out["predicted_rate"].notna().all(), "some template load_ids missing from validation.csv"
        out.to_csv(args.out, index=False)
        print(f"wrote {args.out}: {len(out):,} rows | mean ${out.predicted_rate.mean():,.2f}")
    else:
        print(f"[skip] {args.validation} not found")

    if os.path.exists(args.december):
        dec = pd.read_csv(args.december)
        # the December file has no market_index/quote_signal -> same-period lookup from validation
        dec["predicted_rate"] = np.round(predict(model, dec, same_period=val), 2)
        # Output beside --out; never overwrite the raw assessment input in data/.
        dec_out = os.path.join(out_dir, "december-chart-inputs.csv")
        if os.path.abspath(dec_out) == os.path.abspath(args.december):
            raise ValueError("December input and output paths must differ to preserve the raw input")
        dec.to_csv(dec_out, index=False)
        print(f"wrote {dec_out}: {len(dec):,} rows | mean ${dec.predicted_rate.mean():,.2f}")
    else:
        print(f"[skip] {args.december} not found")
