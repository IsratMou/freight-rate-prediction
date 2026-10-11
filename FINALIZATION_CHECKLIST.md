# Submission handoff — selected same-date-feature policy (2026-10-09)

## Selected inference policy and reconciliation

The candidate explicitly selected **same-date supplied-feature median** for missing inference-time market signals. Non-null per-load inputs are preserved. For the 249 `validation.csv` rows whose `market_index` is null, use that date's median `market_index` from *supplied validation feature rows*. All 31 December scenario dates get missing `market_index` and `quote_signal` from the same-date validation features. This uses **no validation `posted_rate` or other targets**, but assumes the full same-date feature batch is available when producing predictions. If that batch is unavailable, the model's stored training medians are the documented fallback, **not** the submitted policy. Training-partition missing values remain imputed using training medians. No validation row is dropped.

The user's Colab `validation_predictions.csv` had SHA-256 **`b8085412cf18441c36257fa8fefb0f7785c88fd71ce056a9028a323cde3a0b0d`**. Applying this same-date policy in an independent run reproduced the *entire file byte-for-byte*. The previous repository version used training-only medians for these 249 rows (SHA-256 `62e941…`); this has been corrected in the final repo/zip. The 31 December predictions and official chart remain unchanged. The notebook itself is not available in this workspace; the exact matching file hash was provided by the user (not downloaded here).

## Locally verified ✅

- Fresh isolated-copy rerun in `/tmp/freight_fresh_same20261009/` after clearing all generated outputs: trained on raw CSVs, produced both outputs, ran supplied scorer, reference `validate.py`, selected-policy `validation_signal_sensitivity.py`, December sensitivity, optional model-family audit, and DOCX builder. The two output CSVs, scorer chart, holdout/rolling/policy/family/sensitivity CSVs were **byte-identical** to the repo copies. The raw December input was unchanged.
- Holdout Jan–Aug → Sep–Oct: **selected same-date policy** clean n=9,379 MAE **$48.13**, MAPE **2.07%**, R² **0.99716**; ALL n=9,523 MAE **$104.08**, R² **0.82813**. Same fit, training-median reference clean $48.30 / ALL $104.27. Exactly 73 held-out rows lacked market_index. `reports/validation_signal_sensitivity.csv` reproduces both with the same evaluation mask. The older `validate.py`/model-family audit/rolling evidence is clearly labeled as the **reference training-median protocol**, not passed off as the selected-policy metric.
- Final validation output: 12,000/12,000, order from the supplied template, mean **$2,394.03**, range $205.66–$6,855.23, 0 null rates, same SHA-256 as the user's current Colab file.
- Final December output: 31/31, range $824.42–$854.51, mean $839.05. Scenario sensitivity: training-only median fallback mean $853.91; peak Dec 31 instead of submitted Dec 25. The fallback is **not** the submission file.
- Exact duplicate rows = 0; 47,933 repeated pickup–delivery lane occurrences are not duplicated rows. The chronological split reflects future-month deployment, not a duplication/leakage claim.
- Supplied scorer validated both CSVs and created the official chart. This validates submission structure/positive rates and image generation, **not hidden-target accuracy**.
- Packaged deliverables: `outputs/validation_predictions.csv`, `outputs/december-chart-inputs.csv`, `reports/candidate_december.png`, `reports/freight_rate_report.docx` (79 paragraphs, 8 tables, 6 images), README and source scripts. `intcells/cell10.py`, `cell10_en.py` and the 4-page visualization cell in the accompanying workspace were updated; the updated English/Bengali Cell 10 scripts also generated the user's exact validation hash in isolated local runs.

## Reproduce in a clean checkout

Put the four kit CSVs (see `data/README.md`) into `data/`. From repo root:

```bash
pip install -r requirements.txt
python freight_rate_model.py --train data/train-test.csv --validation data/validation.csv \
  --template data/validation-predictions-template.csv \
  --december data/december-chart-inputs.csv \
  --out outputs/validation_predictions.csv
python score.py --predictions outputs/validation_predictions.csv \
  --december-predictions outputs/december-chart-inputs.csv
python validate.py                          # reference-policy holdout/rolling
python validation_signal_sensitivity.py    # selected-vs-reference holdout
python december_signal_sensitivity.py      # submitted-vs-fallback scenario
pip install 'scikit-learn>=1.4,<2'         # optional family-audit dependency
python model_family_audit.py data/train-test.csv  # reference-policy comparison
python make_report.py
```

The prediction CLI writes **both output CSVs to `outputs/`**; it does not modify the raw inputs in `data/`.

## Still pending (user-side / external)

- The user's *current saved* Colab validation CSV has the selected-policy hash. The **updated Cell 10 source code** has been local-run but has **not yet been rerun by the user in Colab**; replace the old cell before future notebook reruns, otherwise it will overwrite the saved file with training-median predictions. The Colab December file bytes were not provided for a direct hash comparison, though the user's stdout and local reproductions agree.
- Push the new repository/zip to GitHub and verify that the URL is accessible. Record the 2–3 minute Loom (`../loom_script.md` in workspace), add an accessible Loom link to the README, and attach both output CSVs + report if the submission form requires separate uploads.
- Hidden validation targets/accuracy remain unknown until the evaluator scores the submission. A scorer pass does not establish accuracy.

## Local submission file SHA-256

| File | SHA-256 |
|---|---|
| `outputs/validation_predictions.csv` | `b8085412cf18441c36257fa8fefb0f7785c88fd71ce056a9028a323cde3a0b0d` |
| `outputs/december-chart-inputs.csv` | `991fb285e44544a90b8e9956c19d54f6c678a4a239e3777a9f674635de6661e8` |
| `reports/candidate_december.png` | `0d9c49ab1f11f2950a9d94b48d85767d76d25132742c11a1a9d0856d91c26e8a` |
