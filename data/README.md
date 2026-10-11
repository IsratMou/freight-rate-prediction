# data/

This folder is where the assessment CSV files go. They are **not committed** (see
`.gitignore`) — download them from the assessment package and place them here with
these names:

| file from the assessment package        | expected name here                    |
|-----------------------------------------|---------------------------------------|
| `train-test.csv`                        | `data/train-test.csv`                 |
| `validation.csv`                        | `data/validation.csv`                 |
| `validation-predictions-template.csv`   | `data/validation-predictions-template.csv` |
| `december-chart-inputs.csv`             | `data/december-chart-inputs.csv`      |

Then run, from the repository root:

```bash
python freight_rate_model.py \
    --train       data/train-test.csv \
    --validation  data/validation.csv \
    --template    data/validation-predictions-template.csv \
    --december    data/december-chart-inputs.csv \
    --out         outputs/validation_predictions.csv
```

Note: the CLI defaults use underscore names (`data/train_test.csv`, …); the flags above
match the assessment package's hyphenated names, so pass them explicitly (or rename the
files).

`score.py` also reads the completed December file in place — pass the copy in
`outputs/` so `data/` stays pristine.
