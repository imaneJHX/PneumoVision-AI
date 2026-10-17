# Model artifacts

The local training pipeline writes the Keras model to:

```text
models/pneumonia_cnn.keras
```

Optional real evaluation data can be placed at:

```text
models/evaluation_data.csv
```

The CSV must contain `y_true` (`0` for NORMAL, `1` for PNEUMONIA) and `y_score`
(the corresponding raw pneumonia sigmoid output) for every evaluated case. Model and
evaluation artifacts are intentionally excluded from Git. Generate them locally with
`python -m training.train` and `python -m training.evaluate`.

Improvement candidates are deliberately stored under `training/runs/experiments/`, not
in this directory. Reviewing a candidate never changes the production artifact. Promotion
to `models/pneumonia_cnn.keras` requires an explicit decision after validation comparison
and final untouched-test evaluation.
