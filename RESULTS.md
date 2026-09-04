# Results

This document is written by `scripts/write_results.py` from the saved training reports, so every number here is what the code produced on the run currently in `artifacts/`. Editing it by hand would only make it wrong at the next training run.

## What was held out

The catalogue holds 31 storms. 28 are used for training and validation, and 3 are held out completely:

- **Gannon storm** (2024-05-08 to 2024-05-14), minimum SYM/H -518 nT. The largest storm in more than twenty years. Held out entirely, so every number reported for it is a genuine forecast.
- **Halloween storm** (2003-10-28 to 2003-11-03), minimum SYM/H -432 nT. The reference extreme event. Held out as a second independent test.
- **October 2024 storm** (2024-10-09 to 2024-10-13), minimum SYM/H -390 nT. The second largest event of cycle 25 so far. Held out as a third independent test, and the only test event that is not also one of the two largest storms in the catalogue.

The split is by whole storm, so no minute in the test set sits next to a minute the model trained on. The scaler, the alarm threshold, and the calibration map are all fitted without touching the test storms.

## The model

- 19,000 parameters, 24 channels, 315 minutes of history reaching the output
- trained in 24 minutes on a laptop with no graphics card, using NumPy alone
- best epoch 1, selected on a validation split made of whole storms
- dropout 0.25, weight decay 0.0001, input jitter 0.05 standard deviations

## Forecast skill at 0.1 nT per second

This is the alerting level where an Indian low latitude observatory starts to show a disturbance worth acting on. Scores are on the held out storms, with the alarm threshold chosen on validation and applied unchanged.

| Horizon | Probability of detection | False alarm ratio | Heidke skill score | Peirce score | Brier skill score |
| --- | --- | --- | --- | --- | --- |
| 30 minutes | 0.83 | 0.37 | 0.67 | 0.76 | 0.34 |
| 45 minutes | 0.78 | 0.38 | 0.63 | 0.70 | 0.28 |
| 60 minutes | 0.78 | 0.43 | 0.60 | 0.69 | 0.28 |
| 90 minutes | 0.71 | 0.44 | 0.57 | 0.63 | 0.28 |

### What the clock is worth

The same model, the same storms, and the same settings, trained once on the clock of the spacecraft that measured the solar wind and once on the archive convention that shifts it forward to the Earth. The second one has no warning time to forecast into, because a shock appears in its input at the moment it strikes.

| Horizon | spacecraft clock HSS | bow shock clock HSS | Difference |
| --- | --- | --- | --- |
| 30 minutes | 0.67 | 0.63 | +0.04 |
| 45 minutes | 0.63 | 0.61 | +0.03 |
| 60 minutes | 0.60 | 0.59 | +0.01 |
| 90 minutes | 0.57 | 0.54 | +0.03 |

## At the higher 0.3 nT per second level

| Horizon | Probability of detection | False alarm ratio | Heidke skill score | Peirce score | Brier skill score |
| --- | --- | --- | --- | --- | --- |
| 30 minutes | 0.75 | 0.85 | 0.23 | 0.65 | 0.05 |
| 45 minutes | 0.36 | 0.72 | 0.29 | 0.34 | 0.02 |
| 60 minutes | 0.66 | 0.85 | 0.21 | 0.57 | 0.03 |
| 90 minutes | 0.40 | 0.78 | 0.26 | 0.37 | 0.03 |

## Calibration

A stated ninetieth percentile should sit above the observation ninety percent of the time. The decision layer treats these numbers as real probabilities and would be misled by overconfident ones.

| Nominal level | Observed before calibration | Observed after |
| --- | --- | --- |
| 0.10 | 0.398 | 0.299 |
| 0.25 | 0.519 | 0.337 |
| 0.50 | 0.661 | 0.546 |
| 0.75 | 0.782 | 0.730 |
| 0.90 | 0.885 | 0.860 |
| 0.98 | 0.960 | 0.966 |

Mean absolute coverage error: 0.132 before calibration, 0.067 after.

## Reproducing this

```bash
pip install -r requirements.txt
python -m setu.cli train --time-base l1 --tag l1
python -m setu.cli train --time-base bowshock --tag bowshock
python scripts/write_results.py
python scripts/make_figures.py
```

The data is downloaded on first use and cached under `data/raw`, so the first run is slow and later ones are not.
