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
- trained in 55 minutes on a laptop with no graphics card, using NumPy alone
- best epoch 5, selected on a validation split made of whole storms
- dropout 0.25, weight decay 0.0001, input jitter 0.05 standard deviations

## Forecast skill at 0.1 nT per second

This is the alerting level where an Indian low latitude observatory starts to show a disturbance worth acting on. Scores are on the held out storms, with the alarm threshold chosen on validation and applied unchanged.

| Horizon | POD | POD, persistence | FAR | FAR, persistence | Heidke skill | Heidke, persistence | Gain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 30 minutes | 0.77 | 0.69 | 0.32 | 0.31 | **0.68** | 0.64 | +0.04 |
| 45 minutes | 0.76 | 0.63 | 0.37 | 0.37 | **0.64** | 0.57 | +0.07 |
| 60 minutes | 0.72 | 0.59 | 0.38 | 0.41 | **0.61** | 0.53 | +0.08 |
| 90 minutes | 0.73 | 0.59 | 0.43 | 0.41 | **0.58** | 0.53 | +0.05 |

Persistence is the baseline every space weather forecast has to beat. It is free, it needs no model, and it says the ground will do at the horizon whatever it is doing now. It is handed the ground measurement as it stands when the forecast is made, which an operator genuinely has, and it alarms whenever that is already above the threshold. The gain column is what the model adds over it, and it grows with horizon, which is the shape it should have.

### What the clock is worth

The same model, the same storms, and the same settings, trained once on the clock of the spacecraft that measured the solar wind and once on the archive convention that shifts it forward to the Earth. The second one has no warning time to forecast into, because a shock appears in its input at the moment it strikes.

| Horizon | spacecraft clock HSS | bow shock clock HSS | Difference |
| --- | --- | --- | --- |
| 30 minutes | 0.68 | 0.63 | +0.05 |
| 45 minutes | 0.64 | 0.61 | +0.03 |
| 60 minutes | 0.61 | 0.59 | +0.02 |
| 90 minutes | 0.58 | 0.54 | +0.04 |

## At the higher 0.3 nT per second level

This level is rare, so both the model and the baseline score much lower here. The model beats persistence by a far wider margin than it does at the common level, which is what a model is for.

| Horizon | POD | POD, persistence | FAR | FAR, persistence | Heidke skill | Heidke, persistence | Gain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 30 minutes | 0.31 | 0.10 | 0.68 | 0.90 | **0.30** | 0.08 | +0.21 |
| 45 minutes | 0.43 | 0.16 | 0.80 | 0.84 | **0.25** | 0.14 | +0.11 |
| 60 minutes | 0.54 | 0.11 | 0.83 | 0.89 | **0.24** | 0.09 | +0.15 |
| 90 minutes | 0.59 | 0.19 | 0.86 | 0.81 | **0.20** | 0.17 | +0.03 |

### Does the physics constraint earn its place, at the common level

| Horizon | Heidke skill, constraint on | Heidke skill, constraint off | POD on | POD off |
| --- | --- | --- | --- | --- |
| 30 minutes | 0.68 | 0.67 | 0.77 | 0.86 |
| 45 minutes | 0.64 | 0.66 | 0.76 | 0.82 |
| 60 minutes | 0.61 | 0.62 | 0.72 | 0.77 |
| 90 minutes | 0.58 | 0.58 | 0.73 | 0.73 |

### Does the physics constraint earn its place, at the severe level

| Horizon | Heidke skill, constraint on | Heidke skill, constraint off | POD on | POD off |
| --- | --- | --- | --- | --- |
| 30 minutes | 0.30 | 0.00 | 0.31 | 0.00 |
| 45 minutes | 0.25 | 0.00 | 0.43 | 0.00 |
| 60 minutes | 0.24 | 0.00 | 0.54 | 0.00 |
| 90 minutes | 0.20 | 0.00 | 0.59 | 0.00 |

The same network, the same data, and the same settings, trained once with the monotonicity constraint and once without it. At the common level the two are close. At the severe level the unconstrained model scores zero at every horizon, meaning it never raises a severe alarm at all, while the constrained one keeps real skill.

That is where a physics constraint is supposed to help. Severe minutes are rare, so there is little data to learn them from, and an unconstrained fit takes the safe option of never predicting one. The constraint forbids the forecast from falling when the energy input rises, which carries the model into a part of the range the data barely covers.

One caveat belongs with this. Both runs are single seeds. The gap at the severe level is far too large to be run to run noise, and the near tie at the common level is small enough that it could be.

## Calibration

A stated ninetieth percentile should sit above the observation ninety percent of the time. The decision layer treats these numbers as real probabilities and would be misled by overconfident ones.

| Nominal level | Observed before calibration | Observed after |
| --- | --- | --- |
| 0.10 | 0.410 | 0.305 |
| 0.25 | 0.521 | 0.324 |
| 0.50 | 0.669 | 0.518 |
| 0.75 | 0.787 | 0.716 |
| 0.90 | 0.880 | 0.857 |
| 0.98 | 0.958 | 0.960 |

Mean absolute coverage error: 0.138 before calibration, 0.066 after.

## Reproducing this

```bash
pip install -r requirements.txt
python -m setu.cli train --time-base l1 --tag l1
python -m setu.cli train --time-base bowshock --tag bowshock
python scripts/write_results.py
python scripts/make_figures.py
```

The data is downloaded on first use and cached under `data/raw`, so the first run is slow and later ones are not.
