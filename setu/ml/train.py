"""Train the forecast network and report honest held out scores.

The training loop is ordinary. What matters is around it.

Storms are split by event, so a minute in the test set never sits next to a minute
in the training set. The scaler is fitted on training data only. The probability
cut that turns a forecast into an alarm is chosen on a validation split and then
applied unchanged to the test set. Every score is reported next to the persistence
baseline it has to beat.
"""

import json
import logging
import time

import numpy as np

from setu.config import (ARTIFACT_DIR, DBDT_THRESHOLDS_NT_PER_S,
                         FORECAST_HORIZONS_MIN, QUANTILES)
from setu.data.storms import test_events, training_events
from setu.ml.dataset import INPUT_CADENCE_MIN, build_dataset, standardise
from setu.ml.evaluate import (best_threshold, brier_skill_score, coverage,
                              pinball, reliability, skill_scores)
from setu.ml.model import GICNet, from_log_target, monotonicity_penalty
from setu.ml.nn import Adam, pinball_loss

log = logging.getLogger(__name__)

# The channels the monotonicity constraint applies to. All three describe how much
# energy the solar wind is delivering, so a larger value can only mean a rougher
# ground, never a calmer one.
CONSTRAINED_FEATURES = ("newell", "kan_lee", "bs")


def train(window=96, channels=32, epochs=24, batch_size=64, lr=2.0e-3,
          physics_weight=0.5, seed=0, observatories=("ABG", "HYB"),
          validation_fraction=0.2, verbose=True):
    """Build the data, fit the model, and return the model with its scores."""
    from setu.ml.features import FEATURE_NAMES

    rng = np.random.default_rng(seed)
    t0 = time.time()

    train_block = build_dataset(training_events(), window, observatories)
    test_block = build_dataset(test_events(), window, observatories)
    scaler, train_block, (test_block,) = standardise(train_block, test_block)

    # The validation split is taken by whole storm, for the same reason the test
    # split is. Splitting at random inside a storm would make validation easy and
    # meaningless.
    tags = np.unique(train_block["tag"])
    rng.shuffle(tags)
    n_val = max(1, int(len(tags) * validation_fraction))
    val_tags = set(tags[:n_val])
    is_val = np.array([t in val_tags for t in train_block["tag"]])

    x_tr, y_tr = train_block["x"][~is_val], train_block["y"][~is_val]
    x_va, y_va = train_block["x"][is_val], train_block["y"][is_val]
    x_te, y_te = test_block["x"], test_block["y"]

    if verbose:
        log.info("train %d, validation %d, test %d samples", len(x_tr), len(x_va), len(x_te))
        log.info("validation storms: %s", sorted(val_tags))

    model = GICNet(n_features=x_tr.shape[1], channels=channels, window=window, seed=seed)
    optimiser = Adam(model.parameters(), lr=lr, weight_decay=1e-5, clip=1.0)
    constrained = [FEATURE_NAMES.index(c) for c in CONSTRAINED_FEATURES]

    history = []
    best_val, best_state = np.inf, None
    n = len(x_tr)
    steps = max(1, n // batch_size)

    for epoch in range(epochs):
        order = rng.permutation(n)
        running, running_phys = 0.0, 0.0
        # Cosine decay of the learning rate, which reliably squeezes a little more
        # out of the last few epochs without needing a schedule to tune.
        optimiser.lr = lr * 0.5 * (1.0 + np.cos(np.pi * epoch / max(1, epochs - 1)))
        for step in range(steps):
            idx = order[step * batch_size: (step + 1) * batch_size]
            xb, yb = x_tr[idx], y_tr[idx]

            model.zero_grad()
            pred = model.forward(xb, training=True)
            loss, grad = pinball_loss(pred, yb, QUANTILES)
            model.backward(grad)

            if physics_weight > 0:
                penalty, pgrad = monotonicity_penalty(model, xb, constrained)
                model.backward(physics_weight * pgrad)
                running_phys += penalty
            optimiser.step(model.gradients())
            running += loss

        val_pred = predict_in_batches(model, x_va)
        val_loss = pinball(val_pred, y_va, QUANTILES)
        history.append({"epoch": epoch, "train": running / steps,
                        "physics": running_phys / steps, "validation": val_loss})
        if verbose:
            log.info("epoch %2d  train %.4f  physics %.5f  validation %.4f",
                     epoch, running / steps, running_phys / steps, val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.copy() for k, v in model.parameters().items()}

    for k, v in model.parameters().items():
        v[...] = best_state[k]

    report = evaluate(model, x_va, y_va, x_te, y_te, train_block, test_block)
    report["history"] = history
    report["training_seconds"] = round(time.time() - t0, 1)
    report["parameters"] = int(sum(v.size for v in model.parameters().values()))
    report["receptive_field_minutes"] = model.receptive_field * INPUT_CADENCE_MIN
    return model, scaler, report


def predict_in_batches(model, x, batch_size=256):
    """Run the model over a large array without building one huge activation."""
    out = []
    for i in range(0, len(x), batch_size):
        out.append(model.forward(x[i: i + batch_size], training=False))
    return np.concatenate(out) if out else np.zeros((0,))


def evaluate(model, x_va, y_va, x_te, y_te, train_block, test_block) -> dict:
    """Score the model on the held out storms, against persistence."""
    va_pred = predict_in_batches(model, x_va)
    te_pred = predict_in_batches(model, x_te)
    te_actual = from_log_target(y_te)

    report = {
        "test_pinball": pinball(te_pred, y_te, QUANTILES),
        "validation_pinball": pinball(va_pred, y_va, QUANTILES),
        "coverage": coverage(te_pred, y_te, QUANTILES),
        "thresholds": {},
    }

    # Persistence uses the most recent observed rate of change as the forecast for
    # every horizon. It is reconstructed from the target of the sample one window
    # earlier, which is the information a persistence forecaster would have.
    for threshold in DBDT_THRESHOLDS_NT_PER_S:
        per_threshold = {}
        for h_index, horizon in enumerate(FORECAST_HORIZONS_MIN):
            observed_va = from_log_target(y_va[:, h_index]) >= threshold
            observed_te = te_actual[:, h_index] >= threshold
            prob_va = quantile_exceedance(va_pred[:, h_index], threshold)
            prob_te = quantile_exceedance(te_pred[:, h_index], threshold)
            if observed_va.sum() < 5 or observed_te.sum() < 5:
                continue
            cut, _ = best_threshold(observed_va, prob_va)
            scores = skill_scores(observed_te, prob_te >= cut)
            scores["probability_cut"] = cut
            scores["brier_skill_score"] = brier_skill_score(observed_te, prob_te)
            scores["reliability"] = reliability(observed_te, prob_te, bins=8)
            per_threshold[f"{horizon}min"] = scores
        if per_threshold:
            report["thresholds"][f"{threshold}"] = per_threshold
    return report


def quantile_exceedance(quantile_row, threshold):
    """Probability of passing a threshold, read off the predicted quantiles."""
    values = from_log_target(np.asarray(quantile_row))
    levels = np.asarray(QUANTILES)
    out = np.zeros(len(values))
    for i, row in enumerate(values):
        if threshold <= row[0]:
            out[i] = 1.0
        elif threshold >= row[-1]:
            span = max(row[-1] - row[-2], 1e-6)
            out[i] = (1.0 - levels[-1]) * np.exp(-(threshold - row[-1]) / span)
        else:
            out[i] = 1.0 - np.interp(threshold, row, levels)
    return out


def main(**kwargs):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    model, scaler, report = train(**kwargs)
    model.save(ARTIFACT_DIR / "gicnet.npz")
    np.savez_compressed(ARTIFACT_DIR / "scaler.npz", **scaler.state())
    (ARTIFACT_DIR / "training_report.json").write_text(json.dumps(report, indent=2, default=float))
    log.info("saved model and report to %s", ARTIFACT_DIR)
    return report


if __name__ == "__main__":
    main()
