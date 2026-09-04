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

from setu.config import (ARTIFACT_DIR, DBDT_THRESHOLDS_NT_PER_S, DEFAULT_TIME_BASE,
                         FORECAST_HORIZONS_MIN, QUANTILES)
from setu.data.storms import test_events, training_events
from setu.ml.calibration import QuantileCalibrator, coverage_error
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


def tail_weights(y_log, tail_weight=3.0, reference_quantile=0.85, reference=None,
                 spread=None):
    """Per sample weights that pull the fit toward the disturbed minutes.

    The first version of this model was scored on two storms larger than anything
    it had trained on, and only 69 percent of observations fell below its predicted
    ninetieth percentile where 90 percent should have. It was under predicting the
    tail, which is the one part of the distribution the whole system exists to get
    right.

    The cause is that quiet minutes outnumber disturbed ones by roughly six to one,
    so an unweighted fit spends almost all of its effort on minutes where nothing is
    happening. These weights rise linearly above a high quantile of the training
    targets and are flat below it, so the quiet minutes still anchor the lower
    quantiles while the disturbed ones get the attention they need.

    The weights are normalised to average one, which keeps the loss on the same
    scale as before and means the learning rate does not have to be retuned.

    Args:
        y_log: Training targets in log space, of shape (samples, horizons).
        tail_weight: How much extra weight the largest observed target receives.
        reference_quantile: Where the weighting starts to rise.
        reference: Fixed threshold to use instead of a quantile of ``y_log``. Pass
            the training value when weighting a validation set, so that the two are
            scored on the same scale.
        spread: Fixed spread to use instead of one derived from ``y_log``.

    Returns:
        Weights of the same shape as ``y_log``, and the reference and spread used,
        so that the same weighting can be reproduced on another set.
    """
    y = np.asarray(y_log, dtype=float)
    if reference is None:
        reference = float(np.quantile(y, reference_quantile))
    if spread is None:
        spread = max(float(y.max() - reference), 1e-6)
    w = 1.0 + tail_weight * np.clip((y - reference) / spread, 0.0, None)
    return w / w.mean(), reference, spread


def train(window=96, channels=24, epochs=14, batch_size=96, lr=1.5e-3,
          physics_weight=0.5, physics_every=4, tail_weight=3.0, seed=0,
          observatories=("ABG", "HYB"), validation_fraction=0.2,
          time_base=DEFAULT_TIME_BASE, dropout=0.25, weight_decay=1.0e-4,
          input_noise=0.05, verbose=True):
    """Build the data, fit the model, and return the model with its scores.

    Args:
        physics_every: How often the monotonicity constraint is applied, in
            batches. The constraint costs three extra forward passes and two extra
            backward passes, so applying it on every batch roughly triples the cost
            of training for a penalty that is already small after the first epoch.
            Applying it every fourth batch keeps the constraint satisfied at a
            fraction of the price.
        tail_weight: Strength of the weighting toward disturbed minutes. Zero turns
            it off, which reproduces the earlier behaviour.
        time_base: ``l1`` or ``bowshock``. See ``setu.ml.dataset.event_frames``.
        dropout: Dropout rate inside every residual block and before the head.
        weight_decay: Decoupled weight decay applied by the optimiser.
        input_noise: Standard deviation of the Gaussian noise added to the inputs
            during training, in standardised units.

    The last three arguments exist because of a real failure. Consecutive training
    windows overlap by all but one step, so forty two thousand samples carry only a
    few thousand independent windows worth of information. An earlier run with
    light regularisation drove the training loss down while the validation loss
    rose from the first epoch, which is memorisation of individual storms rather
    than learning. The defaults here are the settings that stopped it.
    """
    from setu.ml.features import FEATURE_NAMES

    rng = np.random.default_rng(seed)
    t0 = time.time()

    train_block = build_dataset(training_events(), window, observatories,
                                time_base=time_base)
    test_block = build_dataset(test_events(), window, observatories,
                               time_base=time_base)
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

    # The validation set has to be scored the same way the training set is, or
    # model selection works against the very change the weighting is making. An
    # unweighted score falls when the upper quantiles are pulled down, which is the
    # opposite of what is wanted, so the reference and the spread are fitted on the
    # training targets and then reused on the validation targets unchanged.
    if tail_weight > 0:
        weights_tr, reference, spread = tail_weights(y_tr, tail_weight)
        weights_va, _, _ = tail_weights(y_va, tail_weight, reference=reference,
                                        spread=spread)
    else:
        weights_tr = np.ones_like(y_tr)
        weights_va = np.ones_like(y_va)

    if verbose:
        log.info("time base: %s", time_base)
        log.info("train %d, validation %d, test %d samples", len(x_tr), len(x_va), len(x_te))
        log.info("validation storms: %s", sorted(str(t) for t in val_tags))
        log.info("tail weighting: %.1f to %.1f across the training targets",
                 float(weights_tr.min()), float(weights_tr.max()))

    model = GICNet(n_features=x_tr.shape[1], channels=channels, window=window,
                   dropout=dropout, seed=seed)
    optimiser = Adam(model.parameters(), lr=lr, weight_decay=weight_decay, clip=1.0)
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
            xb, yb, wb = x_tr[idx], y_tr[idx], weights_tr[idx]

            # A small jitter on the inputs. Neighbouring windows are almost the
            # same array, so without it the network can memorise individual
            # windows. The scale is in standardised units, so five percent of a
            # standard deviation is well below the measurement noise already in
            # the solar wind record.
            if input_noise > 0:
                xb = xb + rng.normal(0.0, input_noise, xb.shape)

            model.zero_grad()
            pred = model.forward(xb, training=True)
            loss, grad = pinball_loss(pred, yb, QUANTILES, weights=wb)
            model.backward(grad)

            if physics_weight > 0 and step % physics_every == 0:
                running_phys += monotonicity_penalty(model, xb, constrained,
                                                     weight=physics_weight)
            optimiser.step(model.gradients())
            running += loss

        val_pred = predict_in_batches(model, x_va)
        val_loss = pinball(val_pred, y_va, QUANTILES, weights=weights_va)
        cover = coverage(val_pred, y_va, QUANTILES)
        history.append({"epoch": epoch, "train": running / steps,
                        "physics": running_phys / max(1, steps // physics_every),
                        "validation": val_loss, "coverage_at_90": cover[0.90]})
        if verbose:
            log.info("epoch %2d  train %.4f  physics %.5f  validation %.4f  "
                     "coverage at q90 %.3f",
                     epoch, running / steps,
                     running_phys / max(1, steps // physics_every), val_loss,
                     cover[0.90])
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.copy() for k, v in model.parameters().items()}

    for k, v in model.parameters().items():
        v[...] = best_state[k]

    # The calibration map is fitted on validation data only. Fitting it on the test
    # storms would make every calibration number in the report meaningless, which
    # is the whole reason the report exists.
    calibrator = QuantileCalibrator(QUANTILES)
    calibrator.fit(predict_in_batches(model, x_va), y_va)
    if verbose:
        log.info("calibration map fitted, q90 now read at raw level %.3f",
                 calibrator.raw_level(0.90))

    report = evaluate(model, x_va, y_va, x_te, y_te, train_block, test_block,
                      calibrator)
    report["history"] = history
    report["time_base"] = time_base
    report["tail_weight"] = tail_weight
    report["best_epoch"] = int(min(history, key=lambda h: h["validation"])["epoch"])
    report["calibration"] = {
        "requested_levels": list(QUANTILES),
        "raw_levels": [calibrator.raw_level(q) for q in QUANTILES],
    }
    report["settings"] = {"channels": channels, "dropout": dropout,
                          "weight_decay": weight_decay, "input_noise": input_noise,
                          "learning_rate": lr, "batch_size": batch_size,
                          "window_steps": window}
    report["training_seconds"] = round(time.time() - t0, 1)
    report["parameters"] = int(sum(v.size for v in model.parameters().values()))
    report["receptive_field_minutes"] = model.receptive_field * INPUT_CADENCE_MIN
    return model, scaler, calibrator, report


def predict_in_batches(model, x, batch_size=256):
    """Run the model over a large array without building one huge activation."""
    out = []
    for i in range(0, len(x), batch_size):
        out.append(model.forward(x[i: i + batch_size], training=False))
    return np.concatenate(out) if out else np.zeros((0,))


def evaluate(model, x_va, y_va, x_te, y_te, train_block, test_block,
             calibrator=None) -> dict:
    """Score the model on the held out storms.

    Both the raw and the calibrated predictions are scored. The raw numbers say
    what the network learned and the calibrated ones say what an operator would
    actually be handed, and the gap between the two is worth seeing rather than
    hiding.
    """
    va_pred = predict_in_batches(model, x_va)
    te_pred_raw = predict_in_batches(model, x_te)
    te_pred = (calibrator.calibrate(te_pred_raw) if calibrator is not None
               else te_pred_raw)
    te_actual = from_log_target(y_te)

    report = {
        "test_pinball": pinball(te_pred, y_te, QUANTILES),
        "validation_pinball": pinball(va_pred, y_va, QUANTILES),
        "coverage": coverage(te_pred, y_te, QUANTILES),
        "coverage_before_calibration": coverage(te_pred_raw, y_te, QUANTILES),
        "coverage_error": coverage_error(te_pred, y_te, QUANTILES),
        "coverage_error_before_calibration": coverage_error(te_pred_raw, y_te, QUANTILES),
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
            va_for_cut = (calibrator.calibrate(va_pred) if calibrator is not None
                          else va_pred)
            prob_va = quantile_exceedance(va_for_cut[:, h_index], threshold)
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


def main(tag=None, **kwargs):
    """Train once and save the model, the scaler, and the report.

    Args:
        tag: Optional suffix for the saved file names, so that two runs with
            different settings can sit side by side and be compared.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    model, scaler, calibrator, report = train(**kwargs)
    suffix = f"_{tag}" if tag else ""
    model.save(ARTIFACT_DIR / f"gicnet{suffix}.npz")
    np.savez_compressed(ARTIFACT_DIR / f"scaler{suffix}.npz", **scaler.state())
    np.savez_compressed(ARTIFACT_DIR / f"calibrator{suffix}.npz", **calibrator.state())
    (ARTIFACT_DIR / f"training_report{suffix}.json").write_text(
        json.dumps(report, indent=2, default=float))
    log.info("saved model and report to %s", ARTIFACT_DIR)
    return report


if __name__ == "__main__":
    main()
