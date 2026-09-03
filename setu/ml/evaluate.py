"""Scoring, in the language the space weather community actually uses.

A regression error is close to useless for this problem. The quantity being
predicted is quiet almost all of the time, so a model that always says quiet gets a
very small mean error and is worth nothing. What matters is whether the model
raises the alarm when the ground really does shake, and how often it raises it when
nothing happens.

The scores below are the standard contingency table scores. Every one of them is
computed against a stated baseline, because a skill score without a baseline is not
a claim about anything.
"""

import numpy as np


def contingency(observed_event, forecast_event):
    """Counts of hits, false alarms, misses, and correct rejections."""
    o = np.asarray(observed_event, dtype=bool).ravel()
    f = np.asarray(forecast_event, dtype=bool).ravel()
    return {
        "hits": int(np.sum(o & f)),
        "false_alarms": int(np.sum(~o & f)),
        "misses": int(np.sum(o & ~f)),
        "correct_rejections": int(np.sum(~o & ~f)),
    }


def skill_scores(observed_event, forecast_event) -> dict:
    """Probability of detection, false alarm ratio, and three skill scores.

    The Heidke skill score compares the model against random chance that keeps the
    same marginal rates. The Peirce score, also called the true skill statistic,
    is the probability of detection minus the false alarm rate and is not inflated
    by a rare event. The bias says whether the model raises too many or too few
    alarms overall.
    """
    c = contingency(observed_event, forecast_event)
    h, f, m, r = c["hits"], c["false_alarms"], c["misses"], c["correct_rejections"]
    total = h + f + m + r
    pod = h / (h + m) if (h + m) else float("nan")
    far = f / (h + f) if (h + f) else float("nan")
    pofd = f / (f + r) if (f + r) else float("nan")

    expected = ((h + m) * (h + f) + (r + f) * (r + m)) / total if total else float("nan")
    hss = ((h + r - expected) / (total - expected)
           if total and (total - expected) else float("nan"))
    pss = pod - pofd
    csi = h / (h + f + m) if (h + f + m) else float("nan")
    bias = (h + f) / (h + m) if (h + m) else float("nan")

    out = dict(c)
    out.update({"pod": pod, "far": far, "pofd": pofd, "hss": hss,
                "pss": pss, "csi": csi, "bias": bias, "n": total,
                "event_rate": (h + m) / total if total else float("nan")})
    return out


def best_threshold(observed_event, probability, grid=None):
    """Pick the probability cut that maximises the Heidke skill score.

    An operator has to turn a probability into a yes or a no eventually. Choosing
    the cut on the training set and reporting the score on the held out set is the
    honest way to do it, and it is what the training script does.
    """
    grid = np.linspace(0.02, 0.98, 49) if grid is None else np.asarray(grid)
    best, best_score = grid[0], -np.inf
    for p in grid:
        score = skill_scores(observed_event, np.asarray(probability) >= p)["hss"]
        if np.isfinite(score) and score > best_score:
            best, best_score = p, score
    return float(best), float(best_score)


def reliability(observed_event, probability, bins=10):
    """How well the stated probabilities match the observed frequencies.

    A model that says forty percent should be right about forty percent of the
    time. This is the check that makes a probability usable in a cost calculation
    rather than being a number that only ranks cases.
    """
    o = np.asarray(observed_event, dtype=float).ravel()
    p = np.asarray(probability, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if m.sum() == 0:
            continue
        rows.append({"bin_low": float(lo), "bin_high": float(hi),
                     "count": int(m.sum()), "forecast": float(p[m].mean()),
                     "observed": float(o[m].mean())})
    return rows


def brier_score(observed_event, probability) -> float:
    o = np.asarray(observed_event, dtype=float).ravel()
    p = np.asarray(probability, dtype=float).ravel()
    return float(np.mean((p - o) ** 2))


def brier_skill_score(observed_event, probability) -> float:
    """Brier score compared against always forecasting the climatological rate."""
    o = np.asarray(observed_event, dtype=float).ravel()
    reference = brier_score(o, np.full_like(o, o.mean()))
    if reference <= 0:
        return float("nan")
    return float(1.0 - brier_score(o, probability) / reference)


def pinball(pred_quantiles, target, quantiles) -> float:
    """Mean quantile loss, which is a proper score for the whole distribution."""
    q = np.asarray(quantiles).reshape(1, 1, -1)
    diff = np.asarray(target)[:, :, None] - np.asarray(pred_quantiles)
    return float(np.maximum(q * diff, (q - 1.0) * diff).mean())


def coverage(pred_quantiles, target, quantiles) -> dict:
    """Fraction of observations that fall below each predicted quantile.

    For a well calibrated model the fraction below the ninetieth percentile should
    be about ninety percent. This is the simplest possible calibration check and it
    catches an overconfident model immediately.
    """
    t = np.asarray(target)[:, :, None]
    below = (t <= np.asarray(pred_quantiles)).mean(axis=(0, 1))
    return {float(q): float(c) for q, c in zip(quantiles, below)}


def persistence_baseline(recent_dbdt, horizons_steps):
    """The forecast that says the next hour looks like the last few minutes.

    This is the baseline every space weather forecast has to beat, because it is
    free and it is surprisingly hard to improve on during the quiet times that
    make up most of the record.
    """
    recent = np.asarray(recent_dbdt, dtype=float)
    return np.repeat(recent[:, None], len(horizons_steps), axis=1)
