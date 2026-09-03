"""Turn a probabilistic forecast into a set of concrete futures to plan against.

The forecast model does not output one number. It outputs a distribution over how
hard the ground field will shake, which is the only honest thing to output for a
quantity this variable. A distribution cannot be fed to a circuit solver, so it is
sampled into scenarios, and every candidate action is then evaluated against the
whole set of them.

Each scenario carries a weight, and the weights are what make the risk measure in
``setu.decision.policy`` meaningful.
"""

from dataclasses import dataclass

import numpy as np

from setu.physics.earth import EARTH_MODELS
from setu.physics.geoelectric import geoelectric_field


@dataclass
class Scenario:
    """One possible future, ready to be pushed through the physics."""

    peak_dbdt: float          # nanotesla per second at the reference observatory
    weight: float             # probability mass this scenario stands for
    direction_deg: float      # bearing of the disturbance field, degrees from north
    ex: dict = None           # northward electric field by Earth model, volt per km
    ey: dict = None           # eastward electric field by Earth model


def sample_from_quantiles(quantile_values, quantile_levels, n_samples=200,
                          rng=None) -> np.ndarray:
    """Draw values from a distribution that is only known at some quantiles.

    Uniform probabilities are drawn and mapped through the inverse cumulative
    distribution by interpolation between the known quantiles. Above the highest
    known quantile an exponential tail is used, which is the usual shape for this
    quantity and which stops the sampler from silently truncating the extreme
    cases that matter most.
    """
    rng = rng or np.random.default_rng(0)
    values = np.asarray(quantile_values, dtype=float)
    levels = np.asarray(quantile_levels, dtype=float)
    u = rng.random(n_samples)
    out = np.interp(u, levels, values)

    upper = u > levels[-1]
    if upper.any():
        span = max(values[-1] - values[-2], 1e-6)
        remaining = 1.0 - levels[-1]
        excess = (u[upper] - levels[-1]) / remaining
        out[upper] = values[-1] + span * (-np.log(np.clip(1.0 - excess, 1e-9, 1.0)))

    lower = u < levels[0]
    if lower.any():
        out[lower] = values[0] * (u[lower] / levels[0])
    return np.clip(out, 0.0, None)


def dbdt_to_field(peak_dbdt, direction_deg, duration_min=60,
                  period_s=600.0, cadence_s=60.0):
    """Build an electric field per Earth model from a peak rate of change.

    A single peak value is not enough to drive the induction calculation, because
    the electric field depends on the frequency content of the disturbance and not
    only on its size. A representative waveform is built instead, a wave packet at
    the period that carries most storm energy, scaled so that its own peak rate of
    change matches the forecast. That waveform is then pushed through the layered
    Earth response for every ground model in the region.

    The direction is the bearing of the magnetic disturbance itself, measured in
    degrees east of north. It matters a great deal here. The induced electric field
    comes out rotated a quarter turn from the magnetic disturbance, so a magnetic
    perturbation pointing north produces an electric field pointing east, and the
    North East Region network is stretched east to west and is far more exposed to
    an eastward field than to a northward one.
    """
    t = np.arange(0, duration_min * 60.0, cadence_s)
    envelope = np.exp(-0.5 * ((t - t.mean()) / (0.25 * np.ptp(t) + 1e-9)) ** 2)
    shape = np.sin(2.0 * np.pi * t / period_s) * envelope

    # Scale the waveform so that its own peak rate of change is the forecast value.
    rate = np.abs(np.gradient(shape, cadence_s))
    amplitude = peak_dbdt / max(rate.max(), 1e-12)
    bearing = np.radians(direction_deg)
    bx = amplitude * shape * np.cos(bearing)
    by = amplitude * shape * np.sin(bearing)

    ex, ey = {}, {}
    for name, model in EARTH_MODELS.items():
        fx, fy = geoelectric_field(bx, by, model, cadence_s)
        peak = int(np.argmax(np.hypot(fx, fy)))
        ex[name] = float(fx[peak])
        ey[name] = float(fy[peak])
    return ex, ey


def build_scenarios(quantile_values, quantile_levels, n_samples=160,
                    direction_deg=None, seed=0):
    """Sample a scenario set from one forecast.

    The direction of the disturbance is sampled as well as its size, because it is
    genuinely uncertain and because the network cares about it a great deal.

    Directions are drawn around north. The storm time current systems that reach
    Indian latitudes, the ring current and the equatorial electrojet, both flow
    east to west, and a current sheet flowing east to west produces a magnetic
    perturbation in the northward component. That is why the horizontal component
    is the one that moves during a storm at Alibag and at Shillong. The electric
    field that follows is rotated a quarter turn from it and therefore points east
    to west, along the axis the regional grid is built on. The alignment between
    the driver and the network is the reason this region is exposed, and it is not
    an assumption in the model but a consequence of the geometry.
    """
    rng = np.random.default_rng(seed)
    peaks = sample_from_quantiles(quantile_values, quantile_levels, n_samples, rng)
    if direction_deg is None:
        directions = rng.normal(0.0, 25.0, n_samples)
    else:
        directions = np.full(n_samples, float(direction_deg))

    scenarios = []
    weight = 1.0 / n_samples
    cache = {}
    for peak, bearing in zip(peaks, directions):
        # The induction response is linear in amplitude, so the expensive transform
        # is computed once per direction and then simply scaled.
        key = round(float(bearing), 1)
        if key not in cache:
            cache[key] = dbdt_to_field(1.0, key)
        ex_unit, ey_unit = cache[key]
        scenarios.append(Scenario(
            peak_dbdt=float(peak), weight=weight, direction_deg=float(bearing),
            ex={k: v * peak for k, v in ex_unit.items()},
            ey={k: v * peak for k, v in ey_unit.items()},
        ))
    return scenarios
