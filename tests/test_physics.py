"""Checks on the physics chain, from the Earth model to the transformer.

Each test states a property that has to hold for a physical reason, so a failure
points at a real modelling error rather than at a changed number.
"""

import numpy as np
import pytest

from setu.decision.blockers import exhaustive_placement, greedy_placement
from setu.decision.scenarios import build_scenarios, dbdt_to_field
from setu.grid.network import Network
from setu.grid.voltage import VoltageModel
from setu.physics.earth import (BENGAL_TRIPURA_BASIN, EARTH_MODELS, SHILLONG_PLATEAU,
                                EarthModel)
from setu.physics.geoelectric import dbdt, field_magnitude, geoelectric_field
from setu.physics.gic import GICSolver, uniform_field_case


def test_uniform_half_space_matches_the_closed_form():
    """A single layer model has an impedance that can be written down exactly."""
    rho = 100.0
    model = EarthModel("uniform", (rho,), (), "uniform half space")
    f = np.array([1e-3, 1e-2, 1e-1])
    omega = 2 * np.pi * f
    mu0 = 4e-7 * np.pi
    expected = np.sqrt(1j * omega * mu0 * rho)
    assert np.allclose(model.surface_impedance(f), expected, rtol=1e-10)


def test_apparent_resistivity_of_a_half_space_is_flat():
    model = EarthModel("uniform", (250.0,), (), "")
    f = np.logspace(-4, -1, 12)
    assert np.allclose(model.apparent_resistivity(f), 250.0, rtol=1e-8)


def test_resistive_ground_gives_a_larger_electric_field():
    """This is the whole reason the North East is worth studying."""
    t = np.arange(0, 4 * 3600, 60.0)
    bx = 150 * np.sin(2 * np.pi * t / 400.0)
    by = np.zeros_like(bx)
    plateau = field_magnitude(*geoelectric_field(bx, by, SHILLONG_PLATEAU)).max()
    basin = field_magnitude(*geoelectric_field(bx, by, BENGAL_TRIPURA_BASIN)).max()
    assert plateau > 3.0 * basin


def test_geoelectric_field_is_linear_in_the_magnetic_field():
    """Induction is a linear operator, so doubling the input doubles the output."""
    t = np.arange(0, 3600, 60.0)
    bx = 80 * np.sin(2 * np.pi * t / 500.0)
    by = 40 * np.cos(2 * np.pi * t / 700.0)
    one = np.array(geoelectric_field(bx, by, SHILLONG_PLATEAU))
    two = np.array(geoelectric_field(2 * bx, 2 * by, SHILLONG_PLATEAU))
    assert np.allclose(two, 2.0 * one, rtol=1e-9)


def test_a_static_field_induces_nothing():
    constant = np.full(600, 25000.0)
    ex, ey = geoelectric_field(constant, constant, SHILLONG_PLATEAU)
    assert field_magnitude(ex, ey).max() < 1e-9


def test_electric_field_is_rotated_a_quarter_turn_from_the_magnetic_field():
    """A northward magnetic disturbance has to give an eastward electric field."""
    ex, ey = dbdt_to_field(1.0, direction_deg=0.0)
    for name in EARTH_MODELS:
        assert abs(ex[name]) < 1e-9
        assert abs(ey[name]) > 1e-3


def test_earth_currents_sum_to_zero():
    """Kirchhoff. Every ampere that goes into the ground came out somewhere else."""
    solver = GICSolver(Network())
    result = uniform_field_case(solver, 0.7, 1.3)
    assert abs(result.neutral_current.sum()) < 1e-8


def test_current_is_linear_in_the_electric_field():
    solver = GICSolver(Network())
    one = uniform_field_case(solver, 1.0, 0.0).neutral_current
    three = uniform_field_case(solver, 3.0, 0.0).neutral_current
    assert np.allclose(three, 3.0 * one, rtol=1e-9)


def test_the_network_is_more_exposed_east_to_west():
    """The corridor and the direct current route both run east to west."""
    solver = GICSolver(Network())
    northward = np.abs(uniform_field_case(solver, 1.0, 0.0).neutral_current).max()
    eastward = np.abs(uniform_field_case(solver, 0.0, 1.0).neutral_current).max()
    assert eastward > 2.0 * northward


def test_blocking_a_neutral_removes_its_current():
    net = Network()
    blocked = GICSolver(net, {"BWNC"})
    result = uniform_field_case(blocked, 0.0, 1.0)
    index = result.codes.index("BWNC")
    assert abs(result.neutral_current[index]) < 1e-3


def test_blocking_pushes_current_into_the_neighbours():
    """This is why placement cannot be decided one site at a time."""
    net = Network()
    before = uniform_field_case(GICSolver(net), 0.0, 1.0)
    after = uniform_field_case(GICSolver(net, {"BWNC"}), 0.0, 1.0)
    neighbour = before.codes.index("BLPR")
    assert abs(after.neutral_current[neighbour]) > abs(before.neutral_current[neighbour])


def test_voltage_deviation_is_negative_and_grows_with_load():
    net = Network()
    model = VoltageModel(net)
    small = model.voltage_deviation(np.full(net.n, 10.0))
    large = model.voltage_deviation(np.full(net.n, 40.0))
    assert small.max() <= 0.0
    assert large.min() < small.min()


def test_greedy_placement_matches_the_exhaustive_answer_for_a_small_budget():
    """Greedy is used because exhaustive search does not scale. It has to agree
    with exhaustive search where exhaustive search is still possible."""
    scenarios = build_scenarios([1.0, 1.8, 3.2, 5.6, 10.0, 18.0],
                                [0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
                                n_samples=20, seed=3)
    pool = ["BWNC", "ALPD", "BNGR", "MRNI", "AZRA", "SLCR", "IMPL"]
    greedy = greedy_placement(scenarios, budget=3, metric="reactive", candidates=pool)
    exact = exhaustive_placement(scenarios, 3, metric="reactive", candidates=pool)
    assert set(greedy["chosen"]) == set(exact["chosen"])


def test_more_devices_never_make_the_region_worse():
    scenarios = build_scenarios([1.0, 1.8, 3.2, 5.6, 10.0, 18.0],
                                [0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
                                n_samples=15, seed=4)
    steps = greedy_placement(scenarios, budget=5)["steps"]
    scores = [s["score"] for s in steps]
    assert all(b <= a for a, b in zip(scores, scores[1:]))


def test_dbdt_of_a_known_sinusoid():
    t = np.arange(0, 1200, 1.0)
    period = 300.0
    amplitude = 100.0
    bx = amplitude * np.sin(2 * np.pi * t / period)
    rate = dbdt(bx, np.zeros_like(bx), cadence_s=1.0)
    expected = amplitude * 2 * np.pi / period
    assert rate.max() == pytest.approx(expected, rel=0.02)
