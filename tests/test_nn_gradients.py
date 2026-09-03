"""Check every hand written backward pass against finite differences.

The network in this project has no autograd behind it, so a wrong sign or a
transposed index in a backward pass would train quietly to a worse answer instead
of raising an error. These tests are the guard against that.
"""

import numpy as np
import pytest

from setu.ml.nn import CausalConv1d, ChannelNorm, Dense, ResidualBlock, pinball_loss

RNG = np.random.default_rng(11)


def max_relative_error(layer, x, samples=8, eps=1e-6):
    y = layer.forward(x, True)
    upstream = RNG.normal(size=y.shape)
    analytic = layer.backward(upstream)
    worst = 0.0
    for _ in range(samples):
        idx = tuple(RNG.integers(0, s) for s in x.shape)
        plus, minus = x.copy(), x.copy()
        plus[idx] += eps
        minus[idx] -= eps
        numeric = float(((layer.forward(plus, False) - layer.forward(minus, False))
                         * upstream).sum() / (2 * eps))
        worst = max(worst, abs(numeric - analytic[idx]) / (abs(numeric) + 1e-6))
    return worst


@pytest.mark.parametrize("dilation", [1, 2, 8])
def test_causal_conv_gradient(dilation):
    x = RNG.normal(size=(3, 5, 24))
    assert max_relative_error(CausalConv1d(5, 7, 3, dilation, RNG), x) < 1e-5


def test_channel_norm_gradient():
    x = RNG.normal(size=(3, 6, 20))
    assert max_relative_error(ChannelNorm(6), x) < 1e-5


def test_residual_block_gradient():
    x = RNG.normal(size=(2, 8, 24))
    assert max_relative_error(ResidualBlock(8, 2, 3, 0.0, RNG), x) < 1e-5


def test_dense_gradient():
    x = RNG.normal(size=(4, 6))
    assert max_relative_error(Dense(6, 3, RNG), x) < 1e-5


def test_convolution_is_causal():
    """A change in the future must leave every earlier output untouched."""
    conv = CausalConv1d(2, 3, 3, dilation=4, rng=RNG)
    x = RNG.normal(size=(1, 2, 40))
    before = conv.forward(x, False)
    disturbed = x.copy()
    disturbed[:, :, 25:] += 50.0
    after = conv.forward(disturbed, False)
    assert np.abs(before[:, :, :25] - after[:, :, :25]).max() == 0.0


def test_pinball_gradient_matches_finite_difference():
    pred = RNG.normal(size=(5, 2, 3))
    target = RNG.normal(size=(5, 2))
    quantiles = (0.1, 0.5, 0.9)
    _, grad = pinball_loss(pred, target, quantiles)
    eps = 1e-6
    for _ in range(8):
        idx = tuple(RNG.integers(0, s) for s in pred.shape)
        plus, minus = pred.copy(), pred.copy()
        plus[idx] += eps
        minus[idx] -= eps
        numeric = (pinball_loss(plus, target, quantiles)[0]
                   - pinball_loss(minus, target, quantiles)[0]) / (2 * eps)
        assert abs(numeric - grad[idx]) < 1e-6


def test_pinball_recovers_the_quantile():
    """Minimising the loss for one level should land on that sample quantile."""
    sample = RNG.gamma(2.0, 1.0, size=(4000, 1))
    levels = (0.1, 0.5, 0.9)
    guess = np.zeros((1, 1, 3))
    for _ in range(3000):
        pred = np.repeat(guess, sample.shape[0], axis=0)
        _, grad = pinball_loss(pred, sample, levels)
        guess -= 5.0 * grad.sum(axis=0, keepdims=True)
    truth = np.quantile(sample, levels)
    assert np.allclose(guess.ravel(), truth, atol=0.08)
