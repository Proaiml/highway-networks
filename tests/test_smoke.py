"""Duman testleri: modeller kurulur, şekil korunur, gradyan ilk katmana ulaşır (CPU, saniyeler)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import HighwayLayer, HighwayNetwork, PlainDeepNetwork  # noqa: E402


def test_shapes_are_preserved():
    x = torch.randn(8, 32)
    assert HighwayLayer(32)(x).shape == x.shape
    assert HighwayNetwork(32, 10)(x).shape == x.shape
    assert PlainDeepNetwork(32, 10)(x).shape == x.shape


def test_gradient_reaches_the_first_layer_of_a_deep_highway_net():
    torch.manual_seed(0)
    net = HighwayNetwork(32, 30)
    net(torch.randn(16, 32)).pow(2).mean().backward()
    first = next(p for p in net.parameters() if p.grad is not None)
    assert torch.isfinite(first.grad).all() and first.grad.abs().sum() > 0
