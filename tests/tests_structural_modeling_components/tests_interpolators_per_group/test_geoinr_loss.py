from __future__ import annotations

import pytest
import torch

from core.structural_modeling_components.interpolator_functions.geoinr import loss_grad


def test_loss_grad_is_zero_for_aligned_vectors_of_any_length():
    predicted = torch.tensor([[0.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    measured = torch.tensor([[0.0, 0.0, 100.0], [0.0, 0.01, 0.0]])

    assert loss_grad(predicted, measured).item() == pytest.approx(0.0, abs=1e-6)


def test_loss_grad_does_not_depend_on_measured_vector_length():
    predicted = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    measured = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    lengths = torch.tensor([[100.0], [0.01]])

    assert loss_grad(predicted, measured * lengths).item() == pytest.approx(loss_grad(predicted, measured).item())
