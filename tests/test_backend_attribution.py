from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_training.ote_training.torch_models import LSTMClassifier, TCNClassifier
from preprocessing.backend_attribution import (
    compute_integrated_gradients,
    should_disable_cudnn_for_integrated_gradients,
)


def test_should_disable_cudnn_for_integrated_gradients_only_for_cuda_recurrent_models() -> None:
    lstm = LSTMClassifier(input_size=3, hidden_size=4, num_layers=2, dropout=0.2)
    tcn = TCNClassifier(input_size=3, hidden_size=4, num_layers=2, dropout=0.2)

    assert should_disable_cudnn_for_integrated_gradients(lstm, torch.device("cuda")) is True
    assert should_disable_cudnn_for_integrated_gradients(lstm, torch.device("cpu")) is False
    assert should_disable_cudnn_for_integrated_gradients(tcn, torch.device("cuda")) is False


def test_compute_integrated_gradients_restores_training_mode_and_shape() -> None:
    model = LSTMClassifier(input_size=3, hidden_size=4, num_layers=2, dropout=0.2)
    model.train()

    features = np.linspace(0.0, 1.0, num=5 * 4 * 3, dtype=np.float32).reshape(5, 4, 3)
    attributions = compute_integrated_gradients(model, features, steps=4, batch_size=2)

    assert attributions.shape == features.shape
    assert np.isfinite(attributions).all()
    assert model.training is True
