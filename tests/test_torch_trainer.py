from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from model_training.ote_training.torch_trainer import create_loader


def test_create_loader_builds_lazy_window_batches():
    source = SimpleNamespace(
        matrix=np.arange(20, dtype=np.float32).reshape(10, 2),
        window_size=4,
        sampled_indices=None,
    )
    y = np.arange(7, dtype=np.float32)
    w = np.ones(7, dtype=np.float32)

    loader = create_loader(
        X=source,
        y=y,
        w=w,
        batch_size=3,
        shuffle=False,
        pin_memory=False,
        num_workers=0,
    )

    features, targets, weights = next(iter(loader))

    assert tuple(features.shape) == (3, 4, 2)
    np.testing.assert_allclose(
        features[0].numpy(),
        np.array(
            [
                [0, 1],
                [2, 3],
                [4, 5],
                [6, 7],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(targets.numpy(), y[:3])
    np.testing.assert_allclose(weights.numpy(), w[:3])
