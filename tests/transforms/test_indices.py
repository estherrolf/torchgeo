# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Dict

import pytest
import torch
from torch import Tensor

from torchgeo.transforms import (
    AppendBNDVI,
    AppendGBNDVI,
    AppendGNDVI,
    AppendGRNDVI,
    AppendNBR,
    AppendNDBI,
    AppendNDRE,
    AppendNDSI,
    AppendNDVI,
    AppendNDWI,
    AppendNormalizedDifferenceIndex,
    AppendRBNDVI,
    AppendSWI,
    AppendTriBandNormalizedDifferenceIndex,
    AugmentationSequential,
)


@pytest.fixture
def sample() -> Dict[str, Tensor]:
    return {
        "image": torch.arange(3 * 4 * 4, dtype=torch.float).view(3, 4, 4),
        "mask": torch.arange(4 * 4, dtype=torch.long).view(1, 4, 4),
    }


@pytest.fixture
def batch() -> Dict[str, Tensor]:
    return {
        "image": torch.arange(2 * 3 * 4 * 4, dtype=torch.float).view(2, 3, 4, 4),
        "mask": torch.arange(2 * 4 * 4, dtype=torch.long).view(2, 1, 4, 4),
    }


def test_append_index_sample(sample: Dict[str, Tensor]) -> None:
    c, h, w = sample["image"].shape
    aug = AugmentationSequential(
        AppendNormalizedDifferenceIndex(index_a=0, index_b=1),
        data_keys=["image", "mask"],
    )
    output = aug(sample)
    assert output["image"].shape == (1, c + 1, h, w)


def test_append_index_batch(batch: Dict[str, Tensor]) -> None:
    b, c, h, w = batch["image"].shape
    aug = AugmentationSequential(
        AppendNormalizedDifferenceIndex(index_a=0, index_b=1),
        data_keys=["image", "mask"],
    )
    output = aug(batch)
    assert output["image"].shape == (b, c + 1, h, w)


def test_append_triband_index_batch(batch: Dict[str, Tensor]) -> None:
    b, c, h, w = batch["image"].shape
    aug = AugmentationSequential(
        AppendTriBandNormalizedDifferenceIndex(index_a=0, index_b=1, index_c=2),
        data_keys=["image", "mask"],
    )
    output = aug(batch)
    assert output["image"].shape == (b, c + 1, h, w)


@pytest.mark.parametrize(
    "index",
    [
        AppendBNDVI,
        AppendNBR,
        AppendNDBI,
        AppendNDRE,
        AppendNDSI,
        AppendNDVI,
        AppendNDWI,
        AppendSWI,
        AppendGNDVI,
    ],
)
def test_append_normalized_difference_indices(
    sample: Dict[str, Tensor], index: AppendNormalizedDifferenceIndex
) -> None:
    c, h, w = sample["image"].shape
    aug = AugmentationSequential(index(0, 1), data_keys=["image", "mask"])
    output = aug(sample)
    assert output["image"].shape == (1, c + 1, h, w)


@pytest.mark.parametrize("index", [AppendGBNDVI, AppendGRNDVI, AppendRBNDVI])
def test_append_tri_band_normalized_difference_indices(
    sample: Dict[str, Tensor], index: AppendTriBandNormalizedDifferenceIndex
) -> None:
    c, h, w = sample["image"].shape
    aug = AugmentationSequential(index(0, 1, 2), data_keys=["image", "mask"])
    output = aug(sample)
    assert output["image"].shape == (1, c + 1, h, w)
