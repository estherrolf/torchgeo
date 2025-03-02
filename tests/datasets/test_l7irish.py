# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import glob
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.monkeypatch import MonkeyPatch
from rasterio.crs import CRS

import torchgeo.datasets.utils
from torchgeo.datasets import BoundingBox, IntersectionDataset, L7Irish, UnionDataset


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestL7Irish:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> L7Irish:
        monkeypatch.setattr(torchgeo.datasets.l7irish, "download_url", download_url)
        md5s = {
            "austral": "c06147330141517f7eee55ea931c4787",
            "boreal": "4b598e55f0d6d33da3672190ebf96268",
        }

        url = os.path.join("tests", "data", "l7irish", "{}.tar.gz")
        monkeypatch.setattr(L7Irish, "url", url)
        monkeypatch.setattr(L7Irish, "md5s", md5s)
        root = str(tmp_path)
        transforms = nn.Identity()
        return L7Irish(root, transforms=transforms, download=True, checksum=True)

    def test_getitem(self, dataset: L7Irish) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)

    def test_and(self, dataset: L7Irish) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: L7Irish) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_plot(self, dataset: L7Irish) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle="Test")
        plt.close()

    def test_already_extracted(self, dataset: L7Irish) -> None:
        L7Irish(root=dataset.root, download=True)

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join("tests", "data", "l7irish", "*.tar.gz")
        root = str(tmp_path)
        for tarfile in glob.iglob(pathname):
            shutil.copy(tarfile, root)
        L7Irish(root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            L7Irish(str(tmp_path))

    def test_plot_prediction(self, dataset: L7Irish) -> None:
        x = dataset[dataset.bounds]
        x["prediction"] = x["mask"].clone()
        dataset.plot(x, suptitle="Prediction")
        plt.close()

    def test_invalid_query(self, dataset: L7Irish) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]

    def test_rgb_bands_absent_plot(self, dataset: L7Irish) -> None:
        with pytest.raises(
            ValueError, match="Dataset doesn't contain some of the RGB bands"
        ):
            ds = L7Irish(root=dataset.root, bands=["B1", "B2", "B5"])
            x = ds[ds.bounds]
            ds.plot(x, suptitle="Test")
            plt.close()
