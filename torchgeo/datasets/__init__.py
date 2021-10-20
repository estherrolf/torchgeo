# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo datasets."""

from .advance import ADVANCE
from .benin_cashews import BeninSmallHolderCashews
from .cbf import CanadianBuildingFootprints
from .cdl import CDL
from .chesapeake import (
    Chesapeake,
    Chesapeake7,
    Chesapeake13,
    ChesapeakeCVPR,
    ChesapeakeCVPRPrior,
    ChesapeakeDC,
    ChesapeakeDE,
    ChesapeakeMD,
    ChesapeakeNY,
    ChesapeakePA,
    ChesapeakeVA,
    ChesapeakeWV,
)
from .cowc import COWC, COWCCounting, COWCDetection
from .cv4a_kenya_crop_type import CV4AKenyaCropType
from .cyclone import TropicalCycloneWindEstimation
from .enviroatlas import (
    Enviroatlas
)
from .etci2021 import ETCI2021
from .eurosat import EuroSAT
from .geo import (
    GeoDataset,
    RasterDataset,
    VectorDataset,
    VisionClassificationDataset,
    VisionDataset,
    ZipDataset,
)
from .gid15 import GID15
from .landcoverai import LandCoverAI
from .landsat import (
    Landsat,
    Landsat1,
    Landsat2,
    Landsat3,
    Landsat4MSS,
    Landsat4TM,
    Landsat5MSS,
    Landsat5TM,
    Landsat7,
    Landsat8,
    Landsat9,
)
from .levircd import LEVIRCDPlus
from .naip import NAIP
from .nwpu import VHR10
from .patternnet import PatternNet
from .resisc45 import RESISC45
from .sen12ms import SEN12MS
from .sentinel import Sentinel, Sentinel2
from .so2sat import So2Sat
from .spacenet import SpaceNet, SpaceNet1, SpaceNet2, SpaceNet4
from .ucmerced import UCMerced
from .utils import BoundingBox, collate_dict
from .zuericrop import ZueriCrop

__all__ = (
    # GeoDataset
    "CanadianBuildingFootprints",
    "CDL",
    "Chesapeake",
    "Chesapeake7",
    "Chesapeake13",
    "ChesapeakeDC",
    "ChesapeakeDE",
    "ChesapeakeMD",
    "ChesapeakeNY",
    "ChesapeakePA",
    "ChesapeakeVA",
    "ChesapeakeWV",
    "ChesapeakeCVPR",
    "ChesapeakeCVPRPrior",
    "Enviroatlas",
    "Landsat",
    "Landsat1",
    "Landsat2",
    "Landsat3",
    "Landsat4MSS",
    "Landsat4TM",
    "Landsat5MSS",
    "Landsat5TM",
    "Landsat7",
    "Landsat8",
    "Landsat9",
    "NAIP",
    "Sentinel",
    "Sentinel2",
    # VisionDataset
    "ADVANCE",
    "BeninSmallHolderCashews",
    "COWC",
    "COWCCounting",
    "COWCDetection",
    "CV4AKenyaCropType",
    "ETCI2021",
    "EuroSAT",
    "GID15",
    "LandCoverAI",
    "LEVIRCDPlus",
    "PatternNet",
    "RESISC45",
    "SEN12MS",
    "So2Sat",
    "SpaceNet",
    "SpaceNet1",
    "SpaceNet2",
    "SpaceNet4",
    "TropicalCycloneWindEstimation",
    "UCMerced",
    "VHR10",
    "ZueriCrop",
    # Base classes
    "GeoDataset",
    "RasterDataset",
    "VectorDataset",
    "VisionDataset",
    "VisionClassificationDataset",
    "ZipDataset",
    # Utilities
    "BoundingBox",
    "collate_dict",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.datasets"
