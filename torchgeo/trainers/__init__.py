# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo trainers."""

from .chesapeake import ChesapeakeCVPRDataModule, ChesapeakeCVPRSegmentationTask
from .chesapeake_learn_on_prior import ChesapeakeCVPRPriorDataModule, ChesapeakeCVPRPriorSegmentationTask
from .chesapeake_learn_the_prior import ChesapeakeCVPRLearnPriorDataModule, ChesapeakeCVPRLearnPriorTask
from .cyclone import CycloneDataModule, CycloneSimpleRegressionTask
from .enviroatlas import EnviroatlasDataModule, EnviroatlasSegmentationTask
from .enviroatlas_learn_on_prior import EnviroatlasPriorDataModule, EnviroatlasPriorSegmentationTask
from .enviroatlas_learn_the_prior import EnviroatlasLearnPriorDataModule, EnviroatlasLearnPriorTask

from .landcoverai import LandcoverAIDataModule, LandcoverAISegmentationTask
from .naipchesapeake import NAIPChesapeakeDataModule, NAIPChesapeakeSegmentationTask
from .sen12ms import SEN12MSDataModule, SEN12MSSegmentationTask

__all__ = (
    "ChesapeakeCVPRPriorSegmentationTask",
    "ChesapeakeCVPRPriorDataModule",
    "ChesapeakeCVPRSegmentationTask",
    "ChesapeakeCVPRDataModule",
    "ChesapeakeCVPRLearnPriorDataModule",
    "ChesapeakeCVPRLearnPriorTask",
    "EnviroatlasSegmentationTask",
    "EnviroatlasDataModule",
    "EnviroatlasPriorSegmentationTask",
    "EnviroatlasPriorDataModule",
    "EnviroatlasLearnPriorDataModule", 
    "EnviroatlasLearnPriorTask",
    "ChesapeakeCVPRDataModule",
    "CycloneDataModule",
    "CycloneSimpleRegressionTask",
    "LandcoverAIDataModule",
    "LandcoverAISegmentationTask",
    "NAIPChesapeakeDataModule",
    "NAIPChesapeakeSegmentationTask",
    "SEN12MSDataModule",
    "SEN12MSSegmentationTask",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.trainers"
