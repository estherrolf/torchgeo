# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Trainers for the Chesapeake datasets."""

from typing import Any, Callable, Dict, Optional, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor
from torch.nn.modules import Module
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore[attr-defined]
from torchmetrics import Accuracy, IoU, MetricCollection
from torchvision.transforms import Compose


from ..datasets import ChesapeakeCVPRWeakInput
from ..samplers import GridGeoSampler, RandomBatchGeoSampler
from ..models import FCN_modified,FCN_larger_modified
# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"
Module.__module__ = "torch.nn"

import sys
sys.path.append('/home/esther/lc-mapping/scripts')
import landcover_definitions as lc


# CMAP = matplotlib.colors.ListedColormap(
#     [np.array(Chesapeake7.cmap[i + 1]) / 255.0 for i in range(6)]
# )


class ChesapeakeCVPRLearnPriorTask(LightningModule):
    """LightningModule for training models on the Chesapeake CVPR Land Cover dataset.

    This allows using arbitrary models and losses from the
    ``pytorch_segmentation_models`` package.
    """

    def config_task(self, kwargs: Dict[str, Any]) -> None:
        
        if kwargs['condense_barren']:
            n_classes = 4
            # need the n_classes_with_nodata for the padding in test set
            self.n_classes_with_nodata = n_classes + 1
            self.ignore_index = 4 
            
            self.lc_type = 'chesapeake_4_no_zeros'
            
            
        elif kwargs['condense_road_and_impervious']:
            n_classes = 5
            self.lc_type = 'chesapeake_5'
            self.ignore_index = None
        

        print('lc type is ', self.lc_type)
        self.n_classes = n_classes
            
        # four from the blurred NLCD remapped to CC4, plus roads, buildings, waterways and waterbodies
        self.in_channels = 8 
        self.need_to_exp_outputs = False
        self.labels_are_onehot = False
        
        """Configures the task based on kwargs parameters."""
        if kwargs["segmentation_model"] == "fcn":
            self.model = FCN_modified(
                in_channels=self.in_channels,
                classes=self.n_classes,
                num_filters=256,
                output_smooth=kwargs['output_smooth'],
                log_outputs=True
            )
        elif kwargs["segmentation_model"] == "fcn_larger":
            self.model = FCN_larger_modified(
                in_channels=self.in_channels,
                classes=self.n_classes,
                num_filters=256,
                output_smooth=kwargs['output_smooth'],
                log_outputs=True
            )
        else:
            raise ValueError(
                f"Model type '{kwargs['segmentation_model']}' is not valid."
            )

        if kwargs["loss"] == "nll":
            self.loss = nn.NLLLoss() 
            
        elif kwargs["loss"] == "ce":
            self.loss = nn.CrossEntropyLoss()
            
        elif kwargs["loss"] == "jaccard":
            self.loss = smp.losses.JaccardLoss(mode="multiclass")
            
        elif kwargs["loss"] == "l2":
            self.loss = nn.MSELoss(reduction='mean') 
            self.need_to_exp_outputs = True
            self.labels_are_onehot = True
            
        else:
            raise ValueError(f"Loss type '{kwargs['loss']}' is not valid.")
            
        print(self.loss)
        
        
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Initialize the LightningModule with a model and loss function.

        Keyword Args:
            segmentation_model: Name of the segmentation model type to use
            encoder_name: Name of the encoder model backbone to use
            encoder_weights: None or "imagenet" to use imagenet pretrained weights in
                the encoder model
            loss: Name of the loss function
        """
        super().__init__()
        self.save_hyperparameters()  # creates `self.hparams` from kwargs

        self.config_task(kwargs)

        self.train_accuracy = Accuracy(num_classes=self.n_classes_with_nodata,ignore_index=self.ignore_index)
        self.val_accuracy = Accuracy(num_classes=self.n_classes_with_nodata,ignore_index=self.ignore_index)
        self.test_accuracy = Accuracy(num_classes=self.n_classes_with_nodata,ignore_index=self.ignore_index)

        self.train_iou = IoU(num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index)
        self.val_iou = IoU(num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index)
        self.test_iou = IoU(num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index)

    def forward(self, x: Tensor) -> Any:  # type: ignore[override]
        """Forward pass of the model."""
        return self.model(x)

    def training_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tensor:
        """Training step - reports average accuracy and average IoU."""
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        if self.labels_are_onehot:
            self.train_accuracy(y_hat_hard, y.argmax(dim=1))
            self.train_iou(y_hat_hard, y.argmax(dim=1))
        else:
            self.train_accuracy(y_hat_hard, y)
            self.train_iou(y_hat_hard, y)
        return cast(Tensor, loss)

    def training_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level training metrics."""
        self.log("train_acc_q", self.train_accuracy.compute())
        self.log("train_iou_q", self.train_iou.compute())
        self.train_accuracy.reset()
        self.train_iou.reset()

    def validation_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Validation step - reports average accuracy and average IoU."""
        x = batch["image"]
        y = batch["mask"]
        
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the test and validation steps only log per *epoch*
        self.log("val_loss", loss)
        if self.labels_are_onehot:
            self.val_accuracy(y_hat_hard, y.argmax(dim=1))
            self.val_iou(y_hat_hard, y.argmax(dim=1))
        else:
            self.val_accuracy(y_hat_hard, y)
            self.val_iou(y_hat_hard, y)
            
        if batch_idx < 10:
            # Render the image, ground truth mask, and predicted mask for the first
            # image in the batch
            inputs = batch["image"][0].cpu().numpy()
            mask = batch["mask"][0].cpu().numpy()
            q = torch.exp(y_hat[0]).cpu().numpy()
            
            # squish the input layers of the image according to this assumption:
            #[f"nlcd_onehot_blurred_kernelsize_{nlcd_blur_kernelsize}_sigma_{nlcd_blur_sigma}", 
            #           "buildings", "roads", "waterbodies", "waterways", "lc"]
            squished_layers = inputs.copy()
            squished_layers[3] += inputs[4:6].sum(axis=0) # buildings and roads
            squished_layers[0] += inputs[6:8].sum(axis=0) # water
            
            input_vis = lc.vis_lc(squished_layers[:4]/squished_layers[:4].sum(axis=0), 
                                  self.lc_type).T.swapaxes(0,1)
            pred_vis = lc.vis_lc(q, self.lc_type).T.swapaxes(0,1)
            label_vis = lc.vis_lc(mask, self.lc_type).T.swapaxes(0,1)
     #       print(pred_vis.min(), pred_vis.max())
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            
            # WHAT to plot
            
            axs[0].imshow(input_vis, interpolation="none")
            axs[0].axis("off")
            
            axs[1].imshow(pred_vis, interpolation="none")
            axs[1].axis("off")
            
            axs[2].imshow(label_vis, interpolation="none")
            axs[2].axis("off")
            
            plt.tight_layout()

            # the SummaryWriter is a tensorboard object, see:
            # https://pytorch.org/docs/stable/tensorboard.html#
            summary_writer: SummaryWriter = self.logger.experiment
            summary_writer.add_figure(
                f"image/{batch_idx}", fig, global_step=self.global_step
            )
            plt.close()

    def validation_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level validation metrics."""
        self.log("val_acc_q", self.val_accuracy.compute())
        self.log("val_iou_q", self.val_iou.compute())
        self.val_accuracy.reset()
        self.val_iou.reset()

    def test_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Test step identical to the validation step."""
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the test and validation steps only log per *epoch*
        self.log("test_loss", loss)
        
        if self.labels_are_onehot:
            self.test_accuracy(y_hat_hard, y.argmax(dim=1))
            self.test_iou(y_hat_hard, y.argmax(dim=1))
        else:
            self.test_accuracy(y_hat_hard, y)
            self.test_iou(y_hat_hard, y)
            
    def test_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level test metrics."""
        self.log("test_acc_q", self.test_accuracy.compute())
        self.log("test_iou_q", self.test_iou.compute())
        self.test_accuracy.reset()
        self.test_iou.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams["learning_rate"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    patience=self.hparams["learning_rate_schedule_patience"],
                ),
                "monitor": "val_loss",
                "verbose": True,
            },
        }


class ChesapeakeCVPRLearnPriorDataModule(LightningDataModule):
    """LightningDataModule implementation for the Chesapeake CVPR Land Cover dataset.

    Uses the random splits defined per state to partition tiles into train, val,
    and test sets.
    """

    def __init__(
        self,
        root_dir: str,
        states_str: str,
        patches_per_tile: int = 200,
        patch_size: int = 256,
        batch_size: int = 64,
        num_workers: int = 4,
        condense_barren: bool = True,
        condense_road_and_impervious: bool = True,
        onehot_encode_labels: bool = False,
        nlcd_blur_kernelsize: int = 101,
        nlcd_blur_sigma: int = 15,
        train_set: str = "train",
        val_set: str = "val",
        test_set: str = "test",
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for Chesapeake CVPR based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the ChesapeakeCVPR Dataset
                classes
            train_state: The state code to use to train the model, e.g. "ny"
            patches_per_tile: The number of patches per tile to sample
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
        """
        super().__init__()  # type: ignore[no-untyped-call]
        
        states = states_str.split('+')
        for state in states:
            assert state in ["md", "de", "ny", "pa", "va", "wv"]

        
        self.root_dir = root_dir
        self.layers = [f"prior_from_cooccurrences_{nlcd_blur_kernelsize}_{nlcd_blur_sigma}_no_osm_no_buildings",
                       "buildings", "roads", "waterbodies", "waterways", "lc"]
        
        self.num_nlcd_layers = 4
        self.patches_per_tile = patches_per_tile
        self.patch_size = patch_size
        self.original_patch_size = 512
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.condense_road_and_impervious = condense_road_and_impervious
        self.condense_barren = condense_barren
        self.onehot_encode_labels = onehot_encode_labels
        print(self.onehot_encode_labels)
        
        if condense_barren:
            self.n_classes = 4
        
        self.train_sets = [f"{state}-{train_set}" for state in states]
        self.val_sets = [f"{state}-{val_set}" for state in states]
        self.test_sets = [f"{state}-{test_set}" for state in states]
        print(f'train sets are: {self.train_sets}')
        print(f'val sets are: {self.val_sets}')
        print(f'test sets are: {self.test_sets}')
        
    def pad_to(
        self, size: int = 512, image_value: int = 0, mask_value: int = 0
    ) -> Callable[[Dict[str, Tensor]], Dict[str, Tensor]]:
        """Returns a function to perform a padding transform on a single sample."""

        def pad_inner(sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
            _, height, width = sample["image"].shape
            assert height <= size and width <= size

            height_pad = size - height
            width_pad = size - width

            # See https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            # for a description of the format of the padding tuple
            sample["image"] = F.pad(
                sample["image"],
                (0, width_pad, 0, height_pad),
                mode="constant",
                value=image_value,
            )
            sample["mask"] = F.pad(
                sample["mask"],
                (0, width_pad, 0, height_pad),
                mode="constant",
                value=mask_value,
            )
            return sample

        return pad_inner
    
    def center_crop(
        self, size: int = 512
    ) -> Callable[[Dict[str, Tensor]], Dict[str, Tensor]]:
        """Returns a function to perform a center crop transform on a single sample."""

        def center_crop_inner(sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
            _, height, width = sample["image"].shape
            assert height >= size and width >= size

            y1 = (height - size) // 2
            x1 = (width - size) // 2
            sample["image"] = sample["image"][:, y1 : y1 + size, x1 : x1 + size]
            sample["mask"] = sample["mask"][:, y1 : y1 + size, x1 : x1 + size]

            return sample

        return center_crop_inner
    
    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocesses a single sample."""
    
        # sample['image'] contains the weak inputs, sample['mask'] is the lc
        
        # normalize just the NLCD layers because they get stored as 0...255
        sample["image"] = sample["image"].float()
        sample["image"][:self.num_nlcd_layers] = sample["image"][:self.num_nlcd_layers] / 255.0
    
        # handle reindexing the labels
        if self.condense_barren:
            
            # Now remap all the highres labels
            impervious_idxs_highres_orig = [4,5,6]
            impervious_idx_condesed = 4
            for c_idx in impervious_idxs_highres_orig:
                sample["mask"][sample["mask"] == c_idx] = impervious_idx_condesed
            
            nodata_idx_pad_orig = 7
            nodata_idx_pad_condensed = 5 # note that we're about to subtract one
            
            sample["mask"][sample["mask"] == nodata_idx_pad_orig] = nodata_idx_pad_condensed
            # subtract 1 so it starts with 0
            sample["mask"] = sample["mask"] - 1
            
        elif self.condense_road_and_impervious:
            impervious_idx = 5
            road_idx = 6
            # recast road idx to impervious
            sample['mask'][sample['mask'] == road_idx] = impervious_idx
            # subtract 1
            sample['mask'] = sample['mask'] - 1
            

        if self.onehot_encode_labels:
            sample["mask"] = nn.functional.one_hot(sample['mask'].to(torch.int64),
                                                   num_classes=self.n_classes).transpose(0,2).transpose(1,2)    
    
        
        sample["mask"] = sample["mask"].squeeze().long()
        
        return sample

    def prepare_data(self) -> None:
        """Confirms that the dataset is downloaded on the local node.
        This method is called once per node, while :func:`setup` is called once per GPU.
        """
        ChesapeakeCVPRWeakInput(
            self.root_dir,
            splits=self.train_sets,
            layers=self.layers,
            transforms=None,
            download=False,
            checksum=False,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Create the train/val/test splits based on the original Dataset objects.

        The splits should be done here vs. in :func:`__init__` per the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#setup.
        """
        train_transforms = Compose(
            [
                self.center_crop(self.patch_size),
                self.preprocess,
            ]
        )
        val_transforms = Compose(
            [
                self.center_crop(self.patch_size),
                self.preprocess,
            ]
        )
        test_transforms = Compose(
            [
                self.pad_to(self.original_patch_size, image_value=0, mask_value=7),
                self.preprocess,
            ]
        )
        
        
        print('training on ',self.train_sets)
        self.train_dataset = ChesapeakeCVPRWeakInput(
            self.root_dir,
            splits=self.train_sets,
            layers=self.layers,
            transforms=train_transforms,
            download=False,
            checksum=False,
        )
        self.val_dataset = ChesapeakeCVPRWeakInput(
            self.root_dir,
            splits=self.val_sets,
            layers=self.layers,
            transforms=val_transforms,
            download=False,
            checksum=False,
        )
        self.test_dataset = ChesapeakeCVPRWeakInput(
            self.root_dir,
            splits=self.test_sets,
            layers=self.layers,
            transforms=test_transforms,
            download=False,
            checksum=False,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training."""
        print('train set of size ',  self.train_dataset.index.get_size())
        sampler = RandomBatchGeoSampler(
            self.train_dataset.index,
            size=self.original_patch_size,
            batch_size=self.batch_size,
            length=self.patches_per_tile * self.train_dataset.index.get_size() 
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,  # type: ignore[arg-type]
            num_workers=self.num_workers,
         #   pin_memory=False,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation."""
        sampler = RandomBatchGeoSampler(
                self.val_dataset.index,
                size=self.original_patch_size,
                batch_size=self.batch_size,
                length=self.patches_per_tile * self.val_dataset.index.get_size() // 2,
        )
        return DataLoader(
                self.val_dataset,
                batch_sampler=sampler,  # type: ignore[arg-type]
                num_workers=self.num_workers,
        )
    
    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing."""
        sampler = GridGeoSampler(
            self.test_dataset.index,
            size=self.original_patch_size,
            stride=self.original_patch_size,
        )
        return DataLoader(
            self.test_dataset,
            batch_size=16,
            sampler=sampler,  # type: ignore[arg-type]
            num_workers=self.num_workers,
  #          pin_memory=False,
        )