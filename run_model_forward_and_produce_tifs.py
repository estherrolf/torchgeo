import argparse
import os

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torchgeo.models import FCN_modified
#from TileDatasets import TileInferenceDataset
from torchgeo.datasets import TileInferenceDataset
NUM_CLASSES = 4
NUM_FILTERS = 256

NUM_WORKERS = 4
CHIP_SIZE = 128
PADDING = 64
assert PADDING % 2 == 0
HALF_PADDING = PADDING // 2
CHIP_STRIDE = CHIP_SIZE - PADDING


# Modified from script from Caleb to run model forward and produce tifs

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_fn",
        type=str,
        required=True,
        help='The path to a 4 channel NAIP GeoTIFF.',
    )
    parser.add_argument(
        "--model_fn",
        type=str,
        required=True,
        help="Path to the model checkpoint to use. Should match the type specified by `--model`.",
    )
    parser.add_argument(
        "--output_fn",
        type=str,
        required=True,
        help="The path to output the model predictions (as a GeoTIFF). Will fail if this file already exists.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Flag for overwriting `--output_fn` if that file already exists.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="The ID of the GPU to use")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size to use during inference."
    )
    parser.add_argument(
        "--model", default="fcn-modified", choices=("fcn-modified", "otherstuff"), help="Model to use"
    )

    return parser


def trim_state_dict(state_dict):
    new_state_dict = dict()
    for k, v in state_dict.items():
        if k.startswith("model."):
            k = k[6:]
        new_state_dict[k] = v
    return new_state_dict


def image_transforms(img):
    """Gets a unormalized numpy image in HxWxC format, returns ready-to-go Tensor."""
    img = img / 255.0
    img = np.rollaxis(img, 2, 0).astype(np.float32)
    img = torch.from_numpy(img)
    return img


def run_through_tiles(model_ckpt_fn,
                     input_fns,
                     output_fns,
                     model='fcn-modified',
                     batch_size=128,
                     gpu = 0,
                     overwrite=False):

    ## Sanity checking
    for input_fn in input_fns:
        assert os.path.exists(input_fn)
    if not overwrite:
        for output_fn in output_fns:
            assert not os.path.exists(output_fn)

    ## GPU
    device = torch.device(f"cuda:{gpu}")

    ## Load model
    if model == "fcn-modified":
        model = FCN_modified(4, classes=NUM_CLASSES, num_filters=NUM_FILTERS)
    else:
        raise ValueError(f"Model {model} not recognized")

    checkpoint = torch.load(model_ckpt_fn, map_location="cpu")
    model.load_state_dict(trim_state_dict(checkpoint["state_dict"]))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)

    # Run the below code for each input file -- currently only one
    for i,(input_fn, output_fn) in enumerate(zip(input_fns, output_fns)):
        print(f'{i} of {len(input_fns)}')
        ## Setup dataloader
        dataset = TileInferenceDataset(
            input_fn,
            chip_size=CHIP_SIZE,
            stride=CHIP_STRIDE,
            transform=image_transforms,
            verbose=False,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        ## Peak at the input file to see what size the output should be
        with rasterio.open(input_fn) as f:
            input_width, input_height = f.width, f.height
            input_profile = f.profile.copy()

        ## Model inference happens here
        output = np.zeros((NUM_CLASSES, input_height, input_width), dtype=np.float32)
        counts = np.zeros((input_height, input_width), dtype=np.float32)

        for i, (data, coords) in enumerate(dataloader):
            data = data.to(device)
            with torch.no_grad():
                t_output = model(data)
                t_output = torch.exp(t_output).cpu().numpy().squeeze()
           #     print(t_output.shape)

            for j in range(t_output.shape[0]):
                y, x = coords[j]

                output[:, y : y + CHIP_SIZE, x : x + CHIP_SIZE] += t_output[j]
                counts[y : y + CHIP_SIZE, x : x + CHIP_SIZE] += 1

        output = output / counts
        output_hard = output.argmax(axis=0).astype(np.uint8)
        ## Save output
        output_profile = input_profile.copy()
        output_profile["driver"] = "GTiff"
        output_profile["dtype"] = "uint8"
        output_profile["count"] = 4
        output_profile["nodata"] = None

#         print(output[:,:2,:2])
#         print(output_hard.shape)
        
        print(f'writing to: {output_fn}')
        with rasterio.open(output_fn, "w", **output_profile) as f:
            f.write((output * 255.).astype(np.uint8))
            
            
        output_profile["count"] = 1
        output_fn_hard = output_fn.replace('.tif','_argmaxed.tif')
        print(f'writing to: {output_fn_hard}')
        with rasterio.open(output_fn_hard, "w", **output_profile) as f:
            f.write(output_hard, 1)
            f.write_colormap(
                1,
                {
                    0: (0, 197, 255, 255),
                    1: (38, 115, 0, 255),
                    2: (163, 255, 115, 255),
                    3: (156, 156, 156, 255),
                    4: (0, 0, 0, 0)
                },
            )


if __name__ == "__main__":

    parser = setup_parser()
    args = parser.parse_args()

    main(args)
