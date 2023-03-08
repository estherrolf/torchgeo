# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""USAVars dataset."""

import glob
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
import torch
from matplotlib.figure import Figure
from torch import Tensor
import shapely.geometry
import shapely.ops
import sys
from rasterio.crs import CRS

from .geo import GeoDataset
from .utils import BoundingBox, download_url, extract_archive

class USAVars(GeoDataset):
    """USAVars dataset.

    The USAVars dataset is reproduction of the dataset used in the paper "`A
    generalizable and accessible approach to machine learning with global satellite
    imagery <https://doi.org/10.1038/s41467-021-24638-z>`_". Specifically, this dataset
    includes 1 sq km. crops of NAIP imagery resampled to 4m/px cenetered on ~100k points
    that are sampled randomly from the contiguous states in the USA. Each point contains
    three continuous valued labels (taken from the dataset released in the paper): tree
    cover percentage, elevation, and population density.

    Dataset format:
    * images are 4-channel GeoTIFFs
    * labels are singular float values

    Dataset labels:
    * tree cover
    * elevation
    * population density

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1038/s41467-021-24638-z

    .. versionadded:: 0.3
    """

    url_prefix = (
        "https://files.codeocean.com/files/verified/"
        + "fa908bbc-11f9-4421-8bd3-72a4bf00427f_v2.0/data/int/applications"
    )
    pop_csv_suffix = "CONTUS_16_640_POP_100000_0.csv?download"
    uar_csv_suffix = "CONTUS_16_640_UAR_100000_0.csv?download"

    data_url = "https://mosaiks.blob.core.windows.net/datasets/uar.zip"
    dirname = "uar"

    md5 = "677e89fd20e5dd0fe4d29b61827c2456"

    label_urls = {
        "housing": f"{url_prefix}/housing/outcomes_sampled_housing_{pop_csv_suffix}",
        "income": f"{url_prefix}/income/outcomes_sampled_income_{pop_csv_suffix}",
        "roads": f"{url_prefix}/roads/outcomes_sampled_roads_{pop_csv_suffix}",
        "nightlights": f"{url_prefix}/nightlights/"
        + f"outcomes_sampled_nightlights_{pop_csv_suffix}",
        "population": f"{url_prefix}/population/"
        + f"outcomes_sampled_population_{uar_csv_suffix}",
        "elevation": f"{url_prefix}/elevation/"
        + f"outcomes_sampled_elevation_{uar_csv_suffix}",
        "treecover": f"{url_prefix}/treecover/"
        + f"outcomes_sampled_treecover_{uar_csv_suffix}",
    }

    split_metadata = {
        "train": {
            "url": "https://mosaiks.blob.core.windows.net/datasets/train_split.txt",
            "filename": "train_split.txt",
            "md5": "3f58fffbf5fe177611112550297200e7",
        },
        "val": {
            "url": "https://mosaiks.blob.core.windows.net/datasets/val_split.txt",
            "filename": "val_split.txt",
            "md5": "bca7183b132b919dec0fc24fb11662a0",
        },
        "test": {
            "url": "https://mosaiks.blob.core.windows.net/datasets/test_split.txt",
            "filename": "test_split.txt",
            "md5": "97bb36bc003ae0bf556a8d6e8f77141a",
        },
    }

    ALL_LABELS = list(label_urls.keys())
    
    # TODO: check on CRS
    crs = CRS.from_epsg(3857)
    res = 4.0 # as per documentation above, images are resampled to 4m per pixel 
    
    p_src_crs = pyproj.CRS("epsg:3857")

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        labels: Sequence[str] = ALL_LABELS,
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
       # res: Optional[float] = None,
       # crs: Optional[CRS] = None,
    ) -> None:
        """Initialize a new USAVars dataset instance.

        Args:
            root: root directory where dataset can be found
            split: train/val/test split to load
            labels: list of labels to include
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if invalid labels are provided
            ImportError: if pandas is not installed
            RuntimeError: if ``download=False`` and data is not found, or checksums
                don't match
        """
        self.root = root
        
        assert split in self.split_metadata
        self.split = split

        for lab in labels:
            assert lab in self.ALL_LABELS

        self.labels = labels
      #  self.transforms = transforms
        self.download = download
        self.checksum = checksum


        self._verify()
        

        super().__init__(transforms)
        
        try:
            import pandas as pd  # noqa: F401
        except ImportError:
            raise ImportError(
                "pandas is not installed and is required to use this dataset"
            )

        self.files = self._load_files()

        self.label_dfs = {
            lab: pd.read_csv(os.path.join(self.root, lab + ".csv"), index_col="ID")
            for lab in self.labels
        }
        
        # merge all label dfs
        all_df = pd.DataFrame({'ID':{}})
        for i, (key, df) in enumerate(self.label_dfs.items()):
            if i == 0:  
                all_df = all_df.merge(df[['lon','lat',key]], on='ID', how='right')
            else:
                all_df = all_df.merge(df[[key]], on='ID', how='inner')
        
        self.all_labels_df = all_df
        
        # TODO: comment
        mint: float = 0
        maxt: float = sys.maxsize
          
        missing_fps = []
        for i, row in all_df.iterrows():
            if i % 1000 == 0: print(i, end=' ')
             
            if i > 2000: continue
            # for now use the NAIP extents
            tif_fp = os.path.join(self.root, self.dirname, f'tile_{row.ID}.tif')
            if os.path.exists(tif_fp):
                with rasterio.open(tif_fp, 'r') as f:
                    bds = f.bounds       
                    minx, maxx = bds.left, bds.right
                    miny, maxy = bds.bottom, bds.top 

                coords = (minx, maxx, miny, maxy, mint, maxt)
                
               # labels = np.array([row[self.labels]])
                try:
                    self.index.insert(i,coords,
                                      {'label_df_id': row.ID,
                                       'img_fn': tif_fp})
                except: 
                    print()
                    print(i, row.ID, coords)
                    print()
            else:
                missing_fps.append(tif_fp)
                
        print(len(missing_fps))

       
    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/labels and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/labels and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """

        hits = self.index.intersection(tuple(query), objects=True)
        tiles_info = cast(List[Dict[str, str]], [hit.object for hit in hits])

        sample = {"crs": self.crs, "bbox": query}
        
        if len(tiles_info) == 0:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )
        elif len(tiles_info) == 1:
            tile_info = tiles_info[0]

            minx, maxx, miny, maxy, mint, maxt = query
            query_box = shapely.geometry.box(minx, miny, maxx, maxy)
            
            img_fp = tile_info['img_fn']
            with rasterio.open(img_fp) as f:
                dst_crs = f.crs.to_string().lower()

                # todo: do we need to transform?
                query_geom = shapely.geometry.mapping(
                    query_box
                )

                img, _ = rasterio.mask.mask(
                    f, [query_geom], crop=True, all_touched=True
                )
                    
            sample["image"] = torch.from_numpy(img).float()
            this_row = self.all_labels_df[self.all_labels_df["ID"] == tile_info["label_df_id"]]
            labels = this_row[self.labels]
            sample["labels"] = torch.from_numpy(np.array(labels)).float()
            centroid_degrees = this_row[['lon','lat']]
            sample["centroid_degrees"] = torch.from_numpy(np.array(centroid_degrees)).float()
            
            
            
            if self.transforms is not None:
                sample = self.transforms(sample)

            return sample
                                        
        else:
            raise IndexError(f"query: {query} spans multiple tiles which is not valid")

   

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def _load_files(self) -> List[str]:
        """Loads file names."""
        with open(os.path.join(self.root, f"{self.split}_split.txt")) as f:
            files = f.read().splitlines()
        return files

    def _load_image(self, path: str) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        with rasterio.open(path) as f:
            array: "np.typing.NDArray[np.int_]" = f.read()
            tensor = torch.from_numpy(array)
            return tensor

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the extracted files already exist
        pathname = os.path.join(self.root, "uar")
        csv_pathname = os.path.join(self.root, "*.csv")
        split_pathname = os.path.join(self.root, "*_split.txt")

        csv_split_count = (len(glob.glob(csv_pathname)), len(glob.glob(split_pathname)))
        if glob.glob(pathname) and csv_split_count == (7, 3):
            return

        # Check if the zip files have already been downloaded
        pathname = os.path.join(self.root, self.dirname + ".zip")
        if glob.glob(pathname) and csv_split_count == (7, 3):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                f"Dataset not found in `root={self.root}` and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automatically download the dataset."
            )

        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        for f_name in self.label_urls:
            download_url(self.label_urls[f_name], self.root, filename=f_name + ".csv")

        download_url(self.data_url, self.root, md5=self.md5 if self.checksum else None)

        for metadata in self.split_metadata.values():
            download_url(
                metadata["url"],
                self.root,
                md5=metadata["md5"] if self.checksum else None,
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        extract_archive(os.path.join(self.root, self.dirname + ".zip"))
                              
    def _centroid_to_square_vertices(self, lon: float, lat: float, 
                                     zoom_orig_imgs: int = 16, 
                                     numpix_orig_imgs: int = 640 ) -> list:
        """ Translate centroid latlon (in degrees) to bounding box.

        Rewrite/translation of centroidsToSquareVerticies function in R_utils.R of 
        https://github.com/Global-Policy-Lab/mosaiks-paper. 
        
        Args: TODO""" 
        
        # same function and constants as defined in https://github.com/Global-Policy-Lab/mosaiks-paper
        meters_per_pixel = lambda lat_in: 156543.03392 * cos(lat_in * np.pi / 180.) / (2**zoom_orig_imgs)
        m_degree = 111 * 1000                      
        tile_width_meters = meters_per_pixel(lat) * numpix_orig_imgs
        
        dely = tile_width_meters / m_degree / 2.
        delx = tile_width_meters / m_degree / np.cos(lat*np.pi/180) / 2.
                              
        return [lon - delx, lat - dely, lon + delx, lat + dely]  
    
    def _p_transformer(self, crs: str) -> pyproj.Transformer:
    
        p_transformer = pyproj.Transformer.from_crs(
                self.p_src_crs, pyproj.CRS(crs), always_xy=True
            ).transform
        return p_transformer

    def plot(
        self,
        sample: Dict[str, Tensor],
        show_labels: bool = True,
        suptitle: Optional[str] = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_labels: flag indicating whether to show labels above panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        
        image = sample["image"][:3].numpy()  # get RGB inds
        image = np.moveaxis(image, 0, 2)

        fig, axs = plt.subplots(figsize=(10, 10))
        axs.imshow(image)
        axs.axis("off")

        if show_labels:
            labels = sample['labels'].numpy().astype('float')[0]
            title = ""
            for l,v in zip(self.labels, labels):
                title += f'{l}={v:.2f}, '
            axs.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)
            
            labels= s['labels'].numpy().astype('float')[0]

        return fig
