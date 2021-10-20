# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Modified from Chesapeake by Esther

"""Enviroatlas High-Resolution Land Cover Project datasets."""

import abc
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence

import fiona
import numpy as np
import pyproj
import rasterio
import rasterio.mask
import shapely.geometry
import shapely.ops
import torch
from rasterio.crs import CRS

from .geo import GeoDataset, RasterDataset
from .utils import BoundingBox, check_integrity, download_and_extract_archive

ENVIROATLAS_CLASS_DEFINITIONS = {
    0: ("Unclassified"),
    10: ("Water"),
    20: ("Impervious Surface"),
    30: ("Soil and Barren"),
    40: ("Trees and Forest"),
    52: ("Shrubs"),
    70: ("Grass and Herbaceous"),
    80: ("Agriculture"),
    82: ("Orchards"),
    91: ("Woody Wetlands"),
    92: ("Emergent Wetlands")
}

def create_map_raw_lc_to_idx(class_definitions):
    """Map raw landcover values to contiguous index values starting with 0."""
    LC_TO_IDX = np.zeros(np.array(list(class_definitions.keys())).max() + 1, dtype=np.uint8)
    for i, k in enumerate(class_definitions.keys()):
        LC_TO_IDX[k] = i
    return LC_TO_IDX    

# this is the only thing we take from lc
map_raw_enviroatlas_to_idx = create_map_raw_lc_to_idx(ENVIROATLAS_CLASS_DEFINITIONS)

class Enviroatlas_without_prior(GeoDataset):
    """Enviroatlas dataset covering four cities.
    
    Highres label data from EPA envioratlas dataset from 
    https://edg.epa.gov/metadata/catalog/search/resource/details.page? +
    uuid=%7Badf673a0-11b4-40d6-befd-8bf75b370cba%7D.
    roads, waterways, waterbodies, and buildings fused as detailed in {todo cite other repo}.
    """

    crs = CRS.from_epsg(3857)
    res = 1

    valid_layers = [
        "a_naip",
        "b_nlcd",
        "c_roads",
        "d_water",
        "d1_waterways",
        "d2_waterbodies",
        "e_buildings",
        "h_highres_labels",
        "prior_from_cooccurrences_101_31",
    ]
    
    states = ["pittsburgh_pa-2010_1m",
              "durham_nc-2012_1m",
              "austin_tx-2012_1m",
              "phoenix_az-2010_1m"] # dodo add states as they get ready
    
    # only pittsburch has a train and val set
    splits = (
        [f"{state}-train" for state in states[:1]]
        + [f"{state}-val" for state in states[:1]]
        + [f"{state}-test" for state in states]
    )

    p_src_crs = pyproj.CRS("epsg:3857")
    p_transformers = {
        "epsg:26917": pyproj.Transformer.from_crs(
            p_src_crs, pyproj.CRS("epsg:26917"), always_xy=True
        ).transform,
        "epsg:26918": pyproj.Transformer.from_crs(
            p_src_crs, pyproj.CRS("epsg:26918"), always_xy=True
        ).transform,
        "epsg:26914": pyproj.Transformer.from_crs(
            p_src_crs, pyproj.CRS("epsg:26914"), always_xy=True
        ).transform,
        "epsg:26912": pyproj.Transformer.from_crs(
            p_src_crs, pyproj.CRS("epsg:26912"), always_xy=True
        ).transform,
    }

    def __init__(
        self,
        root: str = "data",
        splits: Sequence[str] = ["pittsburgh_pa-2010_1m-train"],
        layers: List[str] = ["a_naip", "h_highres_labels"],
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            split: a string in the format "{state}-{train,val,test}" indicating the
                subset of data to use, for example "ny-train"
            layers: a list containing a subset of "naip-new", "naip-old", "lc", "nlcd",
                "landsat-leaf-on", "landsat-leaf-off", "buildings" indicating which
                layers to load
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            FileNotFoundError: if no files are found in ``root``
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        for split in splits:
            if not split in self.splits: print(split)
            assert split in self.splits
        assert all([layer in self.valid_layers for layer in layers])
        super().__init__(transforms)  # creates self.index and self.transform
        self.root = root
        self.layers = layers
        self.cache = cache
        self.checksum = checksum
        
        self.need_to_reindex_enviroatlas_labels = True

        if download:
            self._download()

        if checksum:
            if not self._check_integrity():
                raise RuntimeError(
                    "Dataset not found or corrupted. "
                    + "You can use download=True to download it"
                )

        # Add all tiles into the index in epsg:3857 based on the included geojson
        mint: float = 0
        maxt: float = sys.maxsize
        with fiona.open(os.path.join(root, "spatial_index.geojson"), "r") as f:
            for i, row in enumerate(f):
                if row["properties"]["split"] == split:
                    box = shapely.geometry.shape(row["geometry"])
                    minx, miny, maxx, maxy = box.bounds
                    coords = (minx, maxx, miny, maxy, mint, maxt)
                    self.index.insert(
                        i,
                        coords,
                        {
                            # TODO FIX for enviroatlas
                            "a_naip": row["properties"]["a_naip"],
                            "b_nlcd": row["properties"]["b_nlcd"],
                            "c_roads": row["properties"]["c_roads"],
                            "d_water": row["properties"]["d_water"],
                            "d1_waterways": row["properties"]["d1_waterways"],
                            "d2_waterbodies": row["properties"]["d2_waterbodies"],
                            "e_buildings": row["properties"]["e_buildings"],
                            "h_highres_labels": row["properties"]["h_highres_labels"],
                            "prior_from_cooccurrences_101_31":row["properties"]['a_naip'].replace('a_naip',
                                                                                                        'prior_from_cooccurrences_101_31')
                        },
                    )

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(query, objects=True)
        filepaths = [hit.object for hit in hits]

        sample = {
            "image": [],
            "mask": [],
            "crs": self.crs,
            "bbox": query,
        }

        if len(filepaths) == 0:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )
        elif len(filepaths) == 1:
            filenames = filepaths[0]
            query_geom_transformed = None  # is set by the first layer

            minx, maxx, miny, maxy, mint, maxt = query
            query_box = shapely.geometry.box(minx, miny, maxx, maxy)

            for layer in self.layers:
    
                fn = filenames[layer]

                with rasterio.open(os.path.join(self.root, fn)) as f:
                    dst_crs = f.crs.to_string().lower()

                    if query_geom_transformed is None:
                        query_box_transformed = shapely.ops.transform(
                            self.p_transformers[dst_crs], query_box
                        ).envelope
                        query_geom_transformed = shapely.geometry.mapping(
                            query_box_transformed
                        )

                    data, _ = rasterio.mask.mask(
                        f, [query_geom_transformed], crop=True, all_touched=True
                    )

                if layer in [
                    "a_naip", 
                    "prior_from_cooccurrences_101_31"
                ]:
                    sample["image"].append(data)
                elif layer in ["h_highres_labels"]:
                    # reindex the enviroatlas labels if they're not already 0-10 index
                    if layer == "h_highres_labels" and self.need_to_reindex_enviroatlas_labels: 
                        data = map_raw_enviroatlas_to_idx[data]
                        sample["mask"].append(data)
        else:
            raise IndexError(f"query: {query} spans multiple tiles which is not valid")

        sample["image"] = np.concatenate(  # type: ignore[no-untyped-call]
            sample["image"], axis=0
        )
        sample["mask"] = np.concatenate(  # type: ignore[no-untyped-call]
            sample["mask"], axis=0
        )

        sample["image"] = torch.from_numpy(  # type: ignore[attr-defined]
            sample["image"]
        )
        sample["mask"] = torch.from_numpy(sample["mask"])  # type: ignore[attr-defined]

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
     
class Enviroatlas(GeoDataset):
    """Enviroatlas dataset covering four cities with prior and weak input data layers.
    
    Highres label data from EPA envioratlas dataset from 
    https://edg.epa.gov/metadata/catalog/search/resource/details.page? +
    uuid=%7Badf673a0-11b4-40d6-befd-8bf75b370cba%7D.
    roads, waterways, waterbodies, and buildings fused as detailed in {todo cite other repo}.
    """


    crs = CRS.from_epsg(3857)
    res = 1

    valid_prior_layers = ['prior_from_cooccurrences_101_31']

    valid_layers = [
        "a_naip",
        "b_nlcd",
        "c_roads",
        "d_water",
        "d1_waterways",
        "d2_waterbodies",
        "e_buildings",
        "h_highres_labels",
    ] + valid_prior_layers
    
    need_to_reindex_enviroatlas_labels = True

    states = ["pittsburgh_pa-2010_1m",
              "durham_nc-2012_1m",
              "austin_tx-2012_1m",
              "phoenix_az-2010_1m"] # dodo add states as they get ready
    
    # only pittsburch has a train and val set
    splits = (
        [f"{state}-train" for state in states[:1]]
        + [f"{state}-val" for state in states[:1]]
        + [f"{state}-test" for state in states]
    )

    p_src_crs = pyproj.CRS("epsg:3857")
    p_transformers = {
        "epsg:26917": pyproj.Transformer.from_crs(
            p_src_crs, pyproj.CRS("epsg:26917"), always_xy=True
        ).transform,
        "epsg:26918": pyproj.Transformer.from_crs(
            p_src_crs, pyproj.CRS("epsg:26918"), always_xy=True
        ).transform,
        "epsg:26914": pyproj.Transformer.from_crs(
            p_src_crs, pyproj.CRS("epsg:26914"), always_xy=True
        ).transform,
        "epsg:26912": pyproj.Transformer.from_crs(
            p_src_crs, pyproj.CRS("epsg:26912"), always_xy=True
        ).transform,
    }

    def __init__(
        self,
        root: str = "data",
        splits: Sequence[str] = ["pittsburgh_pa-2010_1m-train"],
        layers: List[str] = ["a_naip",'prior_whole_city_cooccurrences_101_31'],
        prior_as_input = False,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            split: a string in the format "{state}-{train,val,test}" indicating the
                subset of data to use, for example "ny-train"
            layers: a list containing a subset of "naip-new", "naip-old", "lc", "nlcd",
                "landsat-leaf-on", "landsat-leaf-off", "buildings" indicating which
                layers to load
            prior_as_input: bool describing whether the prior is used as an input (True)
                or as supervision (False). 
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            FileNotFoundError: if no files are found in ``root``
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        for split in splits:
            if not split in self.splits: print(split)
            assert split in self.splits
                     
        for layer in layers:
            if not layer in self.valid_layers: print(f'{layer} not in {self.valid_layers}')
        assert all([layer in self.valid_layers for layer in layers])
        
        super().__init__(transforms)  # creates self.index and self.transform
        self.root = root
        self.layers = layers
        self.cache = cache
        self.checksum = checksum
        
        self.prior_as_input = prior_as_input

        if download:
            self._download()

        if checksum:
            if not self._check_integrity():
                raise RuntimeError(
                    "Dataset not found or corrupted. "
                    + "You can use download=True to download it"
                )

        # Add all tiles into the index in epsg:3857 based on the included geojson
        mint: float = 0
        maxt: float = sys.maxsize
        with fiona.open(os.path.join(root, "spatial_index.geojson"), "r") as f:
            for i, row in enumerate(f):
                if row["properties"]["split"] == split:
                    box = shapely.geometry.shape(row["geometry"])
                    minx, miny, maxx, maxy = box.bounds
                    coords = (minx, maxx, miny, maxy, mint, maxt)
                    self.index.insert(
                        i,
                        coords,
                        {
                            "a_naip": row["properties"]["a_naip"],
                            "b_nlcd": row["properties"]["b_nlcd"],
                            "c_roads": row["properties"]["c_roads"],
                            "d_water": row["properties"]["d_water"],
                            "d1_waterways": row["properties"]["d1_waterways"],
                            "d2_waterbodies": row["properties"]["d2_waterbodies"],
                            "e_buildings": row["properties"]["e_buildings"],
                            "h_highres_labels": row["properties"]["h_highres_labels"],
                            "prior_from_cooccurrences_101_31":row["properties"]["a_naip"].replace('a_naip',
                                                                                                       "prior_from_cooccurrences_101_31")
                        },
                    )

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(query, objects=True)
        filepaths = [hit.object for hit in hits]

        sample = {
            "image": [],
            "mask": [],
            "crs": self.crs,
            "bbox": query,
        }

        if len(filepaths) == 0:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )
        elif len(filepaths) == 1:
            filenames = filepaths[0]
            query_geom_transformed = None  # is set by the first layer

            minx, maxx, miny, maxy, mint, maxt = query
            query_box = shapely.geometry.box(minx, miny, maxx, maxy)

            for layer in self.layers:

                fn = filenames[layer]

                with rasterio.open(os.path.join(self.root, fn)) as f:
                    dst_crs = f.crs.to_string().lower()

                    if query_geom_transformed is None:
                        query_box_transformed = shapely.ops.transform(
                            self.p_transformers[dst_crs], query_box
                        ).envelope
                        query_geom_transformed = shapely.geometry.mapping(
                            query_box_transformed
                        )

                    data, _ = rasterio.mask.mask(
                        f, [query_geom_transformed], crop=True, all_touched=True
                    )

                if layer in [
                    "a_naip"
                ]:
                    sample["image"].append(data)
                    
                elif layer in ["prior_from_cooccurrences_101_31"]:
                    if prior_as_input:
                        sample["image"].append(data)  
                    else:
                        sample["mask"].append(data) 
                        
                elif layer in ["h_highres_labels"]:
                    # reindex the enviroatlas labels if they're not already 0-10 index
                    if layer == "h_highres_labels" and self.need_to_reindex_enviroatlas_labels: 
                        data = map_raw_enviroatlas_to_idx[data]
                        sample["mask"].append(data)
        else:
            raise IndexError(f"query: {query} spans multiple tiles which is not valid")

        sample["image"] = np.concatenate(  # type: ignore[no-untyped-call]
            sample["image"], axis=0
        )
        sample["mask"] = np.concatenate(  # type: ignore[no-untyped-call]
            sample["mask"], axis=0
        )

        sample["image"] = torch.from_numpy(  # type: ignore[attr-defined]
            sample["image"]
        )
        sample["mask"] = torch.from_numpy(sample["mask"])  # type: ignore[attr-defined]

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    
class EnviroatlasWeakInput(GeoDataset):
    """CVPR 2019 Chesapeake Land Cover dataset.

    The `CVPR 2019 Chesapeake Land Cover
    <https://lila.science/datasets/chesapeakelandcover>`_ dataset contains two layers of
    NAIP aerial imagery, Landsat 8 leaf-on and leaf-off imagery, Chesapeake Bay land
    cover labels, NLCD land cover labels, and Microsoft building footprint labels.

    This dataset was organized to accompany the 2019 CVPR paper, "Large Scale
    High-Resolution Land Cover Mapping with Multi-Resolution Data".

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/cvpr.2019.01301
    """

#     url = "https://lilablobssc.blob.core.windows.net/lcmcvpr2019/cvpr_chesapeake_landcover.zip"  # noqa: E501
#     filename = "cvpr_chesapeake_landcover.zip"
#     md5 = "0ea5e7cb861be3fb8a06fedaaaf91af9"

    crs = CRS.from_epsg(3857)
    res = 1

    valid_layers = [
        "a_naip",
        "b_nlcd",
        "c_roads",
        "d_water",
        "d1_waterways",
        "d2_waterbodies",
        "e_buildings",
        "h_highres_labels",
        "prior_from_cooccurrences_101_31_no_osm_no_buildings",
    ]
    
    states = ["pittsburgh_pa-2010_1m",
              "durham_nc-2012_1m",
              "austin_tx-2012_1m",
              "phoenix_az-2010_1m"] # dodo add states as they get ready
    
    # only pittsburch has a train and val set
    splits = (
        [f"{state}-train" for state in states[:1]]
        + [f"{state}-val" for state in states[:1]]
        + [f"{state}-test" for state in states]
        + [f"{state}-val5" for state in states] 
    )

    p_src_crs = pyproj.CRS("epsg:3857")
    p_transformers = {
        "epsg:26917": pyproj.Transformer.from_crs(
            p_src_crs, pyproj.CRS("epsg:26917"), always_xy=True
        ).transform,
        "epsg:26918": pyproj.Transformer.from_crs(
            p_src_crs, pyproj.CRS("epsg:26918"), always_xy=True
        ).transform,
        "epsg:26914": pyproj.Transformer.from_crs(
            p_src_crs, pyproj.CRS("epsg:26914"), always_xy=True
        ).transform,
        "epsg:26912": pyproj.Transformer.from_crs(
            p_src_crs, pyproj.CRS("epsg:26912"), always_xy=True
        ).transform,
    }

    def __init__(
        self,
        root: str = "data",
        splits: Sequence[str] = ["de-train"],
        layers: List[str] = ["prior_from_cooccurrences_101_31", "h_highres_labels"],
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            split: a string in the format "{state}-{train,val,test}" indicating the
                subset of data to use, for example "ny-train"
            layers: a list containing a subset of "naip-new", "naip-old", "lc", "nlcd",
                "landsat-leaf-on", "landsat-leaf-off", "buildings" indicating which
                layers to load
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            FileNotFoundError: if no files are found in ``root``
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        for split in splits:
            if not split in self.splits: print(split)
            assert split in self.splits
            
        for layer in layers:
            if not layer in self.valid_layers: print(f'{layer} not in {self.valid_layers}')

        assert all([layer in self.valid_layers for layer in layers])
        super().__init__(transforms)  # creates self.index and self.transform
        self.root = root
        self.layers = layers
        self.cache = cache
        self.checksum = checksum
        
        self.need_to_reindex_enviroatlas_labels = True

        if download:
            self._download()

        if checksum:
            if not self._check_integrity():
                raise RuntimeError(
                    "Dataset not found or corrupted. "
                    + "You can use download=True to download it"
                )

        # Add all tiles into the index in epsg:3857 based on the included geojson
        mint: float = 0
        maxt: float = sys.maxsize
        with fiona.open(os.path.join(root, "spatial_index.geojson"), "r") as f:
            for i, row in enumerate(f):
                if row["properties"]["split"] == split:
                    box = shapely.geometry.shape(row["geometry"])
                    minx, miny, maxx, maxy = box.bounds
                    coords = (minx, maxx, miny, maxy, mint, maxt)
                    self.index.insert(
                        i,
                        coords,
                        {
                            # TODO FIX for enviroatlas
                            "a_naip": row["properties"]["a_naip"],
                            "b_nlcd": row["properties"]["b_nlcd"],
                            "c_roads": row["properties"]["c_roads"],
                            "d_water": row["properties"]["d_water"],
                            "d1_waterways": row["properties"]["d1_waterways"],
                            "d2_waterbodies": row["properties"]["d2_waterbodies"],
                            "e_buildings": row["properties"]["e_buildings"],
                            "h_highres_labels": row["properties"]["h_highres_labels"],
                            "prior_from_cooccurrences_101_31_no_osm_no_buildings":row["properties"]['a_naip'].replace('a_naip',
                                                                                                        'prior_from_cooccurrences_101_31_no_osm_no_buildings')
                        },
                    )

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(query, objects=True)
        filepaths = [hit.object for hit in hits]

        sample = {
            "image": [],
            "mask": [],
            "crs": self.crs,
            "bbox": query,
        }

        if len(filepaths) == 0:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )
        elif len(filepaths) == 1:
            filenames = filepaths[0]
            query_geom_transformed = None  # is set by the first layer

            minx, maxx, miny, maxy, mint, maxt = query
            query_box = shapely.geometry.box(minx, miny, maxx, maxy)

            for layer in self.layers:
    
                fn = filenames[layer]

                with rasterio.open(os.path.join(self.root, fn)) as f:
                    dst_crs = f.crs.to_string().lower()

                    if query_geom_transformed is None:
                        query_box_transformed = shapely.ops.transform(
                            self.p_transformers[dst_crs], query_box
                        ).envelope
                        query_geom_transformed = shapely.geometry.mapping(
                            query_box_transformed
                        )

                    data, _ = rasterio.mask.mask(
                        f, [query_geom_transformed], crop=True, all_touched=True
                    )
                    

                if layer in [
                    "a_naip", 
                    "prior_from_cooccurrences_101_31_no_osm_no_buildings",
                    "e_buildings",
                    "c_roads",
                    "d1_waterways",
                    "d2_waterbodies",
                    "d_water"                    
                ]:
                    sample["image"].append(data)
                elif layer in ["h_highres_labels"]:
                    # reindex the enviroatlas labels if they're not already 0-10 index
                    if layer == "h_highres_labels" and self.need_to_reindex_enviroatlas_labels: 
                        data = map_raw_enviroatlas_to_idx[data]
                        sample["mask"].append(data)
        else:
            raise IndexError(f"query: {query} spans multiple tiles which is not valid")

        sample["image"] = np.concatenate(  # type: ignore[no-untyped-call]
            sample["image"], axis=0
        )
        sample["mask"] = np.concatenate(  # type: ignore[no-untyped-call]
            sample["mask"], axis=0
        )

        sample["image"] = torch.from_numpy(  # type: ignore[attr-defined]
            sample["image"]
        )
        sample["mask"] = torch.from_numpy(sample["mask"])  # type: ignore[attr-defined]

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
    

    
