# GIS imports
from pyproj import Transformer
import rasterio
from rasterio.warp import Resampling, reproject
from rasterio.transform import xy
from rasterio.features import shapes
from rasterio.merge import merge
from affine import Affine
import fiona
from rasterio.warp import transform_bounds

# dataframe tools
import pandas as pd

# math tools
from skimage.restoration import denoise_tv_bregman
import numpy as np

# misc imports
from typing import List
from pathlib import Path, PosixPath
from itertools import product
import os

def retrieve_hansen_mosaic(bounds:list, data_product:str='last', download_path:PosixPath=Path('./')) -> List[PosixPath]:
    """Given a raster bounds, retrieve the Hansen Landsat mosaic that covers this tile. If the span of the
    provided image extends beyond a single tile, multiple tiles will be downloaded.

    Args:
        bounds (list): rasterio bounding box describing span of the raster in degrees (left, bottom, right, top)
        data_product (str, optional): Data product to download. Defaults to 'last'. Can be one of the Hansen GFC products (treecover, gain, lossyear, datamask, first, last)
        download_path (PosixPath, optional): Download location for data products. Defaults to Path('./').

    Returns:
        List[PosixPath]: a list of paths to the downloaded file(s)
    """
    assert min(bounds) >=-180, "Bounds must be specified in EPSG:4236"
    assert min(bounds) <= 360, "Bounds must be specified in EPSG:4236"

    if not isinstance(download_path, PosixPath):
        download_path = Path(download_path)

    def lon_edge(x):
        if (x <= 0):
            return 10*(int(x/10) - 1)
        else:
            return 10*(int(x/10))

    def lat_edge(x):
        if(x >= 0):
            return 10*(int(x/10) + 1)
        else:
            return 10*(int(x / 10))

    left_edge, right_edge = list(map(lon_edge, [bounds[0], bounds[2]]))
    bottom_edge, top_edge = list(map(lat_edge, [bounds[1], bounds[3]]))

    # number of tiles to download in the x and y directions
    x_tiles = list(np.arange(left_edge, right_edge+10, 10))
    y_tiles = list(np.arange(bottom_edge, top_edge+10, 10))
    nx_ny = list(product(x_tiles, y_tiles))

    def download_one_tile(edges, data_product, download_path=Path('./')):
        left_edge, top_edge = edges
        wget_command = 'https://storage.googleapis.com/earthenginepartners-hansen/GFC-2020-v1.8/Hansen_GFC-2020-v1.8_{}_{}_{}.tif'

        left_edge = (str(left_edge) + 'E' if left_edge >= 0 else str(abs(left_edge)) + 'W').zfill(4) 
        top_edge = (str(top_edge) + 'N' if top_edge >= 0 else str(abs(top_edge)) + 'S').zfill(3)
        wget_command = wget_command.format(data_product, top_edge, left_edge)

        # don't clobber pre-existing files, download quietly, and to the path provided
        wget_command_prefixes = '-nc -q -P {}'.format(download_path)

        wget_command = 'wget {} {}'.format(wget_command_prefixes, wget_command)
        os.system(wget_command)
        return download_path/wget_command.split()[-1].split('/')[-1]
        
    print(f"Downloading {len(nx_ny)} Hansen {data_product} files")
    outputs = list(map(download_one_tile, nx_ny, [data_product]*len(nx_ny), [download_path]*len(nx_ny)))
    return outputs

def denoise(img:np.ndarray, weight=0.2, input_db=False, return_db=False) -> np.ndarray :
    """Apply TV denoising to input SAR image

    Args:
        img (np.ndarray): image to be denoised
        weight (float, optional): Weight parameter for denoising. Lower values result in greater denoising. Defaults to 0.2.
        input_db (bool, optional): Set to True if input image is in logarithmic units. Defaults to False.
        return_db (bool, optional): Set to True if returned image should also be in logarithmic units. Defaults to False.

    Returns:
        np.ndarray: Denoised image
    """

    idx = np.where((np.isnan(img)) | (np.isinf(img)) | (img == 0))
    
    # prevent nan issues
    img[idx] = 1e-5
    
    # Convert to db and make noise additive
    if not input_db:
        img = 10 * np.log10(img)

    # Use TV denoising. The weight parameter lambda = .2 worked well for SAR image
    # Higher values mean less denoising and lower mean image will appear smoother.
    img_tv = denoise_tv_bregman(img, weight)

    if return_db:
        return img_tv
    else:
        return 10**(img_tv / 10.)