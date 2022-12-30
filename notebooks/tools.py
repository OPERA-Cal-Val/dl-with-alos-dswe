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
from typing import Tuple, List
from pathlib import Path, PosixPath
from itertools import product
import os
from collections import defaultdict

def retrieve_hansen_mosaic(bounds, data_product='last', download_path=Path('./')) -> List[PosixPath]:
    '''
    Given a raster bounds, retrieve the Hansen Landsat mosaic that covers this tile. If the span of the
    provided image extends beyond a single tile, multiple tiles will be downloaded 

    Arguments:
    ----------
    bounds : rasterio bounding box describing span of the raster in degrees (left, bottom, right, top)
    data_product : default 'last'. Can be one of the Hansen GFC products (treecover, gain, lossyear, datamask, first, last)

    Returns : a list of paths to the downloaded file(s)
    '''

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