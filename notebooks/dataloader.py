# ML imports
import torch

# gis imports
import rasterio

# math/pandas/imtools imports
import numpy as np

# misc imports
from tools import denoise
import random

class sarDataLoader(torch.utils.data.Dataset):
    def __init__(self, x_paths, y_paths, transforms=None, return_sar=True, return_optical=True, return_dem=True, channel_drop=False):
        super().__init__()
        self.data = x_paths
        self.label = y_paths
        self.transforms = transforms
        self.return_sar = return_sar
        self.return_optical = return_optical
        self.return_dem = return_dem
        self.channel_drop = channel_drop
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, inference=False):
        img = self.data.iloc[idx]
        
        with rasterio.open(img.hh) as ds:
            sar_profile = ds.profile
        
        data_chips = []
        if self.return_sar:
            with rasterio.open(img.hh) as ds:
                data_chips.append(denoise(np.clip(ds.read(1), 0, 1), weight=0.2, return_db=True) - self.hh_mean)
            with rasterio.open(img.hv) as ds:
                data_chips.append(denoise(np.clip(ds.read(1), 0, 1), weight=0.2, return_db=True) - self.hv_mean)

        if self.return_optical:
            with rasterio.open(img.red) as ds:
                data_chips.append(ds.read(1) - self.red_mean)

            with rasterio.open(img.nir) as ds:
                data_chips.append(ds.read(1) - self.nir_mean)

            with rasterio.open(img.swir1) as ds:
                data_chips.append(ds.read(1) - self.swir1_mean)

            with rasterio.open(img.swir2) as ds:
                data_chips.append(ds.read(1) - self.swir2_mean)

        if self.return_dem:
            with rasterio.open(img.dem) as ds:
                data_chips.append(ds.read(1))

        x_arr = np.stack(data_chips, axis=-1).astype(np.float32)
        
        if (self.return_optical is True) and (self.channel_drop is True) and (random.random() > 0.2):
            for c in random.choices([2, 3, 4, 5], k=3):
                x_arr[:, :, c] = 0.

        # Check data for NaN/Inf values
        try:
            assert not (np.any(np.isnan(x_arr)) or np.any((np.isinf(x_arr))))
        except AssertionError:
            print(f"NaN values found. IDX : {idx}, {img.hh}")
            # nearest neighbors in denoise() didn't solve nan/inf issues. fill array with zeros
            for i in range(x_arr.shape[-1]):
                if np.isnan(np.nanmax(x_arr[:, :, i])):
                    x_arr[:, :, i] = 0.

        if type(self.label) != type(None):
            with rasterio.open(self.label.iloc[idx].labels) as ds:
                y_arr = ds.read(1)
        else:
            y_arr = np.zeros(x_arr.shape[:2])

        if self.transforms:
            transformed = self.transforms(image=x_arr, mask=y_arr)
            x_arr = transformed['image']
            y_arr = transformed['mask']
        
        if inference:
            return x_arr.transpose([2, 0, 1]), y_arr, sar_profile
        else:
            return x_arr.transpose([2, 0, 1]), y_arr