# Basic lib
import os
import sys
from typing import Union, Optional, List

# Lib for data collection
import geopandas as gpd
from pystac.extensions.eo import EOExtension as eo
from pystac_client import Client
import planetary_computer
import rasterio
from rasterio import windows, features, warp

# Lib for ML
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Porject lib
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.augmentation import get_transforms_from_config

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_dataloader_from_cfg(cfg: dict):

    # Load arguments
    geojson_path = cfg["geojson_path"]
    transform = get_transforms_from_config(cfg["transforms"])
    collections = cfg["collections"]
    datetime = cfg["datetime"]
    batch_size = cfg.get("batch_size", 4)
    shuffle = cfg.get("shuffle", True)
    download_dir = cfg.get("download_dir", None)

    # Loading dataset and dataloader
    dataset = SatelliteDataset(geojson_path, collections=collections, datetime=datetime, download_dir=download_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    return dataset, loader


class SatelliteDataset(Dataset):
    def __init__(self, geojson_path: str, collections: str="sentinel-2-l2a", datetime: str="2022-01-01/2022-12-30", band: str="visual", download_dir: Optional[str]=None, transform=None) -> None:
        self.data = gpd.read_file(geojson_path)
        self.collections = collections
        self.datetime = datetime
        self.band = band
        self.transform = transform
        self.catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )

        filename = f"{os.path.basename(geojson_path).replace('.geojson', '')}_{collections}_{datetime.replace('/', '_')}_{band}"
        if download_dir is None:
            self.download_dir = os.path.join(root, f"data/temp/{filename}")
        else:
            self.download_dir = download_dir
        
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        area_of_interest = self.data.iloc[idx]['geometry']
        label = self.data.iloc[idx]['label']
        data_id = self.data.iloc[idx]['id']
        data_path = os.path.join(self.download_dir, f"data_{data_id}.pt")
        
        if not os.path.exists(data_path):
            search = self.catalog.search(
                collections=[self.collections],
                intersects=area_of_interest,
                datetime=self.datetime,
                query={"eo:cloud_cover": {"lt": 10}}
            )

            items = search.item_collection()
            least_cloudy_item = min(items, key=lambda item: eo.ext(item).cloud_cover)
            asset_href = least_cloudy_item.assets[self.band].href

            with rasterio.open(asset_href) as ds:
                aoi_bounds = features.bounds(area_of_interest)
                warped_aoi_bounds = warp.transform_bounds("epsg:4326", ds.crs, *aoi_bounds)
                aoi_window = windows.from_bounds(transform=ds.transform, *warped_aoi_bounds)
                band_data = ds.read(window=aoi_window)

            img = Image.fromarray(np.transpose(band_data, axes=[1, 2, 0]))
            
            # Save to the download directory
            torch.save(img, data_path)
        else:
            # Load from the intermediate directory
            img = torch.load(data_path)
        
        sample = (img, label)
        
        if self.transform:
            sample = self.transform(sample)

        return sample

def collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return [torch.stack(images), torch.tensor(labels)]


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    transforms_list = [
        {
            "name": "to_tensor", 
            "params": {}
        },
        {
            "name": "resize", 
            "params": {"size": (224, 224)}
        },
        {
            "name": "random_flip", 
            "params": {"probability_vertical": 0.5, "probability_horizontal": 0.5}
        },
    ]

    collections = "sentinel-2-l2a"
    geojson_path = "./data/raw/test.geojson"
    datetime = "2022-01-01/2022-12-30"
    band = "visual"
    composed_transforms = get_transforms_from_config(transforms_list)
    dataset = SatelliteDataset(geojson_path, collections, datetime, band, transform=None)

    for img, label in dataset:
        print(f"Origin size: {img.size}, label: {label}")
    #     plt.imshow(img.permute(1, 2, 0))
    # plt.show()