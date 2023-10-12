import os
from typing import Union, Optional

import geopandas as gpd
from pystac.extensions.eo import EOExtension as eo
from pystac_client import Client
import planetary_computer
import rasterio
from rasterio import windows, features, warp

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class SatelliteDataset(Dataset):
    def __init__(self, geojson_path: str, datetime: str, transform=None, download_dir: Optional[str]=None) -> None:
        self.data = gpd.read_file(geojson_path)
        self.datetime = datetime
        self.transform = transform
        self.catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )

        if download_dir is None:
            filename = os.path.basename(geojson_path).replace(".geojson", "")
            self.download_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"data/temp/{filename}")
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
                collections=["sentinel-2-l2a"],
                intersects=area_of_interest,
                datetime=self.datetime,
                query={"eo:cloud_cover": {"lt": 10}}
            )

            items = search.item_collection()
            least_cloudy_item = min(items, key=lambda item: eo.ext(item).cloud_cover)
            asset_href = least_cloudy_item.assets["visual"].href

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
    images = [item['image'] for item in batch]
    labels = [item['label'] for item in batch]
    return {'images': torch.stack(images), 'labels': torch.tensor(labels)}

if __name__ == '__main__':

    from augmentation import get_transforms_from_config
    import matplotlib.pyplot as plt

    transforms_list = [
        {
            "name": "to_tensor", 
            "params": {}
        },
        {
            "name": "resize", 
            "params": {"size": (256, 256)}
        },
        {
            "name": "random_flip", 
            "params": {"probability_vertical": 0.5, "probability_horizontal": 0.5}
        },
    ]

    composed_transforms = get_transforms_from_config(transforms_list)
    geojson_path = "./data/raw/train.geojson"
    datetime = "2022-01-01/2022-12-30"
    dataset = SatelliteDataset(geojson_path, datetime, transform=composed_transforms)

    for img, label in dataset:
        print(f"Origin size: {img.size()}, label: {label}")
        plt.imshow(img.permute(1, 2, 0))
    plt.show()