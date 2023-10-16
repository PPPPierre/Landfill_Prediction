# Basic lib
import os
import sys
import logging
from typing import Optional, Tuple, List

import geopandas as gpd
from pystac_client import Client
import planetary_computer

import torch
import torchvision.transforms as transforms
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.dataset import download_aoi_data, get_transforms_from_config
from src.model import get_model_from_cfg

def crop_image_with_overlap(
        image: np.ndarray, 
        patch_size: Tuple[int, int], 
        overlap: Tuple[float, float]
        ):
    overlap_h, overlap_w = overlap
    assert 0 <= overlap_h < 1 and 0 <= overlap_w < 1

    slices = []
    h, w, _ = image.shape
    patch_h, patch_w = patch_size
    if h < (1.5 * patch_h):
        patch_h = h
        overlap_h = 0
    if w < (1.5 * patch_w):
        patch_w = h
        overlap_w = 0

    patch_num_h,  patch_num_w = int(h // patch_h / (1 - overlap[0])), int(w // patch_w / (1 - overlap[1]))

    if patch_num_h == 1:
        interval_h = 0
    else:
        interval_h = (h - patch_h) // (patch_num_h - 1)
    
    if patch_num_w == 1:
        interval_w = 0
    else:
        interval_w = (w - patch_w) // (patch_num_w - 1)

    for i in range(0, patch_num_h):
        for j in range(0, patch_num_w):
            start_h = i * interval_h
            start_w = j * interval_w
            slices.append(image[start_h: start_h + patch_h, start_w: start_w + patch_w])
    
    return slices

def predict_slices(
        img_patches: List[np.ndarray], 
        model: torch.nn.Module, 
        device: torch.device,
        transform: Optional[transforms.Compose]=None, 
        ):
    results = []
    if not transform:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    with torch.no_grad():
        for s in img_patches:
            input_tensor = transform(s).unsqueeze(0).to(device)
            output = model(input_tensor)
            probs = torch.sigmoid(output)
            results.append(probs[0].item())
    
    return results

def post_processing(results):
    return np.max(results)


def predict(cfg: dict, save_dir: str):
    # Logger
    logger = logging.getLogger("__main__")

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"device: {device}")

    # Load model
    model_cfg = cfg["model"]
    model = get_model_from_cfg(model_cfg)
    model = model.to(device)
    threshold = model_cfg["threshold"]

    # Load model
    if 'weight' in model_cfg:
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weight_path = os.path.join(root_path, model_cfg['weight'])
        checkpoint = torch.load(weight_path)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    logger.info(f"Model successfully loaded")
    logger.info(f"{model}")

    # Load data
    data_cfg = cfg["data"]
    geojson_path = data_cfg["geojson_path"]
    transform = get_transforms_from_config(data_cfg["transform"])
    collections = data_cfg["collections"]
    datetime = data_cfg["datetime"]
    band = data_cfg["band"]
    patch_size = data_cfg["patch_size"]
    overlap = data_cfg["overlap"]
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    data = gpd.read_file(geojson_path)

    logger.info(f"Total amount of data to be predicted: {len(data)}")

    model.eval()
    # Start prediction
    for i in range(len(data)):
        area_of_interest = data.iloc[i]['geometry']
        band_data = download_aoi_data(area_of_interest, catalog, collections, datetime, band)
        img = np.transpose(band_data, axes=[1, 2, 0])
        img_patches = crop_image_with_overlap(img, patch_size, overlap)
        results = predict_slices(img_patches, model, device, transform)
        prob = post_processing(results)
        label = 1 if prob >= threshold else 0
        data.at[i, 'label'] = label
        logger.info(f"Image: {i}, img shape: {img.shape}, prob: {prob}, label: {label}")

    # Save result
    filename = f"{os.path.basename(geojson_path).replace('.geojson', '')}_result.geojson"
    file_path = os.path.join(save_dir, filename)
    data.to_file(file_path, driver='GeoJSON')
    return file_path


if __name__ == '__main__':
    image = np.random.randint(0, 255, (512, 512, 3)).astype(np.uint8)
    
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = torch.nn.Linear(3 * 64 * 64, 1)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    model = SimpleModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    slices = crop_image_with_overlap(image, (64, 64), (0.25, 0.25))
    print(f"Number of slices: {len(slices)}")

    results = predict_slices(slices, model, device)
    print(f"Results: {results}")

    post_processed_result = post_processing(results)
    print(f"Post Processed Result: {post_processed_result}")




