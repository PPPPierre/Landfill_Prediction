import datetime
import yaml
import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore") 

import torch
import mlflow
import mlflow.pytorch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from src.dataset import get_dataloader_from_cfg
from src.model import get_model_from_cfg
from src.utils.logger import init_logger
from src.train import train
from src.utils.seed import setup_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for Landfill Detection")
    parser.add_argument("--config", type=str, required=True, help="Path to the yaml configuration file")
    return parser.parse_args()

def main(config: dict):
    root_path = os.path.dirname(os.path.abspath(__file__))
    time_stamp = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%SZ")
    result_dir = os.path.join(root_path, 'results', time_stamp)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    logger = init_logger("__main__", result_dir, 'train')

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {config}")
    
    # Setup random seed
    seed = config.get('seed', 0)
    setup_seed(seed)
    logger.info(f"Random seed: {seed}")

    # Loading cfgs
    train_cfg = config["train"]
    test_cfg = config["test"]

    # Loading data
    train_set, train_loader = get_dataloader_from_cfg(train_cfg["data"])
    test_set, test_loader = get_dataloader_from_cfg(test_cfg["data"])
    logger.info(f"Training data: {len(train_set)}, Testing data: {len(test_set)}")

    # Init model
    model_cfg = config["model"]
    model = get_model_from_cfg(model_cfg)
    model = model.to(device)

    train(config, time_stamp, train_loader, test_loader, model, device)

if __name__ == '__main__':
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    main(config)
