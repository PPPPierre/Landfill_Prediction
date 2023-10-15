import os
import sys
import argparse
import yaml
import datetime
import warnings
warnings.filterwarnings("ignore") 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from src.utils.logger import init_logger
from src.utils.seed import setup_seed
from src.predict import predict

def parse_args():
    parser = argparse.ArgumentParser(description="Script for prediction of Landfill")
    parser.add_argument("--config", type=str, required=True, help="Path to the yaml configuration file")
    return parser.parse_args()

def main(config: dict):
    # Set save dir
    result_dir = config.get("result_dir", None)
    if result_dir is None:
        root_path = os.path.dirname(os.path.abspath(__file__))
        time_stamp = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%SZ")
        job_name = config.get("job_name", "pred")
        version = config.get("version", "1.0.0")
        result_dir = os.path.join(root_path, 'results', f"{job_name}_{version}_{time_stamp}")
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

    # Initialize log
    logger = init_logger("__main__", result_dir, 'train')
    
    # Setup random seed
    seed = config.get('seed', 0)
    setup_seed(seed)
    logger.info(f"Random seed: {seed}")

    # predict
    predict(cfg=config, save_dir=result_dir)

if __name__ == '__main__':
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    main(config)