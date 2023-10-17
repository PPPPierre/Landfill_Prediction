import os
import sys
import argparse
import yaml
import datetime
import warnings
import traceback
warnings.filterwarnings("ignore") 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.train import train
from src.predict import predict
from src.utils.logger import init_logger
from src.utils.seed import setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="ML pipeline launch script for Landfill Detection")
    parser.add_argument("--config", type=str, required=True, help="Path to the yaml configuration file")
    return parser.parse_args()

def main(config: dict):

    # Set save dir
    result_dir = config.get('result_dir', None)
    if not result_dir:
        root_path = SCRIPT_DIR
        time_stamp = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%SZ")
        task_name = config.get("task_name", "task")
        version = config.get("version", "1.0.0")
        result_dir = os.path.join(root_path, 'results', f"{task_name}_{version}_{time_stamp}")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    logger = init_logger("__main__", result_dir, 'run')
    logger.info(f"{config}")

    # Setup random seed
    seed = config.get('seed', 0)
    setup_seed(seed)
    logger.info(f"Random seed: {seed}")

    # Enter pipe line
    task_type = config.get('task_type', None)
    try:
        if task_type == 'train':
            # train
            train(config=config, save_dir=result_dir)
        elif task_type == 'prediction':
            # prediction
            predict(config=config, save_dir=result_dir)
        return True
    except Exception:
        error_message = traceback.format_exc()
        logger.error(error_message)
        return False

if __name__ == '__main__':
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    main(config)
