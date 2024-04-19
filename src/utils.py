import numpy as np
import torch
import json
from pathlib import Path
import logging

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_logger():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_args(folder, args, name="config.json"):
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / name, 'w') as f:
        json.dump(args, f)
