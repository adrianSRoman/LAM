import argparse
import json
import os

import librosa
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from util.utils import initialize_config, load_checkpoint

"""
Parameters
"""
parser = argparse.ArgumentParser("LAM: Latent Acoustic Map")
parser.add_argument("-C", "--config", type=str, required=True, help="Model and dataset for enhancement (*.json).")
parser.add_argument("-D", "--device", default="-1", type=str, help="GPU for acoustic mapping. default: CPU")
parser.add_argument("-O", "--output_dir", type=str, required=True, help="Where to save DCASE format output csv.")
parser.add_argument("-M", "--model_checkpoint_path", type=str, required=True, help="Checkpoint.")
args = parser.parse_args()

"""
Preparation
"""
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
config = json.load(open(args.config))
model_checkpoint_path = args.model_checkpoint_path
output_dir = args.output_dir
assert os.path.exists(output_dir), "Inference outpud directory should exist."

"""
DataLoader
"""
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
dataloader = DataLoader(dataset=initialize_config(config["dataset"]), batch_size=1, num_workers=0)

"""
Model
"""
model = initialize_config(config["model"])
model.load_state_dict(load_checkpoint(model_checkpoint_path, device))
model.to(device)
model.eval()

"""
Inference loop
"""
for audio, name in tqdm(dataloader):
    assert len(name) == 1, "Only support batch size is 1 in enhancement stage."
    name = name[0]
    padded_length = 0

    audio = audio.to(device)  # [1, 1, T]

    # convert audio to visibility matrix
    
    # perform inference

    # write output to dcase format