import argparse
import json
import csv
import os

import librosa
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.utils.data import DataLoader

import trainer.kmeans as km
from util.utils import initialize_config, load_checkpoint, get_field, convert_polar_to_cartesian
from dataset.gen_dataset.gen_dataset import get_visibility_matrix

"""
Parameters
"""
parser = argparse.ArgumentParser("LAM: Latent Acoustic Map")
parser.add_argument("-C", "--config", type=str, required=True, help="Model and dataset for enhancement (*.json).")
parser.add_argument("-D", "--device", default="0", type=str, help="GPU for acoustic mapping. default: CPU")
args = parser.parse_args()

"""
Preparation
"""
#os.environ["CUDA_VISIBLE_DEVICES"] = args.device
config = json.load(open(args.config))
output_dir = config["output_dir"]
os.makedirs(output_dir, exist_ok=True) # generate output directory if doesn't exist
model_checkpoint_path = config["model_path"]
assert os.path.exists(output_dir), "Inference output directory not found."

"""
DataLoader
"""
device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
dataloader = DataLoader(dataset=initialize_config(config["dataset"]), batch_size=1, num_workers=4)

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
output_dict = {}
for audio, name in tqdm(dataloader):
    assert len(name) == 1, "Only support batch size is 1 in enhancement stage."
    name = name[0]
    padded_length = 0
    audio = audio.cpu().detach().numpy()
    audio = audio[0].T
    # compute visibility matrix from audio
    S_in,_ = get_visibility_matrix(audio, fs=config["FS"], apgd=False)
    S_in = torch.from_numpy(S_in).to(device).permute(1, 0, 2, 3)
    # perform inference
    S_out, I_pred = model(S_in)#.squeeze(0))
    I_pred = torch.prod(I_pred, dim=1, keepdim=True).squeeze(1) # product along the freq-bands and remove freq dim 
    I_pred = I_pred.cpu().detach().numpy()
    # write output to dcase format
    R = get_field()
    # loop through each 100ms audio frame
    for i in range(I_pred.shape[0]):
        output_dict[i] = [] # list of DoA outputs per frame
        lon, lat = km.get_kmeans_clusters(I_pred[i], R, N_max=config["n_max"])
        # loop through the available clusters
        for iloc in range(len(lon)): # store predicted doa labels (1 <= pred_doa <= 3)
            output_dict[i].append([lon[iloc], lat[iloc]])

    # Write to CSV in DCASE format
    with open(os.path.join(output_dir, f"{name}.csv"), mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for i in range(I_pred.shape[0]):
            for j in range(len(output_dict[i])): # iterate through number of predicted DOAs
                x, y, z = convert_polar_to_cartesian(output_dict[i][j][0], output_dict[i][j][1])
                row = [i, 0, 0, x, y, z]
                csv_writer.writerow(row)
