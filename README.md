# Latent Acoustic Mapping for Direction of Arrival Estimation: A Self-Supervised Approach

<p align="center">
  <img src="docs/lam_arch.png" alt="LAM Architecture" width="700"/>
</p>

> [!WARNING]
> LAM is still undergoing active development. The repo will be finalized prior to WASPAA 2025. However, please open an issue and describe any errors you encounter. Also, make sure to pull often, as we are actively adding more features.

## Installation

See [installation instructions](docs/INSTALL.md).

## Datasets

| Dataset     | Format | Type | URL                                                                                 |
|-------------|-------------|-----------|-------------------------------------------------------------------------------------|
| EigenScape | em32 | real | [Link](https://zenodo.org/records/1012809) |
| STARSS23 | mic & em32 | real | [Link](https://zenodo.org/records/7880637)|
| LOCATA | em32 | real | [Link](https://zenodo.org/records/3630471) |
| SpatialScaper Simulated Audio | mic & em32 | synthetic | [Link](https://github.com/marl/SpatialScaper) |

## Generate dataset

See [more details on how to generate the HDF dataset](dataset/gen_dataset/README.md).

## Training

Use `train.py` to train the model. 

- `-h`, display help information
- `-C, --config`, specify the configuration file required for training
- `-R, --resume`, continue training from the checkpoint of the last saved model

Please refer to the config files [config/train/README](config/train/README.md) to understand how to setup your training config.

Example:
```
# The configuration file used to train the model is "config/train/train.json"
python train.py -C config/train/train.json

# continue training from the last saved model checkpoint
python train.py -C config/train/train.json -R
```

## Inference (K-means)

Use `infer.py` to run inference with a pre-trained model.

- `-h`, display help information
- `-D, --device`, GPU index to be use (0 for single GPU / default)
- `-C, --config`, Configuration for k-means inference (*.json).

Please refer to the config files [config/infer/README](config/infer/README.md) to understand how to setup your inference config.

```
python infer.py -C /path/to/config/inference.json -D 0
```

Example:
```
python infer.py -C config/inference/inference.json -D 0
```

## DoA Metrics from Infered K-means Output

```
python doa_metrics.py -C /path/to/config/inference.json
```

## Sound Event Localization using LAM

Use LAM's spherical acoustic maps (SAMs) as features to a SELD network (DCASE-style). Please refer to the [seld](seld) directory, where you can perform batch feature extraction of SAMS and then train a network to perform DOA on datasets like STARSS23 or LOCATA.

## Visualization
```
# Run tensorboard pointing to your directory of logs generated during training
tensorboard --logdir train

# You can use --port to specify the port of the tensorboard static server
tensorboard --logdir train --port <port> --bind_all
```

# Pre-trained Models

| Model  | Input | Checkpoint |
|-------------|-------------|-----------------|
| UpLAM | 4-channel | [UpLAM.pth](checkpoints/UpLAM.pth) |
| LAM | 32-channel | [LAM.pth](checkpoints/LAM.pth) |

## Citation

If you find our work useful, please cite our paper:

```
@article{roman2025latent,
  title={Latent Acoustic Mapping for Direction of Arrival Estimation: A Self-Supervised Approach},
  author={Roman, Adrian S, Roman, Iran R and Bello, Juan P},
  journal={IEEE Workshop on Appplications of Signal Processing to Audio and Acoustics},
  year={2025}
}
