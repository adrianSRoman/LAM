# Latent Acoustic Mapping for Direction of Arrival Estimation: A Self-Supervised Approach

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

## Visualization
```
# Run tensorboard pointing to your directory of logs generated during training
tensorboard --logdir train

# You can use --port to specify the port of the tensorboard static server
tensorboard --logdir train --port <port> --bind_all
```
