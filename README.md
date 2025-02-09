# LAM
Work in progress: Latent Acoustic Map

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

## Inference

Use `infer.py` to run inference with a pre-trained model.

```
python infer.py -C /path/to/config/inference.json -D 0 -O /path/to/destination_dir -M /path/to/train/checkpoints/checkpoint.pth
```

Example:
```
python infer.py -C config/inference/inference.json -D 0 -O inference_out_dir -M train/checkpoints/model_0100.pth
```

## Visualization
```
# Run tensorboard pointing to your directory of logs generated during training
tensorboard --logdir train

# You can use --port to specify the port of the tensorboard static server
tensorboard --logdir train --port <port> --bind_all
```
