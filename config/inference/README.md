# LAM and UpLAM K-Means Inference Config Files


Define a configuration file to run inference with LAM or UpLAM on a dataset.

- `model` defines the LAM variant you need to use.

- `dataset` defines the path to the wavefiles assets you desire to use.

- `model_path` defines the path to the LAM checkpoint you desire to use.

- `output_dir` defines the output path where .csv files with DoAE inference from K-means will be located.

- `n_max` defines the number of maximum points to be considered for K-means clustering.
