# LAM and UpLAM K-Means Training Config Files


Define a configuration file to run training with LAM or UpLAM on a dataset. Below are the fields you need to modify from the provided template for your own use.

- `root_dir` defines the path to your cloned LAM directory

- `trainer` -> `upsample` defines whether you need to do covariance matrix upsampling. For `LAM` is `false`, for `UpLAM` is `true`. 

- `model` defines the LAM variant you need to use: `LAM` or `UpLAM`.

- `train_dataset` -> `path_to_datasets` defines the path to the training dataset. This is a directory containing HDF files for each dataset processed. The `leave_out` list is used in case you desire not to use one of the datasets for training.

- `validation_dataset` -> `path_to_datasets` defines the path to the validation dataset. This is a directory containing HDF files for each dataset processed. The `leave_out` list is used in case you desire not to use one of the datasets for validation.

- `train_dataloader` defines the batch size and number of workers. We recommend a batch size of 8 up to 32 for either model. 
