# LAM HDF Dataset Generator

### Description

This script generates the dataset in HDF format to train the LAM model. 

Requirements: a dataset containing 32-channel audio recordings. We subsample 4 channels to generate the lower resolution data. By defaul we use logarithmic spacing for frequency bands ranging from 50Hz to 4500Hz, with a total of 16 bands. 

### Generate HDF dataset

Excute the following command:

```
python gen_dataset.py --dataset_name <your_dataset_name>  --data_src /path/to/<your_multichannel_dir> --save_path /path/to/<your_dest_dir>
```
