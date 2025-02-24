# Adaptation of the DCASE SELDNet23 for LAM SAMs processing

[Please visit the original SELNet23 repo for more information on the SELDnet model itself](https://github.com/sharathadavanne/seld-dcase2023)

This repository consists of multiple Python scripts forming one big architecture used to train the SELDnet.
* The `batch_feature_extraction.py` is a standalone wrapper script, that extracts the features, labels, and normalizes the training and test split features for a given dataset. Make sure you update the location of the downloaded datasets before.
* The `parameter.py` script consists of all the training, model, and feature parameters. If a user has to change some parameters, they have to create a sub-task with unique id here. Check code for examples.
* The `cls_feature_class.py` script has routines for labels creation, features extraction and normalization.
* The `cls_data_generator.py` script provides feature + label data in generator mode for training.
* The `seldnet_model.py` script implements the SELDnet architecture.
* The `SELD_evaluation_metrics.py` script implements the metrics for joint evaluation of detection and localization.
* The `train_seldnet.py` is a wrapper script that trains the SELDnet. The training stops when the SELD error (check paper) stops improving.
* The `cls_compute_seld_results.py` script computes the metrics results on your DCASE output format files. You can switch between using polar or Cartesian based scoring. Ideally both should give identical results.

Additionally, we also provide supporting scripts that help analyse the results.
 * `visualize_seldnet_output.py` script to visualize the SELDnet output


### Prerequisites

The provided codebase has been tested on python 3.8.11 and torch 1.10.0


### Training the SELDnet

In order to quickly train SELDnet follow the steps below.

* For the chosen dataset (Ambisonic or Microphone), download the respective zip file. This contains both the audio files and the respective metadata. Unzip the files under the same 'base_folder/', ie, if you are Ambisonic dataset, then the 'base_folder/' should have two folders - 'foa_dev/' and 'metadata_dev/' after unzipping.

* Now update the respective dataset name and its path in `parameter.py` script. For the above example, you will change `dataset='foa'` and `dataset_dir='base_folder/'`. Also provide a directory path `feat_label_dir` in the same `parameter.py` script where all the features and labels will be dumped.

* Extract features from the downloaded dataset by running the `batch_feature_extraction.py` script. Run the script as shown below. This will dump the normalized features and labels in the `feat_label_dir` folder. The python script allows you to compute all possible features and labels. You can control this by editing the `parameter.py` script before running the `batch_feature_extraction.py` script.

```
python3 batch_feature_extraction.py
```
* Extract features including LAM SAMs using the following command: 

```
python3 batch_feature_extraction.py 6
```

* Finally, you can now train the SELDnet using the default parameters using
```
python3 train_seldnet.py 6
```

* Additionally, you can add/change parameters by using a unique identifier \<task-id\> in if-else loop as seen in the `parameter.py` script and call them as following. Where \<job-id\> is a unique identifier which is used for output filenames (models, training plots). You can use any number or string for this.
```
python3 train_seldnet.py <task-id> <job-id>
```

This repo and its contents have the MIT License.
