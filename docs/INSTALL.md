# Installation steps for LAM

## Repo setup

We recommend using conda, this eases some dependencies with cuda for running all the available submodules in this repo.

```bash
conda create --name <env_name> python=3.8 -y
conda activate <env_name>
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
```

## Install requirements

```bash
pip install -e .
pip install -r requirements.txt
```
