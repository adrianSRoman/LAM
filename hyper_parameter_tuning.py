import json
import itertools
from train import main

if __name__ == '__main__':
    learning_rates = [1e-5, 1e-4, 1e-6]
    loss_functions = ["MSEL1Loss"]
    batch_sizes = [8, 16, 32]
    l1_weights = [1e-3, 1e-4, 1e-5]

    experiments_path = "config/finetuning"
    template_config_json = f"{experiments_path}/template_config.json"
    train_config = json.load(open(template_config_json))

    experiment_variants = list(itertools.product(learning_rates, loss_functions, l1_weights, batch_sizes))
    print(f"Total number of experiments", len(experiment_variants))
    experiment_count = 0
    for lr, loss_func, l1_weight, batch_size in experiment_variants:
        print(f"Starting experiment {experiment_count} / {len(experiment_variants)} ...")
        # new config JSON file
        json_filename = f"finetuning_{lr:.0e}_{loss_func}_{l1_weight}_{batch_size}"
        json_filepath = f"{experiments_path}/{json_filename}.json"

        train_config["optimizer"]["lr"] = lr
        train_config["loss_function"]["main"] = loss_func
        train_config["loss_function"]["args"]["l1_weight"] = l1_weight
        train_config["train_dataloader"]["batch_size"] = batch_size

        train_config["trainer"]["epochs"] = 100

        json.dump(train_config, open(json_filepath, 'w'), indent=4)

        config = json.load(open(json_filepath))
        config["experiment_name"] = json_filename
        config["config_path"] = json_filepath

        main(config, resume=False)

        experiment_count += 1
