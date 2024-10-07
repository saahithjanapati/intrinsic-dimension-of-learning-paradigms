import argparse
from pathlib import Path
from utils import *
from safetensors.torch import load_file
import torch
from tqdm import tqdm
from calculate_id import *
import numpy as np
import json
import sys


def fetch_tensors_for_icl_experiment(model_name, dataset_name, k, layer_idx):
    path_to_tensors = Path(f"results/icl/activations/{model_name}/{dataset_name}/{k}-shot/layer-{layer_idx}")
    if not path_to_tensors.exists():
        return None, None
    return fetch_tensors(path_to_tensors)



def fetch_tensors_for_ft_experiment(model_name, dataset_name, k, layer_idx):
    path_to_tensors = Path(f"results/ft/activations/{model_name}/{dataset_name}/{k}-shot/layer-{layer_idx}")
    if not path_to_tensors.exists():
        return None, None
    return fetch_tensors(path_to_tensors)



def fetch_tensors_for_few_sample_ft_experiment(model_name, dataset_name, k, layer_idx, num_training_samples, training_set_id):
    path_to_tensors = Path(f"results/few-sample-ft/activations/{model_name}/{dataset_name}/{k}-shot/layer-{layer_idx}")
    if not path_to_tensors.exists():
        return None, None
    return fetch_tensors(path_to_tensors)



def process_layers(models, datasets, k_values, methods, fetch_function, result_path_template, **fetch_kwargs):
    for model_name in models:
        for dataset_name in datasets:
            for k in k_values:

                methods = [tuple(method) for method in methods]
                all_results = {method: {} for method in methods}
                num_layers = get_num_layers(model_name)

                print("STARTING THE PROCESSING!!!")

                for layer_idx in tqdm(range(0, num_layers), desc='Layers', leave=False):
                    data, filenames = fetch_function(model_name, dataset_name, k, layer_idx, **fetch_kwargs)

                    if data == None and filenames == None:
                        continue

                    for method in methods:
                        if method[0] == 'mle':
                            n_neighbors = method[1]
                            ids = np.array(calculate_lid(data, n_neighbors))

                            mean_estimate = np.mean(ids)
                            std_estimate = np.std(ids)

                            all_results[method][layer_idx] = {
                                'estimate': mean_estimate,
                                'std': std_estimate
                            }

                        elif method[0] == 'twonn':
                            id_val = calculate_id_two_nn(data)
                            all_results[method][layer_idx] = id_val

                print("COMPLETED THE PROCESSING!!!")

                # Save the results for each method
                for method in methods:
                    method_name = f"{method[0]}_{method[1]}" if len(method) > 1 else method[0]

                    save_path = Path(result_path_template.format(
                        model_name=model_name,
                        dataset_name=dataset_name,
                        k=k,
                        method_name=method_name
                    ))
                    save_path.parent.mkdir(parents=True, exist_ok=True)

                    with save_path.open('w') as f:
                        json.dump(all_results[method], f, indent=4)

                    print("--------------------------------------------------")



def run_icl_experiment(config):
    process_layers(
        config['models'],
        config['datasets'],
        config['num_shots'],
        config['methods'],
        fetch_tensors_for_icl_experiment,
        "results/id/icl/{model_name}/{dataset_name}/{k}-shot/{method_name}.json"
    )



def run_ft_experiment(config):
    process_layers(
        config['models'],
        config['datasets'],
        config['num_shots'],
        config['methods'],
        fetch_tensors_for_ft_experiment,
        "results/id/ft/{model_name}/{dataset_name}/{k}-shot/{method_name}.json"
    )



def run_few_sample_ft_experiment(config):
    for training_set_id in config["training_set_ids"]:
        for num_training_samples in config["num_training_samples_list"]:
            str1 = f"/{num_training_samples}-samples/{training_set_id}/"
            process_layers(
                config['models'],
                config['datasets'],
                config['num_shots'],
                config['methods'],
                fetch_tensors_for_few_sample_ft_experiment,
                "results/id/few-sample-ft/{model_name}/{dataset_name}" + str1 + "{k}-shot/{method_name}.json",
                num_training_samples=num_training_samples,
                training_set_id=training_set_id
            )



def main():
    path_to_yaml = sys.argv[1]
    config = load_config(path_to_yaml)

    if config['experiment_type'] == 'icl':
        run_icl_experiment(config)
    elif config['experiment_type'] == 'ft':
        run_ft_experiment(config)
    elif config['experiment_type'] == 'few_ft':
        run_few_sample_ft_experiment(config)

if __name__ == "__main__":
    main()