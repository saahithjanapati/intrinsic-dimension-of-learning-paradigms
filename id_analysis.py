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
import os
import torch
import matplotlib.pyplot as plt


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



def fetch_tensors_for_detailed_ft_experiment(model_name, dataset_name, k, layer_idx, checkpoint_number, split_name, lora_r = 64):
    if lora_r != 64:
        path_to_tensors = Path(f"results/detailed-ft/activations/{model_name}-lora_r_{lora_r}_checkpoint_{checkpoint_number}/{dataset_name}/{split_name}/{k}-shot/layer-{layer_idx}")
    
    else:
        path_to_tensors = Path(f"results/detailed-ft/activations/{model_name}_checkpoint_{checkpoint_number}/{dataset_name}/{split_name}/{k}-shot/layer-{layer_idx}")
    
    print("path_to_tensors:", path_to_tensors)
    if not path_to_tensors.exists():
        return None, None
    return fetch_tensors(path_to_tensors)







def calculate_id_two_nn_with_plot(data, verbose=False):
    # Step 1: Compute all pairwise Euclidean distances
    data = data.cpu()
    dist_matrix = torch.cdist(data, data, p=2)
    if verbose:
        print("DISTANCE MATRIX")
        print(dist_matrix)
    
    num_points = data.size(0)
    
    # Step 2: Find the two nearest neighbors for each point
    sorted_distances, _ = torch.sort(dist_matrix, dim=1)
    r1 = sorted_distances[:, 1]  # First nearest neighbor distance
    r2 = sorted_distances[:, 2]  # Second nearest neighbor distance

    if verbose:
        print("First nearest neighbor distances (r1)")
        print(r1)
        print("Second nearest neighbor distances (r2)")
        print(r2)
    
    # Step 3: Calculate the ratios
    r1[r1 == 0] = 1e-10  # Avoid division by zero
    mu = r2 / r1

    if verbose:
        print("Ratios (mu)")
        print(mu)
    
    # Step 4: Compute empirical CDF
    mu_sorted, _ = torch.sort(mu)
    F_emp = torch.arange(1, num_points + 1, dtype=torch.float32) / num_points

    if verbose:
        print("Sorted ratios (mu_sorted)")
        print(mu_sorted)
        print("Empirical CDF (F_emp)")
        print(F_emp)
    
    # Step 5: Discard top 10% of the highest mu values for stability
    cutoff = int(0.9 * num_points)
    mu_sorted = mu_sorted[:cutoff]
    F_emp = F_emp[:cutoff]

    # Step 6: Logarithmic transformation
    log_mu = torch.log(mu_sorted).to(torch.float32)
    log_one_minus_F = torch.log(1 - F_emp)

    if verbose:
        print("Logarithmic transformation (log_mu)")
        print(log_mu)
        print("Logarithmic transformation (log_one_minus_F)")
        print(log_one_minus_F)
    
    # Step 7: Linear fit to determine slope (intrinsic dimension)
    A = log_mu.unsqueeze(1)  # Shape (N, 1)
    b = log_one_minus_F.unsqueeze(1)

    if verbose:
        print("Matrix A")
        print(A)
        print("Matrix b")
        print(b)
    
    # Solve the least squares problem manually
    ATA = torch.matmul(A.T, A)
    ATb = torch.matmul(A.T, b)
    
    if verbose:
        print("Matrix ATA")
        print(ATA)
        print("Matrix ATb")
        print(ATb)
    
    # Check if ATA is invertible
    try:
        slope = torch.matmul(torch.inverse(ATA), ATb)[0].item()
    except RuntimeError as e:
        if verbose:
            print("Matrix inversion error:", e)
        return float('inf'), None, None  # Return infinity to indicate failure

    id = -slope  # The slope gives us the intrinsic dimension

    # Step 8: Plot the log-log data points
    # plt.figure(figsize=(8, 6))
    # plt.scatter(log_mu, log_one_minus_F, color='blue', s=10, label="Data points")
    # plt.plot(log_mu, (slope * log_mu), color='red', label=f"Fitted Line (Slope = {id:.2f})")
    # plt.xlabel("log(μ)")
    # plt.ylabel("log(1 - F(μ))")
    # plt.title("Log-Log Plot for TWO-NN Estimation")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(log_mu, log_one_minus_F, color='blue', s=10, label="Data points")
    ax.plot(log_mu, (slope * log_mu), color='red', label=f"Fitted Line (Slope = {id:.2f})")
    ax.set_xlabel("log(μ)")
    ax.set_ylabel("log(1 - F(μ))")
    ax.set_title("Log-Log Plot for TWO-NN Estimation")
    ax.legend()
    ax.grid(True)

    return id, log_mu, log_one_minus_F, fig, ax

# Example usage (data must be provided)
# Assuming `data` is a PyTorch tensor with the input data points
# id, log_mu, log_one_minus_F = calculate_id_two_nn_with_plot(data, verbose=True)






def perform_id_analysis(data, experiment_details):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from calculate_id import calculate_lid, calculate_id_two_nn
    from tqdm import tqdm  # Import tqdm for progress bar

    # N_values = [10, 20, 50, 100, 500, 750, 1000, 2000, 3000, 4000, 5000]
    N_values = []

    results = {}

    # Add a progress bar for the outer loop
    for N in tqdm(N_values, desc="Processing N values"):
        method_1_results = []
        method_2_results = []

        num_samples = 10 if N < 5000 else 1

        for _ in range(num_samples):
            sampled_data = data[np.random.choice(data.shape[0], N, replace=False)]

            # Calculate ID using method 1
            lid_value = calculate_lid(sampled_data, k=25, random_neighbors=False, verbose=False)
            method_1_results.append(lid_value)

            # Calculate ID using method 2
            id_two_nn_value = calculate_id_two_nn(sampled_data, verbose=False)
            method_2_results.append(id_two_nn_value)

        results[N] = {
            "method_1": {
                "mean": np.mean(method_1_results),
                "std": np.std(method_1_results)
            },
            "method_2": {
                "mean": np.mean(method_2_results),
                "std": np.std(method_2_results)
            }
        }

    # Create directory for results if it doesn't exist
    if 'checkpoint_number' in experiment_details:
        results_dir = f"id_analysis_results/{experiment_details['type']}/{experiment_details['model_name']}/{experiment_details['dataset_name']}/checkpoint_{experiment_details['checkpoint_number']}/layer_{experiment_details['layer_idx']}"
    else:
        results_dir = f"id_analysis_results/{experiment_details['type']}/{experiment_details['model_name']}/{experiment_details['dataset_name']}/layer_{experiment_details['layer_idx']}"
    os.makedirs(results_dir, exist_ok=True)

    # Save results to a JSON file
    with open(f"{results_dir}/results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Plotting
    def plot_results(method, filename):
        means = [results[N][method]["mean"] for N in N_values]
        stds = [results[N][method]["std"] for N in N_values[:-1]] + [0]  # No std for N=5000

        plt.figure()
        plt.errorbar(N_values, means, yerr=stds, fmt='-o', capsize=5)
        plt.xlabel('N')
        plt.ylabel('Mean ID')
        plt.title(f'ID Analysis using {method}')
        plt.savefig(f"{results_dir}/{filename}")
        plt.close()

    plot_results("method_1", "mle.png")
    plot_results("method_2", "twonn.png")


    # make plot of twonn data by using the calculate_id_two_nn_with_plot function
    id, log_mu, log_one_minus_F, fig, ax = calculate_id_two_nn_with_plot(data, verbose=False)
    fig.savefig(f"{results_dir}/twonn_plot.png")
    plt.close(fig)






def main():



    experiments = [
    {
        "type": "icl",
        "model_name": "meta-llama/Llama-2-13b-hf",
        "dataset_name": "mmlu",
        "checkpoint_number": 620,
        "split_name": "validation",
        "layer_idx": 15,
        "k": [0]
    },
    {
        "type": "icl",
        "model_name": "meta-llama/Llama-2-13b-hf",
        "dataset_name": "mnli",
        "layer_idx": 15,
        "k": [0]
    }
    ]



    for experiment_details in experiments:

        # cases for different experiment types
        if experiment_details["type"] == "icl":
            model_name = experiment_details["model_name"]
            dataset_name = experiment_details["dataset_name"]
            k_values = experiment_details["k"]
            layer_idx = experiment_details["layer_idx"]

            for k in k_values:
                data, filenames = fetch_tensors_for_icl_experiment(model_name, dataset_name, k, layer_idx)

                if data is None and filenames is None:
                    print("No data found for the specified experiment.")
                    return

                # perform ID analysis on the fetched data
                id_analysis_results = perform_id_analysis(data, experiment_details)
            

        elif experiment_details["type"] == "detailed_ft":
            model_name = experiment_details["model_name"]
            dataset_name = experiment_details["dataset_name"]
            checkpoint_number = experiment_details["checkpoint_number"]
            split_name = experiment_details["split_name"]
            k_values = experiment_details["k"]
            layer_idx = experiment_details["layer_idx"]

            k = k_values[0]
            fetch_tensors_for_detailed_ft_experiment(model_name, dataset_name, k, layer_idx, checkpoint_number, split_name)


            data, filenames = fetch_tensors_for_detailed_ft_experiment(model_name, dataset_name, k, layer_idx, checkpoint_number, split_name)

            if data is None and filenames is None:
                print("No data found for the specified experiment.")
                return
            
            # perform ID analysis on the fetched data
            id_analysis_results = perform_id_analysis(data, experiment_details)

        
        elif experiment_details["type"] == "ft":
            pass

        
        
        elif experiment_details["type"] == "few_sample_ft":
            pass
        
        else:
            pass



if __name__ == "__main__":
    main()


# experiment details



# fetch results for specified experiment



# create id analysis results 