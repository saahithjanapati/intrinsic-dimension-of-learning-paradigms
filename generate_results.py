import argparse
import json
import os
import numpy as np
from pathlib import Path
from sklearn.metrics import auc
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from matplotlib.backends.backend_pdf import PdfPages



PATH_TO_RESULTS_DIR = Path("concise_results/")  # set this to the results directory

# Global function to set the theme and color palette
def set_plot_theme():
    sns.set_theme(style="darkgrid")
    sns.set_palette("deep")


def get_average_accuracy(model, dataset, experiment_type, accuracy_type, checkpoint_number=None, split=None):
    def load_accuracy(path):
        try:
            with path.open("r") as f:
                data = json.load(f)
                return data.get("accuracy", None)
        except FileNotFoundError:
            print(f"File not found: {path}")
            return None
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {path}")
            return None

    if "icl" in experiment_type:
        k_shot = int(experiment_type.split("-")[1])
        path_to_accuracy = PATH_TO_RESULTS_DIR / f"icl/accuracy_results/{model}/{dataset}/{k_shot}-shot/"

    elif experiment_type == "finetune 1k":
        path_to_accuracy = PATH_TO_RESULTS_DIR / f"ft/accuracy_results/{model}_final/{dataset}/0-shot/"

    elif experiment_type == "finetune 10":
        path_to_accuracy = PATH_TO_RESULTS_DIR / f"few-sample-ft/accuracy_results/{model}_final/{dataset}/0-shot/"

    elif experiment_type == "detailed-ft":
        path_to_accuracy = PATH_TO_RESULTS_DIR / f"detailed-ft/accuracy_results/{model}_checkpoint_{checkpoint_number}/{dataset}/{split}/0-shot/"



    else:
        print(f"Unknown experiment type: {experiment_type}")
        return None

    if accuracy_type == "logit-accuracy":
        path_to_accuracy /= "accuracy.json"
    elif accuracy_type == "generation accuracy":
        path_to_accuracy /= "acc-results.json"
    else:
        print(f"Unknown accuracy type: {accuracy_type}")
        return None

    return load_accuracy(path_to_accuracy)



def get_intrinsic_dimensions(model, dataset, experiment_type, estimator, checkpoint_number=None, split=None):
    path_to_id = PATH_TO_RESULTS_DIR / "id/"

    if "icl" in experiment_type:
        k_shot = int(experiment_type.split("-")[1])
        path_to_id /= f"icl/{model}/{dataset}/{k_shot}-shot/{estimator}.json"

    elif experiment_type == "finetune 1k":
        path_to_id /= f"ft/{model}_final/{dataset}/0-shot/"
        path_to_id /= f"{estimator}.json"

    elif experiment_type == "finetune 10":
        path_to_id /= f"few-sample-ft/{model}_final/{dataset}/10-samples/0/0-shot/"
        path_to_id /= f"{estimator}.json"

    elif experiment_type == "detailed-ft":
        path_to_id /= f"detailed-ft/{model}_checkpoint_{checkpoint_number}/{dataset}/{split}/0-shot/"
        path_to_id /= f"{estimator}.json"


    try:
        with path_to_id.open("r") as f:
            data = json.load(f)

            if estimator == "twonn":
                id_vals = [data[str(i)] for i in range(len(data))]
                return id_vals[1:]
            
            else:
                id_vals = [data[str(i)]["estimate"] for i in range(len(data))]
                return id_vals[1:]
            
    except FileNotFoundError:
        print(f"File not found: {path_to_id}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {path_to_id}")
        return None



def get_num_layers(model_name):
    """Returns the number of layers for a given model name."""
    model_layers = {
        "meta-llama/Llama-2-70b-hf": 81,
        "meta-llama/Llama-2-13b-hf": 41,
        "meta-llama/Llama-2-7b-hf": 33,
        "meta-llama/Meta-Llama-3-8B": 33,
        "meta-llama/Meta-Llama-3-70B": 81,
        "EleutherAI/pythia-1b": 17,
        "EleutherAI/pythia-1.4b": 25,
        "EleutherAI/pythia-2.8b": 33,
        "EleutherAI/pythia-6.9b": 33,
        "EleutherAI/pythia-12b": 37,
        "mistralai/Mistral-7B-v0.3": 33,
        "google/gemma-2-27b": 47,
        "google/gemma-2-9b": 43,
    }

    layers = model_layers.get(model_name)
    if layers is None:
        for key in model_layers:
            if key in model_name:
                layers = model_layers[key]
                break
    
    if layers is None:
        print(f"Model name {model_name} not found in the layers dictionary")
    else:
        print(f"Model {model_name} has {layers} layers")
    
    return layers


################################################################################

models = [
        "meta-llama/Meta-Llama-3-8B", 
        "mistralai/Mistral-7B-v0.3", 
        "meta-llama/Llama-2-7b-hf", 
        "meta-llama/Llama-2-13b-hf"]

datasets = ["sst2", "cola", "qnli", "qqp", "mnli", "ag_news", "commonsense_qa", "mmlu"]
mle_estimator = "mle_50"
experiment_types = ["icl-0", "icl-1", "icl-2", "icl-5", "icl-10", "finetune 1k", "finetune 10"]
accuracy_method = "logit-accuracy"



def generate_auc_data():
    """
    This method computes AUC data for all the experiments and stores them in a single file.
    """

    def compute_normalized_auc(x_values, y_values):
        # Normalize x_values to [0, 1] range
        x_normalized = (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values))
        # Compute AUC
        return auc(x_normalized, y_values)

    results_dir = "results_and_figures"
    os.makedirs(results_dir, exist_ok=True)

    auc_results = {}

    for model in models:
        layers = get_num_layers(model)
        if layers is None:
            continue

        layers -= 1
        
        x_values = np.arange(1, layers+1)  # Exclude the first layer (index 0)

        for dataset in datasets:
            for experiment_type in experiment_types:
                id_values = get_intrinsic_dimensions(model, dataset, experiment_type, mle_estimator)

                if id_values is None:
                    print(f"Skipping AUC computation; no ID data found for {model}, {dataset}, {experiment_type}")
                    continue

                if len(id_values) != layers:
                    print(f"Warning: ID values length {len(id_values)} does not match layers {layers} for {model}, {dataset}, {experiment_type}")
                    continue

                normalized_auc = compute_normalized_auc(x_values, id_values)

                # Store the result in the dictionary
                auc_results[f"{model}_{dataset}_{experiment_type}"] = {'normalized_auc': normalized_auc}

    # Save all results in a single file
    result_file = Path(results_dir) / f"all_auc_results_{mle_estimator}.json"
    with result_file.open('w') as f:
        json.dump(auc_results, f, indent=4)
    
    print("AUC Data computation complete and stored in a single file.")

    # Add code to generate Figure 1



def generate_auc_boxplot():
    import seaborn as sns
    import matplotlib.pyplot as plt
    import json
    from pathlib import Path

    # Set the theme and color palette
    set_plot_theme()

    # Load the AUC data from the saved JSON file
    results_dir = "results_and_figures"
    result_file = Path(results_dir) / f"all_auc_results_{mle_estimator}.json"
    
    try:
        with result_file.open('r') as f:
            auc_results = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {result_file}")
        return
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {result_file}")
        return

    # Organize AUC data by experiment type
    auc_data_by_experiment = {exp_type: [] for exp_type in experiment_types}
    
    for key, value in auc_results.items():
        # Split the key by underscores and check the last part
        key_parts = key.split('_')
        experiment_type = key_parts[-1]
        if experiment_type in experiment_types:
            auc_data_by_experiment[experiment_type].append(value['normalized_auc'])

    # Prepare data for seaborn
    data = []
    for exp_type, auc_values in auc_data_by_experiment.items():
        for auc in auc_values:
            data.append({'Experiment Type': exp_type, 'Normalized AUC': auc})
    data = pd.DataFrame(data)

    # Create a boxplot
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Experiment Type', y='Normalized AUC', data=data)
    plt.title('AUC Boxplot by Experiment Type', fontsize=16)
    plt.xlabel('Experiment Type', fontsize=14)
    plt.ylabel('AUC Normalized by Number of Layers', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    plot_file = Path(results_dir) / "auc_boxplot.pdf"
    plt.savefig(plot_file)
    plt.close()

    print(f"Boxplot saved to {plot_file}")



# def generate_inter_method_auc_heatmap():
#     """
#     loop over all the datasets
#     make sure intrinsic dimension data for all experiment_types we have. if not, skip it.

#     each dataset will have some models for which we ran the experiment on that dataset.

#     for each model, we will have run several experiment_types on that model

#     Load up the experiment types. For each ordered pair (experiment_type_1, experiment_type_2) for that model, dataset combo, do the following:

#         - compute the signed area between the curves, (curve_1 - curve_2, where curve_1 corresponds to experiment_1 and curve_2 corresponds to experiment_2) and add this to a running sum for the signed area between
#         curve_1 and curve_2

#         - store the signed area in a table and also keep count that we counted this pair of experiments for the ordered pair (curve_1, curve_2)
    
#     once we've processed all the experiments, normalize the signed areas for each ordered pair by the number of pairs that were counted for this sum
    
#     Then generate a heatmap, where each row corresponds to a choice of cruve_1 and each column corresponds to a choice of cruve_2
#     color the cells of the heatmap according the normalized areas
#     """

def generate_inter_method_auc_heatmap():
    # Initialize a dictionary to store signed areas
    signed_areas = {exp1: {exp2: 0 for exp2 in experiment_types} for exp1 in experiment_types}
    counts = {exp1: {exp2: 0 for exp2 in experiment_types} for exp1 in experiment_types}

    for dataset in datasets:
        for model in models:
            # Collect intrinsic dimension data for all experiment types
            id_data = {}
            for experiment_type in experiment_types:
                id_values = get_intrinsic_dimensions(model, dataset, experiment_type, mle_estimator)
                if id_values != None:
                    id_data[experiment_type] = id_values[1:] # cut out layer 0

            # Ensure we have data for all experiment types
            if len(id_data) != len(experiment_types):
                continue

            # Get the number of layers for the model
            layers = get_num_layers(model)
            if layers is None:
                continue

            layers -= 1 # cut out layer 0

            # Compute signed areas for each ordered pair of experiment types
            for exp1 in experiment_types:
                for exp2 in experiment_types:
                    if exp1 != exp2:

                        curve1 = np.array(id_data[exp1])
                        curve2 = np.array(id_data[exp2])

                        if len(curve1) != len(curve2):
                            print("INVALID PAIR", model, dataset, exp1, exp2)
                            continue


                        normalized_signed_area = np.trapz(curve1 - curve2) / layers
                        signed_areas[exp1][exp2] += normalized_signed_area
                        counts[exp1][exp2] += 1

    # Normalize signed areas by the number of pairs counted
    for exp1 in experiment_types:
        for exp2 in experiment_types:
            if counts[exp1][exp2] > 0:
                signed_areas[exp1][exp2] /= counts[exp1][exp2]

    # Convert signed areas to a DataFrame for seaborn
    signed_areas_df = pd.DataFrame(signed_areas)

    # Generate the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(signed_areas_df, annot=True, cmap="coolwarm", center=0)
    plt.title('Normalized Signed Difference of Area Between ID With Different Learning Types', fontsize=16)
    plt.xlabel('Experiment Type 1', fontsize=14)
    plt.ylabel('Experiment Type 2', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()

    # Save the heatmap
    results_dir = "results_and_figures"
    plot_file = Path(results_dir) / "inter_method_auc_heatmap.pdf"
    plt.savefig(plot_file)
    plt.close()

    print(f"Heatmap saved to {plot_file}")


# def generate_two_column_summary():
#     MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
#     dataset = "ag_news"

#     """
#     make a graphic with two graphs organized side by side. 
    
#     The first grpah should be a bar-plot. 
#     the x-axis should correspond to experiment-type and the y-axis should correspond to accuracy.

#     The second graph should be a line graph. The x-axis should be the layer_idx and the y-axis should be intrinsic dimension
#     Each line on this graph should correspond to an experiment_type

#     Use the MODEL_NAME and dataset provided
#     """
#     pass



def generate_expanded_icl_appendix():
    """
    I want to generate 
    """
    
    pass



def generate_two_column_summary():
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
    dataset = "ag_news"

    set_plot_theme()
    # Prepare data for the bar plot (accuracy)
    accuracy_data = []
    for experiment_type in experiment_types:
        accuracy = get_average_accuracy(MODEL_NAME, dataset, experiment_type, accuracy_method)
        if accuracy is not None:
            accuracy_data.append({'Experiment Type': experiment_type, 'Accuracy': accuracy})

    accuracy_df = pd.DataFrame(accuracy_data)

    # Prepare data for the line plot (intrinsic dimension)
    id_data = {}
    for experiment_type in experiment_types:
        id_values = get_intrinsic_dimensions(MODEL_NAME, dataset, experiment_type, mle_estimator)
        if id_values is not None:
            id_data[experiment_type] = id_values

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar plot for accuracy
    sns.barplot(x='Experiment Type', y='Accuracy', data=accuracy_df, ax=axes[0])
    axes[0].set_title(f'Accuracy by Experiment Type\nModel: {MODEL_NAME}, Dataset: {dataset}')
    axes[0].set_xlabel('Experiment Type')
    axes[0].set_ylabel('Accuracy')
    axes[0].tick_params(axis='x', rotation=45)

    # Line plot for intrinsic dimension
    for experiment_type, id_values in id_data.items():
        axes[1].plot(range(1, len(id_values) + 1), id_values, label=experiment_type)

    axes[1].set_title(f'Intrinsic Dimension by Layer\nModel: {MODEL_NAME}, Dataset: {dataset}')
    axes[1].set_xlabel('Layer Index')
    axes[1].set_ylabel('Intrinsic Dimension')
    axes[1].legend(title='Experiment Type')

    plt.tight_layout()

    # Save the plot to a PDF file
    results_dir = "results_and_figures"
    os.makedirs(results_dir, exist_ok=True)
    plot_file = Path(results_dir) / "two_column_summary.pdf"
    plt.savefig(plot_file)
    plt.close()

    print(f"Two-column summary saved to {plot_file}")



# def generate_accuracy_ranking_table():
#     """
#     I want to generate a table containing information regarding the accuracies of various models on the datasets tested

#     It should look like this:


#                 | exp_type_1  | exp_type_2  | exp_type_3
#                 ----------------------------------------
#                 | avg_rank_1  |  avg_rank_2  | avg_rank_3 
#                 ----------------------------------------
#     dataset_1   |avg_acc_1_d1 | avg_acc_2_d1 | avg_acc_3_d1

#     dataset_2   |avg_acc_2_d2 | avg_acc_2_d2 | avg_acc_3_d2


#     and so on for all the combinations of model and dataset.

#     avg_rank_i should be the average rank when comparing accuracy of exp_type_i with all the other experiment types


#     avg_acc_i_dj should correspond to the average accuracy of exp_type_i on dataset_j. It is average because you should average over all the models we have.

#     Only use data from combinations of (model, dataset) where we have accuracy data for all the experiment types

#     save the resulting table in csv format

#     You can use the method defined in this code to fetch the accuracies for the experiments. Use the "logit-accuracy" method for the accuracies.
#     """
#     pass

def generate_accuracy_ranking_table():
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import os

    # Initialize a dictionary to store accuracies
    accuracy_data = {dataset: {exp_type: [] for exp_type in experiment_types} for dataset in datasets}

    # Collect accuracy data
    for model in models:
        for dataset in datasets:
            model_data_complete = True
            model_accuracies = []
            for experiment_type in experiment_types:
                accuracy = get_average_accuracy(model, dataset, experiment_type, accuracy_method)
                if accuracy is None:
                    model_data_complete = False
                    break
                model_accuracies.append(accuracy)
                accuracy_data[dataset][experiment_type].append(accuracy)

            if not model_data_complete:
                # If any experiment type is missing, skip this model-dataset combination
                for experiment_type in experiment_types:
                    if model_accuracies:
                        accuracy_data[dataset][experiment_type].pop()

    # Calculate average accuracies and ranks
    avg_accuracies = {dataset: {} for dataset in datasets}
    avg_ranks = {dataset: {} for dataset in datasets}

    for dataset in datasets:
        for experiment_type in experiment_types:
            if accuracy_data[dataset][experiment_type]:
                avg_accuracies[dataset][experiment_type] = np.mean(accuracy_data[dataset][experiment_type])
            else:
                avg_accuracies[dataset][experiment_type] = None

        # Calculate ranks
        accuracies = [avg_accuracies[dataset][exp_type] for exp_type in experiment_types]
        if None not in accuracies:
            ranks = pd.Series(accuracies).rank(ascending=False).tolist()
            for i, exp_type in enumerate(experiment_types):
                avg_ranks[dataset][exp_type] = ranks[i]
        else:
            for exp_type in experiment_types:
                avg_ranks[dataset][exp_type] = None

    # Calculate overall average rank across datasets
    overall_avg_ranks = {exp_type: [] for exp_type in experiment_types}
    for dataset in datasets:
        for exp_type in experiment_types:
            if avg_ranks[dataset][exp_type] is not None:
                overall_avg_ranks[exp_type].append(avg_ranks[dataset][exp_type])

    for exp_type in experiment_types:
        if overall_avg_ranks[exp_type]:
            overall_avg_ranks[exp_type] = np.mean(overall_avg_ranks[exp_type])
        else:
            overall_avg_ranks[exp_type] = None

    # Prepare data for CSV
    csv_data = []
    header = [""] + experiment_types
    csv_data.append(header)
    csv_data.append(["Overall Average Rank"] + [overall_avg_ranks[exp_type] for exp_type in experiment_types])

    for dataset in datasets:
        row = [dataset] + [avg_accuracies[dataset][exp_type] for exp_type in experiment_types]
        csv_data.append(row)

    # Save to CSV
    results_dir = "results_and_figures"
    os.makedirs(results_dir, exist_ok=True)
    csv_file = Path(results_dir) / "accuracy_ranking_table.csv"
    pd.DataFrame(csv_data).to_csv(csv_file, index=False, header=False)

    print(f"Accuracy ranking table saved to {csv_file}")



# def generate_twonn_validation(method="twonn"):
#     """
#     the mle_estimator is set to mle_50. This is one method to calculate intrinsic dimension. Another method is to use the twonn estimator


#     Loop over all possible experiments and and make two lists: one list of id estimates using mle_50 (the preset mle_estimator) 
#     and the other list  using the corresponding twonn estimates.
    
#      Then, plot them on a scatterplot and calculate the line of best fit and the correlation coeeficient.

#     plot the line of best fit on the scatterplot and save it.
#     also save the data corresponding to the line of best fit to a file
#     """
#     pass




# def generate_expanded_icl_figure():
#     """
#     for the models:
#         - "meta-llama/Llama-2-13b-hf"
#         - "meta-llama/Meta-Llama-3-8B"
    
#     and the datasets:
#         - qnli
#         - commonsense_qa
#         - mmlu

#     I did ICL with more k-values to get a better understanding of the impact of k on ID:

#     specifically, I used the following values for k
#         - 0
#         - 1
#         - 2
#         - 5
#         - 7
#         - 10
#         - 12
#         - 14
#         - 16
#         - 18
#         - 20
    
#     for each combination of model and dataset

#     1) a barplot, with each icl config as a category and the y-value as accuracy
#     2) a line plot, with the x-axis as the layer idx and the y-axis as the ID pattern for that 
#     3) a line plot, with the x-axis as the value of k and the y-axis as the AUC of id curve


#     This gives the graphs for each (model, dataset) combo
#     make such a graph for all the aforementioned (model, dataset combinations) and save them to a file called expanded_icl_figures.pdf
#     """
#     pass


# def generate_single_icl_figure(model="meta-llama/Meta-Llama-3-8B", dataset="mmlu"):
def generate_single_icl_figure(model="meta-llama/Llama-2-13b-hf", dataset="mmlu"):
    """
    Generate a single ICL figure for a specified model and dataset.
    """
    set_plot_theme()

    k_values = [0, 1, 2, 5, 7, 10, 12, 14, 16, 18, 20]
    results_dir = "results_and_figures"
    os.makedirs(results_dir, exist_ok=True)
    plot_file = Path(results_dir) / f"icl_figure_single.pdf"

    try:
        # Prepare data for the bar plot (accuracy)
        accuracy_data = []
        auc_data = []

        for k in k_values:
            experiment_type = f"icl-{k}"
            accuracy = get_average_accuracy(model, dataset, experiment_type, accuracy_method)
            id_values = get_intrinsic_dimensions(model, dataset, experiment_type, mle_estimator)

            if accuracy is not None:
                accuracy_data.append({'k': k, 'Accuracy': accuracy})

            if id_values is not None:
                layers = len(id_values)
                x_values = np.arange(1, layers + 1)
                # Normalize x_values to [0, 1] range
                x_normalized = (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values))
                normalized_auc_value = auc(x_normalized, id_values)
                auc_data.append({'k': k, 'AUC': normalized_auc_value})

        # Create plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Bar plot for accuracy
        accuracy_df = pd.DataFrame(accuracy_data)
        sns.barplot(x='k', y='Accuracy', data=accuracy_df, ax=axes[0], palette='tab10')
        axes[0].set_title(f'Accuracy by k\nModel: {model}, Dataset: {dataset}')
        axes[0].set_xlabel('k')
        axes[0].set_ylabel('Accuracy')

        # Line plot for ID pattern
        palette = sns.color_palette('tab10', len(k_values))
        for idx, k in enumerate(k_values):
            experiment_type = f"icl-{k}"
            id_values = get_intrinsic_dimensions(model, dataset, experiment_type, mle_estimator)
            if id_values is not None:
                axes[1].plot(range(1, len(id_values) + 1), id_values, label=f'k={k}', color=palette[idx])

        axes[1].set_title(f'Intrinsic Dimension by Layer\nModel: {model}, Dataset: {dataset}')
        axes[1].set_xlabel('Layer Index')
        axes[1].set_ylabel('Intrinsic Dimension')
        axes[1].legend(title='k')

        # Line plot for AUC of ID curve
        auc_df = pd.DataFrame(auc_data)
        sns.lineplot(x='k', y='AUC', data=auc_df, ax=axes[2], palette='tab10')
        axes[2].set_title(f'Normalized AUC of ID Curve by k\nModel: {model}, Dataset: {dataset}')
        axes[2].set_xlabel('k')
        axes[2].set_ylabel('Normalized AUC')

        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close(fig)

        print(f"ICL figure saved to {plot_file}")

    except Exception as e:
        print(f"Error processing model {model} with dataset {dataset}: {e}")




def generate_expanded_icl_figure():
    """
    Generate expanded ICL figures for specified models and datasets.
    """
    set_plot_theme()

    models = [
        "meta-llama/Llama-2-13b-hf",
        "meta-llama/Meta-Llama-3-8B"
    ]
    datasets = ["qnli", "commonsense_qa", "mmlu"]
    k_values = [0, 1, 2, 5, 7, 10, 12, 14, 16, 18, 20]
    results_dir = "results_and_figures"
    os.makedirs(results_dir, exist_ok=True)
    plot_file = Path(results_dir) / "expanded_icl_figures.pdf"

    with PdfPages(plot_file) as pdf:
        for model in models:
            for dataset in datasets:
                try:
                    # Prepare data for the bar plot (accuracy)
                    accuracy_data = []
                    auc_data = []

                    for k in k_values:
                        experiment_type = f"icl-{k}"
                        accuracy = get_average_accuracy(model, dataset, experiment_type, accuracy_method)
                        id_values = get_intrinsic_dimensions(model, dataset, experiment_type, mle_estimator)

                        if accuracy is not None:
                            accuracy_data.append({'k': k, 'Accuracy': accuracy})

                        if id_values is not None:
                            layers = len(id_values)
                            x_values = np.arange(1, layers + 1)
                            # Normalize x_values to [0, 1] range
                            x_normalized = (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values))
                            normalized_auc_value = auc(x_normalized, id_values)
                            auc_data.append({'k': k, 'AUC': normalized_auc_value})

                    # Create plots
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                    # Bar plot for accuracy
                    accuracy_df = pd.DataFrame(accuracy_data)
                    sns.barplot(x='k', y='Accuracy', data=accuracy_df, ax=axes[0])
                    axes[0].set_title(f'Accuracy by k\nModel: {model}, Dataset: {dataset}')
                    axes[0].set_xlabel('k')
                    axes[0].set_ylabel('Accuracy')

                    # Line plot for ID pattern
                    for k in k_values:
                        experiment_type = f"icl-{k}"
                        id_values = get_intrinsic_dimensions(model, dataset, experiment_type, mle_estimator)
                        if id_values is not None:
                            axes[1].plot(range(1, len(id_values) + 1), id_values, label=f'k={k}')

                    axes[1].set_title(f'Intrinsic Dimension by Layer\nModel: {model}, Dataset: {dataset}')
                    axes[1].set_xlabel('Layer Index')
                    axes[1].set_ylabel('Intrinsic Dimension')
                    axes[1].legend(title='k')

                    # Line plot for AUC of ID curve
                    auc_df = pd.DataFrame(auc_data)
                    sns.lineplot(x='k', y='AUC', data=auc_df, ax=axes[2])
                    axes[2].set_title(f'Normalized AUC of ID Curve by k\nModel: {model}, Dataset: {dataset}')
                    axes[2].set_xlabel('k')
                    axes[2].set_ylabel('Normalized AUC')

                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

                except Exception as e:
                    print(f"Error processing model {model} with dataset {dataset}: {e}")

    print(f"Expanded ICL figures saved to {plot_file}")




def generate_detailed_ft():
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np

    # Define models, datasets, and checkpoints
    models = ["meta-llama/Llama-2-13b-hf", "meta-llama/Meta-Llama-3-8B"]
    datasets = ["qnli", "commonsense_qa", "mmlu"]
    checkpoints = [62, 124, 186, 248, 310, 372, 434, 496, 558, 620, 682, 744, 806, 868, 930]

    results_dir = "results_and_figures"
    os.makedirs(results_dir, exist_ok=True)
    plot_file = Path(results_dir) / "detailed_ft_figures.pdf"

    with PdfPages(plot_file) as pdf:
        for model in models:
            for dataset in datasets:
                try:
                    # Prepare data for the plots
                    training_accuracies = []
                    validation_accuracies = []
                    train_auc_values = []
                    test_auc_values = []
                    train_id_data = {checkpoint: [] for checkpoint in checkpoints}
                    test_id_data = {checkpoint: [] for checkpoint in checkpoints}

                    for checkpoint in checkpoints:
                        # Fetch accuracy data
                        train_accuracy = get_average_accuracy(model, dataset, "detailed-ft", "logit-accuracy", checkpoint, "train")
                        val_accuracy = get_average_accuracy(model, dataset, "detailed-ft", "logit-accuracy", checkpoint, "validation")
                        
                        train_id_values = get_intrinsic_dimensions(model, dataset, "detailed-ft", mle_estimator, checkpoint, "train")
                        test_id_values = get_intrinsic_dimensions(model, dataset, "detailed-ft", mle_estimator, checkpoint, "validation")

                        if train_accuracy is not None:
                            training_accuracies.append(train_accuracy)
                        if val_accuracy is not None:
                            validation_accuracies.append(val_accuracy)
                        if train_id_values is not None:
                            train_id_data[checkpoint] = train_id_values
                            # Calculate AUC for training ID values
                            layers = len(train_id_values)
                            x_values = np.arange(1, layers + 1)
                            x_normalized = (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values))
                            train_auc_value = auc(x_normalized, train_id_values)
                            train_auc_values.append(train_auc_value)
                        if test_id_values is not None:
                            test_id_data[checkpoint] = test_id_values
                            # Calculate AUC for test ID values
                            layers = len(test_id_values)
                            x_values = np.arange(1, layers + 1)
                            x_normalized = (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values))
                            test_auc_value = auc(x_normalized, test_id_values)
                            test_auc_values.append(test_auc_value)

                    # Create plots
                    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

                    # Line plot for training accuracy
                    axes[0, 0].plot(checkpoints, training_accuracies, marker='o')
                    axes[0, 0].set_title(f'Training Accuracy\nModel: {model}, Dataset: {dataset}')
                    axes[0, 0].set_xlabel('Checkpoint')
                    axes[0, 0].set_ylabel('Accuracy')

                    # Line plot for validation accuracy
                    axes[0, 1].plot(checkpoints, validation_accuracies, marker='o')
                    axes[0, 1].set_title(f'Validation Accuracy\nModel: {model}, Dataset: {dataset}')
                    axes[0, 1].set_xlabel('Checkpoint')
                    axes[0, 1].set_ylabel('Accuracy')

                    # Line plot for training ID values
                    for checkpoint, id_values in train_id_data.items():
                        if id_values:
                            axes[1, 0].plot(range(1, len(id_values) + 1), id_values, label=f'Checkpoint {checkpoint}')
                    axes[1, 0].set_title(f'Training Intrinsic Dimension by Layer\nModel: {model}, Dataset: {dataset}')
                    axes[1, 0].set_xlabel('Layer Index')
                    axes[1, 0].set_ylabel('Intrinsic Dimension')
                    axes[1, 0].legend(title='Checkpoint')

                    # Line plot for test ID values
                    for checkpoint, id_values in test_id_data.items():
                        if id_values:
                            axes[1, 1].plot(range(1, len(id_values) + 1), id_values, label=f'Checkpoint {checkpoint}')
                    axes[1, 1].set_title(f'Test Intrinsic Dimension by Layer\nModel: {model}, Dataset: {dataset}')
                    axes[1, 1].set_xlabel('Layer Index')
                    axes[1, 1].set_ylabel('Intrinsic Dimension')
                    axes[1, 1].legend(title='Checkpoint')

                    # Line plot for training AUC values
                    axes[2, 0].plot(checkpoints, train_auc_values, marker='o')
                    axes[2, 0].set_title(f'Training AUC of ID Curve\nModel: {model}, Dataset: {dataset}')
                    axes[2, 0].set_xlabel('Checkpoint')
                    axes[2, 0].set_ylabel('AUC')

                    # Line plot for test AUC values
                    axes[2, 1].plot(checkpoints, test_auc_values, marker='o')
                    axes[2, 1].set_title(f'Test AUC of ID Curve\nModel: {model}, Dataset: {dataset}')
                    axes[2, 1].set_xlabel('Checkpoint')
                    axes[2, 1].set_ylabel('AUC')

                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

                except Exception as e:
                    print(f"Error processing model {model} with dataset {dataset}: {e}")

    print(f"Detailed fine-tuning figures saved to {plot_file}")



def generate_twonn_validation(method="twonn"):
    set_plot_theme()

    # Lists to store ID estimates
    mle_estimates = []
    twonn_estimates = []

    # Loop over all models, datasets, and experiment types
    for model in models:
        for dataset in datasets:
            for experiment_type in experiment_types:
                # Get ID estimates using mle_50
                mle_id_values = get_intrinsic_dimensions(model, dataset, experiment_type, mle_estimator)
                # Get ID estimates using twonn
                twonn_id_values = get_intrinsic_dimensions(model, dataset, experiment_type, method)

                if mle_id_values is not None and twonn_id_values is not None:
                    # Ensure both lists have the same length
                    if len(mle_id_values) == len(twonn_id_values):
                        mle_estimates.extend(mle_id_values)
                        twonn_estimates.extend(twonn_id_values)

    # Convert lists to numpy arrays for analysis
    mle_estimates = np.array(mle_estimates)
    twonn_estimates = np.array(twonn_estimates)

    # Calculate the line of best fit
    slope, intercept, r_value, p_value, std_err = linregress(mle_estimates, twonn_estimates)

    # Plot the scatterplot using Seaborn
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=mle_estimates, y=twonn_estimates, alpha=0.5, label='Data points')
    plt.plot(mle_estimates, slope * mle_estimates + intercept, color='red', label=f'Best fit line (r={r_value:.2f})')
    plt.xlabel('MLE ID Estimates')
    plt.ylabel('TwoNN ID Estimates')
    plt.title('Scatterplot of MLE vs TwoNN ID Estimates')
    plt.legend()
    plt.tight_layout()

    # Save the plot
    results_dir = "results_and_figures"
    os.makedirs(results_dir, exist_ok=True)
    plot_file = Path(results_dir) / "twonn_validation_scatterplot.pdf"
    plt.savefig(plot_file)
    plt.close()

    # Save the line of best fit data
    fit_data = {
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
        "p_value": p_value,
        "std_err": std_err
    }
    fit_data_file = Path(results_dir) / "twonn_validation_fit_data.json"
    with fit_data_file.open('w') as f:
        json.dump(fit_data, f, indent=4)

    print(f"Scatterplot saved to {plot_file}")
    print(f"Line of best fit data saved to {fit_data_file}")




def generate_id_by_experiment_type_boxplot():
    """
    Create boxplots using all the intrinsic dimension measurements across all experiments and visualize their distributions by experiment type.
    """
    set_plot_theme()

    # Dictionary to store intrinsic dimension values by experiment type
    id_data_by_experiment_type = {experiment_type: [] for experiment_type in experiment_types}

    # Collect intrinsic dimension data
    for model in models:
        for dataset in datasets:
            for experiment_type in experiment_types:
                id_values = get_intrinsic_dimensions(model, dataset, experiment_type, mle_estimator)
                if id_values is not None:
                    id_data_by_experiment_type[experiment_type].extend(id_values)

    # Prepare data for seaborn
    data = []
    for experiment_type, id_values in id_data_by_experiment_type.items():
        for id_value in id_values:
            data.append({'Experiment Type': experiment_type, 'Intrinsic Dimension': id_value})
    data = pd.DataFrame(data)

    # Create a boxplot
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Experiment Type', y='Intrinsic Dimension', data=data)
    plt.title('Intrinsic Dimension Distribution by Experiment Type', fontsize=16)
    plt.xlabel('Experiment Type', fontsize=14)
    plt.ylabel('Intrinsic Dimension', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    results_dir = "results_and_figures"
    os.makedirs(results_dir, exist_ok=True)
    plot_file = Path(results_dir) / "id_by_experiment_type_boxplot.pdf"
    plt.savefig(plot_file)
    plt.close()

    print(f"Boxplot saved to {plot_file}")




# def generate_correlation_data():
#     """
#     i want to explore the correlations between different types
    
#     """
#     pass




def generate_all():
    # Generate AUC data
    print("Generating AUC normalized by layer data...")
    generate_auc_data()
    print("------" * 40)

    # Generate AUC Boxplot
    print("Generating AUC Boxplot")
    generate_auc_boxplot()
    print("------" * 40)

    # Generate Normalized Signed Area Difference Heatmap
    print("Normalized Signed Area Difference Heatmap")
    generate_inter_method_auc_heatmap()
    print("------" * 40)

    # Generate Two Column Summary
    print("Generating Two Column Summary")
    generate_two_column_summary()
    print("------" * 40)

    # Generate Accuracy Ranking Table
    print("Generating Accuracy Ranking Table")
    generate_accuracy_ranking_table()
    print("------" * 40)

    # Generate TwoNN Validation
    print("Generating TwoNN Validation")
    generate_twonn_validation()
    print("------" * 40)

    # Generate ID by Model Boxplot
    print("Generating ID by experiment_type Boxplot")
    generate_id_by_experiment_type_boxplot()
    print("------" * 40)





def main():
    parser = argparse.ArgumentParser(description="Generate figures based on the provided argument.")
    choices = [
        "auc_data", 
        "auc_boxplot", 
        "all", 
        "inter_method_auc_heatmap",
        "two_column_summary",
        "acc_ranking_table",
        "twonn_validation",
        "id_range_boxplot",
        "generate_expanded_icl",
        "generate_detailed_ft_pdf",
        "generate_single_icl_expanded"
        ]


    parser.add_argument("figure", type=str, choices=choices, help="Figure to generate (figure1, figure2, or figure3)")
    args = parser.parse_args()

    if args.figure == "generate_expanded_icl":
        generate_expanded_icl_figure()
    
    if args.figure == "generate_single_icl_expanded":
        generate_single_icl_figure()

    if args.figure == "auc_data": # done
        generate_auc_data()

    if args.figure == "auc_boxplot": # done
        generate_auc_boxplot()

    

    if args.figure == "inter_method_auc_heatmap": # done
        generate_inter_method_auc_heatmap()

    if args.figure == "two_column_summary": # done
        generate_two_column_summary()

    if args.figure == 'acc_ranking_table':  # done
        generate_accuracy_ranking_table()

    if args.figure == 'twonn_validation': # done
        generate_twonn_validation()

    if args.figure == 'id_range_boxplot': # done
        generate_id_by_experiment_type_boxplot()

    if args.figure == "correlation_table":
        generate_correlation_table()

    if args.figure == "generate_detailed_ft_pdf":
        generate_detailed_ft()


    if args.figure == "all":
        generate_all()

    if args.figure == 'all_plz':
        generate_all()



if __name__ == "__main__":
    main()
