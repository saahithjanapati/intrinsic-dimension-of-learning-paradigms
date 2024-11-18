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
    sns.set_theme(style="white")
    sns.set_palette("deep")


def get_average_accuracy(model, dataset, experiment_type, accuracy_type, checkpoint_number=None, split=None, lora_r=None):
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
        if lora_r == None or lora_r == 64:
            path_to_accuracy = PATH_TO_RESULTS_DIR / f"detailed-ft/accuracy_results/{model}_checkpoint_{checkpoint_number}/{dataset}/{split}/0-shot/"
        
        else:
            path_to_accuracy = PATH_TO_RESULTS_DIR / f"detailed-ft/accuracy_results/{model}-lora_r_{lora_r}_checkpoint_{checkpoint_number}/{dataset}/{split}/0-shot/"


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



def get_intrinsic_dimensions(model, dataset, experiment_type, estimator, checkpoint_number=None, split=None, lora_r=None):
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

        if lora_r == None or lora_r == 64:
            path_to_id /= f"detailed-ft/{model}_checkpoint_{checkpoint_number}/{dataset}/{split}/0-shot/"
            path_to_id /= f"{estimator}.json"

        else:
            path_to_id /= f"detailed-ft/{model}-lora_r_{lora_r}_checkpoint_{checkpoint_number}/{dataset}/{split}/0-shot/"
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
        "meta-llama/Llama-2-13b-hf",
        ]

datasets = ["sst2", "cola", "qnli", "qqp", "mnli", "ag_news", "commonsense_qa", "mmlu"]
mle_estimator = "twonn"
experiment_types = ["icl-0", "icl-1", "icl-2", "icl-5", "icl-10", "finetune 1k"]
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
    # set_plot_theme()

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
        # Rename "finetune 1k" to "fine-tune"
        exp_type_label = "fine-tune" if exp_type == "finetune 1k" else exp_type
        for auc in auc_values:
            data.append({'Experiment Type': exp_type_label, 'Normalized AUC': auc})
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
                if id_values is not None:
                    id_data[experiment_type] = id_values[1:]  # cut out layer 0

            # Ensure we have data for all experiment types
            if len(id_data) != len(experiment_types):
                continue

            # Get the number of layers for the model
            layers = get_num_layers(model)
            if layers is None:
                continue

            layers -= 1  # cut out layer 0

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

    # Rename "finetune 1k" to "fine-tune" in the DataFrame
    signed_areas_df.rename(index={"finetune 1k": "fine-tune"}, columns={"finetune 1k": "fine-tune"}, inplace=True)

    # Generate the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(signed_areas_df, annot=True, cmap="coolwarm", center=0)
    plt.title('Average Normalized Signed Difference of Area Between ID With Different Learning Types', fontsize=16)
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





def generate_comparison_figures():
    set_plot_theme()
    results_dir = "results_and_figures"
    os.makedirs(results_dir, exist_ok=True)

    mle_estimator = "twonn"

    # datasets = ["mmlu"]
    # models = ["meta-llama/Meta-Llama-3-8B",]

    for dataset in datasets:
        fig, axes = plt.subplots(4, 3, figsize=(21, 24))  # 4 models x 3 plots each

        # Add dataset name as a bold title at the top of the figure
        fig.suptitle(f"Dataset: {dataset}", fontsize=24)

        for i, model in enumerate(models):
            model_name = model.split("/")[1].strip("-hf")

            # Prepare data for the bar plot (accuracy)
            accuracy_data = []
            auc_data = []
            id_data = {}

            for experiment_type in experiment_types:
                experiment_type_label = "fine-tune" if experiment_type == "finetune 1k" else experiment_type
                accuracy = get_average_accuracy(model, dataset, experiment_type, accuracy_method)
                id_values = get_intrinsic_dimensions(model, dataset, experiment_type, mle_estimator)
                
                if accuracy is not None:
                    accuracy_data.append({'Experiment Type': experiment_type_label, 'Accuracy': accuracy})

                if id_values is not None:
                    x_values = np.arange(1, len(id_values) + 1)
                    x_normalized = (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values))
                    normalized_auc = auc(x_normalized, id_values)
                    auc_data.append({'Experiment Type': experiment_type_label, 'Normalized AUC': normalized_auc})
                    id_data[experiment_type_label] = id_values

            accuracy_df = pd.DataFrame(accuracy_data)
            auc_df = pd.DataFrame(auc_data)

            # Bar plot for accuracy
            sns.barplot(x='Experiment Type', y='Accuracy', data=accuracy_df, ax=axes[i, 0])
            axes[i, 0].set_title(f'Accuracy by Experiment Type for {model_name}', fontsize=12)
            axes[i, 0].set_xlabel('Experiment Type')
            axes[i, 0].set_ylabel('Accuracy')
            axes[i, 0].set_ylim(0, 1)  # Ensure y-axis spans 0-1
            axes[i, 0].tick_params(axis='x', rotation=45)

            # Line plot for intrinsic dimension
            for experiment_type_label, id_values in id_data.items():
                axes[i, 1].plot(range(1, len(id_values) + 1), id_values, label=experiment_type_label)

            axes[i, 1].set_title(f'Intrinsic Dimension Curves for {model_name}', fontsize=12)
            axes[i, 1].set_xlabel('Layer Index')
            axes[i, 1].set_ylabel('Intrinsic Dimension')
            axes[i, 1].legend(title='Experiment Type')

            # Bar plot for normalized AUC
            sns.barplot(x='Experiment Type', y='Normalized AUC', data=auc_df, ax=axes[i, 2])
            axes[i, 2].set_title(f'Normalized AUC by Experiment Type for {model_name}', fontsize=12)
            axes[i, 2].set_xlabel('Experiment Type')
            axes[i, 2].set_ylabel('Normalized AUC')
            axes[i, 2].tick_params(axis='x', rotation=45)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the suptitle
        plot_file = Path(results_dir) / f"comparison-{dataset}-{mle_estimator}.pdf"
        # plot_file = Path(results_dir) / f"comparison-{dataset}-{mle_estimator}-single.pdf"

        plt.savefig(plot_file)
        plt.close()

        print(f"Comparison figure saved to {plot_file}")





def generate_two_column_summary(MODEL_NAME="meta-llama/Meta-Llama-3-8B", dataset="mmlu", include_third_graph=True):

    set_plot_theme()
    # Prepare data for the bar plot (accuracy)
    accuracy_data = []
    auc_data = []
    for experiment_type in experiment_types:
        experiment_type_label = experiment_type
        if experiment_type == "finetune 1k":
            experiment_type_label = "fine-tune"
        accuracy = get_average_accuracy(MODEL_NAME, dataset, experiment_type, accuracy_method)
        id_values = get_intrinsic_dimensions(MODEL_NAME, dataset, experiment_type, mle_estimator)
        
        if accuracy is not None:
            accuracy_data.append({'Experiment Type': experiment_type_label, 'Accuracy': accuracy})
        
        if id_values is not None:
            # Normalize x_values to [0, 1] range
            x_values = np.arange(1, len(id_values) + 1)
            x_normalized = (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values))
            normalized_auc = auc(x_normalized, id_values)
            auc_data.append({'Experiment Type': experiment_type_label, 'Normalized AUC': normalized_auc})

    accuracy_df = pd.DataFrame(accuracy_data)
    auc_df = pd.DataFrame(auc_data)

    # Prepare data for the line plot (intrinsic dimension)
    id_data = {}
    for experiment_type in experiment_types:
        id_values = get_intrinsic_dimensions(MODEL_NAME, dataset, experiment_type, mle_estimator)
        if id_values is not None:
            if experiment_type == "finetune 1k":
                experiment_type = "fine-tune"
            id_data[experiment_type] = id_values

    # Determine the number of subplots based on the include_third_graph flag
    num_plots = 3 if include_third_graph else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(21, 6))

    model_name = MODEL_NAME.split("/")[1].strip("-hf")

    # Bar plot for accuracy
    sns.barplot(x='Experiment Type', y='Accuracy', data=accuracy_df, ax=axes[0])
    axes[0].set_title(f'Accuracy by Experiment Type\nModel: {model_name}, Dataset: {dataset}')
    axes[0].set_xlabel('Experiment Type')
    axes[0].set_ylabel('Accuracy')
    axes[0].tick_params(axis='x', rotation=45)

    # Line plot for intrinsic dimension
    for experiment_type, id_values in id_data.items():
        experiment_type_label = experiment_type
        if experiment_type == "finetune 1k":
            experiment_type_label = "fine-tune"
        axes[1].plot(range(1, len(id_values) + 1), id_values, label=experiment_type_label)

    axes[1].set_title(f'Intrinsic Dimension by Layer\nModel: {model_name}, Dataset: {dataset}')
    axes[1].set_xlabel('Layer Index')
    axes[1].set_ylabel('Intrinsic Dimension')
    axes[1].legend(title='Experiment Type')

    # Bar plot for normalized AUC, if include_third_graph is True
    if include_third_graph:
        sns.barplot(x='Experiment Type', y='Normalized AUC', data=auc_df, ax=axes[2])
        axes[2].set_title(f'Normalized AUC by Experiment Type\nModel: {model_name}, Dataset: {dataset}')
        axes[2].set_xlabel('Experiment Type')
        axes[2].set_ylabel('Normalized AUC')
        axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # Save the plot to a PDF file
    results_dir = "results_and_figures"
    os.makedirs(results_dir, exist_ok=True)
    plot_file = Path(results_dir) / "two_column_summary.pdf"
    plt.savefig(plot_file)
    plt.close()

    print(f"Two-column summary saved to {plot_file}")



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
            for experiment_type in experiment_types:
                accuracy = get_average_accuracy(model, dataset, experiment_type, accuracy_method)

                if accuracy is not None:
                    accuracy_data[dataset][experiment_type].append(accuracy)

    # Calculate average accuracies
    avg_accuracies = {dataset: {} for dataset in datasets}

    for dataset in datasets:
        for experiment_type in experiment_types:
            if accuracy_data[dataset][experiment_type]:
                avg_accuracies[dataset][experiment_type] = np.mean(accuracy_data[dataset][experiment_type])
            else:
                avg_accuracies[dataset][experiment_type] = None

    # Prepare data for CSV
    csv_data = []
    header = ["Dataset"] + experiment_types
    csv_data.append(header)

    for dataset in datasets:
        row = [dataset] + [avg_accuracies[dataset][exp_type] for exp_type in experiment_types]
        csv_data.append(row)

    # Save to CSV
    results_dir = "results_and_figures"
    os.makedirs(results_dir, exist_ok=True)
    csv_file = Path(results_dir) / "accuracy_ranking_table.csv"
    pd.DataFrame(csv_data).to_csv(csv_file, index=False, header=False)

    print(f"Accuracy ranking table saved to {csv_file}")




def generate_auc_by_model_boxplot(experiment_type="icl-5"):
# def generate_auc_by_model_boxplot(experiment_type="finetune 1k"):
    """
    Generates a boxplot of the AUC data for a given experiment type.
    Each box corresponds to a model.
    The x-axis corresponds to the model name and the y-axis corresponds to the normalized AUC value.
    Points for all datasets are included in each boxplot.
    """
    set_plot_theme()

    # Order the models as specified
    model_order = [
        "mistralai/Mistral-7B-v0.3",
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Llama-2-13b-hf"
    ]

    from sklearn.metrics import auc

    # Initialize a dictionary to store AUC values for each model
    auc_data = {model: [] for model in models}

    # Collect AUC data for the specified experiment type
    for model in models:
        for dataset in datasets:
            id_values = get_intrinsic_dimensions(model, dataset, experiment_type, mle_estimator)
            if id_values is not None:
                # Normalize x_values to [0, 1] range
                x_values = np.arange(1, len(id_values) + 1)
                x_normalized = (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values))
                normalized_auc = auc(x_normalized, id_values)
                auc_data[model].append(normalized_auc)

    # Prepare data for seaborn
    data = []
    for model, auc_values in auc_data.items():
        for auc_val in auc_values:
            data.append({'Model': model.split("/")[1], 'Normalized AUC': auc_val})
    data = pd.DataFrame(data)

    # Create a boxplot with skinnier boxes and specified order
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Model', y='Normalized AUC', data=data, width=0.3, order=[m.split("/")[1] for m in model_order])
    if experiment_type == "finetune 1k":
        experiment_type = "fine-tune"
    plt.title(f'Normalized AUC Boxplot for Experiment Type: {experiment_type}', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Normalized AUC', fontsize=14)
    plt.xticks()
    plt.tight_layout()

    # Save the plot
    results_dir = "results_and_figures"
    os.makedirs(results_dir, exist_ok=True)
    plot_file = Path(results_dir) / f"auc_boxplot_{experiment_type}.pdf"
    plt.savefig(plot_file)
    plt.close()

    print(f"Boxplot saved to {plot_file}")



def generate_average_auc_by_model():
    """
    For each model, compute the average AUC across datasets for the icl-5 and finetune 1k experiment types.
    Create a bar plot with the model name on the x-axis and the average AUC on the y-axis.
    Each model should have two bars: one for icl-5 and one for finetune 1k.
    """
    set_plot_theme()

    # Initialize a dictionary to store AUC values
    auc_data = {model: {'icl-5': [], 'finetune 1k': []} for model in models}
    from sklearn.metrics import auc as auc_func

    # Collect AUC data
    for model in models:
        for dataset in datasets:
            for experiment_type in ['icl-5', 'finetune 1k']:
                id_values = get_intrinsic_dimensions(model, dataset, experiment_type, mle_estimator)
                if id_values is not None:
                    # Normalize x_values to [0, 1] range
                    x_values = np.arange(1, len(id_values) + 1)
                    x_normalized = (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values))
                    normalized_auc = auc_func(x_normalized, id_values)
                    auc_data[model][experiment_type].append(normalized_auc)

    # Calculate average AUCs
    avg_auc_data = {model: {} for model in models}
    for model in models:
        for experiment_type in ['icl-5', 'finetune 1k']:
            if auc_data[model][experiment_type]:
                avg_auc_data[model][experiment_type] = np.mean(auc_data[model][experiment_type])
            else:
                avg_auc_data[model][experiment_type] = None

    # Prepare data for plotting
    plot_data = []
    for model in models:
        for experiment_type in ['icl-5', 'finetune 1k']:
            plot_data.append({
                'Model': model,
                'Experiment Type': experiment_type,
                'Average AUC': avg_auc_data[model][experiment_type]
            })
    plot_df = pd.DataFrame(plot_data)

    # Create a bar plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Model', y='Average AUC', hue='Experiment Type', data=plot_df)
    plt.title('Average AUC by Model and Experiment Type', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Average AUC', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    results_dir = "results_and_figures"
    os.makedirs(results_dir, exist_ok=True)
    plot_file = Path(results_dir) / "average_auc_by_model.pdf"
    plt.savefig(plot_file)
    plt.close()

    print(f"Average AUC by model plot saved to {plot_file}")



def generate_single_icl_figure(model="meta-llama/Meta-Llama-3-8B", dataset="mmlu"):
# def generate_single_icl_figure(model="meta-llama/Llama-2-13b-hf", dataset="mmlu"):
    """
    Generate a single ICL figure for a specified model and dataset.
    """
    # set_plot_theme()

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

        # Line plot for AUC of ID curve with dots on each point
        auc_df = pd.DataFrame(auc_data)
        sns.lineplot(x='k', y='AUC', data=auc_df, ax=axes[2], palette='tab10', marker='o')
        axes[2].set_title(f'Normalized AUC of ID Curve by k\nModel: {model}, Dataset: {dataset}')
        axes[2].set_xlabel('k')
        axes[2].set_ylabel('Normalized AUC')
        axes[2].set_xticks(k_values)  # Set integer ticks for x-axis

        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close(fig)

        print(f"ICL figure saved to {plot_file}")

    except Exception as e:
        print(f"Error processing model {model} with dataset {dataset}: {e}")



def generate_id_by_layer_for_checkpoints(
    # model = "meta-llama/Llama-2-8b-hf", 
    model = "meta-llama/Meta-Llama-3-8B",
    dataset = "mmlu", 
    split = "validation"):

    # Define checkpoints
    # checkpoints = [0, 62, 124, 186, 248, 310, 372, 434, 496, 558, 620, 682, 744, 806, 868, 930]
    checkpoints = [0, 62, 124, 186, 248, 310, 372, 434, 496, 558, 620, 682, 744, 806, 868, 930]

    # Prepare data for the plot
    id_data = {checkpoint: [] for checkpoint in checkpoints}

    for checkpoint in checkpoints:
        id_values = get_intrinsic_dimensions(model, dataset, "detailed-ft", mle_estimator, checkpoint, split)
        if id_values is not None:
            id_data[checkpoint] = id_values

    # Create the plot
    plt.figure(figsize=(10, 6))
    # Use a gradient from red to blue
    palette = sns.color_palette("coolwarm", len(checkpoints))

    for idx, (checkpoint, id_values) in enumerate(id_data.items()):
        if id_values:
            color = palette[idx]  # Get color from the palette
            plt.plot(range(1, len(id_values) + 1), id_values, label=f'{checkpoint}', color=color)

    plt.title(f'Intrinsic Dimension by Layer\nModel: {model}, Dataset: {dataset}, Split: {split}')
    plt.xlabel('Layer Index')
    plt.ylabel('Intrinsic Dimension')
    plt.legend(title='# Training Steps', loc='upper right')
    plt.tight_layout()

    # Save or show the plot
    results_dir = "results_and_figures"
    os.makedirs(results_dir, exist_ok=True)
    plot_file = Path(results_dir) / f"id_by_layer_single.pdf"
    plt.savefig(plot_file)
    plt.close()

    print(f"Plot saved to {plot_file}")



def generate_detailed_ft():
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np

    # Define models, datasets, and checkpoints
    models = ["meta-llama/Llama-2-13b-hf", "meta-llama/Meta-Llama-3-8B"]
    datasets = ["qnli", "commonsense_qa", "mmlu"]
    checkpoints = [0, 62, 124, 186, 248, 310, 372, 434, 496, 558, 620, 682, 744, 806, 868, 930]

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




def generate_combined_ft_figure_for_model_dataset(
    model="meta-llama/Meta-Llama-3-8B",
    # dataset="mmlu",
    dataset = "mmlu",
    split="validation"
):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    # Define checkpoints
    checkpoints = [0, 62, 124, 186, 248, 310, 372, 434, 496, 558, 620, 682, 744, 806, 868, 930]

    # Prepare data for the plots
    train_id_data = {checkpoint: [] for checkpoint in checkpoints}
    test_id_data = {checkpoint: [] for checkpoint in checkpoints}
    training_accuracies = []
    validation_accuracies = []
    train_auc_values = []
    test_auc_values = []

    for checkpoint in checkpoints:
        # Fetch intrinsic dimension data
        train_id_values = get_intrinsic_dimensions(model, dataset, "detailed-ft", mle_estimator, checkpoint, "train")
        test_id_values = get_intrinsic_dimensions(model, dataset, "detailed-ft", mle_estimator, checkpoint, "validation")

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

        # Fetch accuracy data
        train_accuracy = get_average_accuracy(model, dataset, "detailed-ft", "logit-accuracy", checkpoint, "train")
        val_accuracy = get_average_accuracy(model, dataset, "detailed-ft", "logit-accuracy", checkpoint, "validation")

        if train_accuracy is not None:
            training_accuracies.append(train_accuracy)
        if val_accuracy is not None:
            validation_accuracies.append(val_accuracy)

    # Create the plots
    plt.figure(figsize=(10, 18))
    palette = sns.color_palette("coolwarm", len(checkpoints))


    model_name = model.split("/")[1].strip("-hf")

    plt.suptitle(f"Fine-tuning results for {model_name} on {dataset}\n", fontsize=16)
    plt.tight_layout(rect=[0, 0.2, 1, 0.6])  # Adjust the rect parameter to add more space


    # Graph 1: Line plot for intrinsic dimension by layer
    plt.subplot(3, 1, 1)
    for idx, (checkpoint, id_values) in enumerate(train_id_data.items()):
        if id_values:
            color = palette[idx]
            plt.plot(range(1, len(id_values) + 1), id_values, label=f'{checkpoint}', color=color)
    plt.title(f'Intrinsic Dimension Curves for Checkpoints on Validation Split')
    plt.xlabel('Layer Index')
    plt.ylabel('Intrinsic Dimension')
    plt.legend(title='# Training Steps', loc='upper left')

    # Graph 2: Line plot for training and validation accuracy
    plt.subplot(3, 1, 2)
    plt.plot(checkpoints, training_accuracies, 'go-', label='Training Accuracy')  # Green solid line with dots
    plt.plot(checkpoints, validation_accuracies, 'mo-', label='Validation Accuracy')  # Purple solid line with dots
    plt.title(f'Accuracy by Checkpoint')
    plt.xlabel('Checkpoint')
    plt.ylabel('Accuracy')
    plt.legend()

    # Graph 3: Line plot for training and validation AUC
    plt.subplot(3, 1, 3)
    plt.plot(checkpoints, train_auc_values, 'go-', label='Training AUC')  # Green solid line with dots
    plt.plot(checkpoints, test_auc_values, 'mo-', label='Validation AUC')  # Purple solid line with dots
    plt.title(f'Normalized AUC by Checkpoint')
    plt.xlabel('Checkpoint')
    plt.ylabel('AUC')
    plt.legend()

    plt.tight_layout()

    # Save the plot
    results_dir = "results_and_figures"
    os.makedirs(results_dir, exist_ok=True)
    plot_file = Path(results_dir) / f"combined_ft_figure_1.pdf"
    plt.savefig(plot_file)
    plt.close()

    print(f"Combined figure saved to {plot_file}")

# Example usage:
# generate_combined_figure_for_model_dataset("meta-llama/Llama-2-13b-hf", "qnli")




def generate_combined_ft_figures():
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    models = ["meta-llama/Llama-2-13b-hf", "meta-llama/Meta-Llama-3-8B"]
    datasets = ["qnli", "commonsense_qa", "mmlu", "sst2", "cola", "ag_news", "mnli", "qqp"]

    checkpoints = [0, 62, 124, 186, 248, 310, 372, 434, 496, 558, 620, 682, 744, 806, 868, 930]

    results_dir = "results_and_figures"
    os.makedirs(results_dir, exist_ok=True)

    for model in models:
        for dataset in datasets:
            # Prepare data for the plots
            train_id_data = {checkpoint: [] for checkpoint in checkpoints}
            test_id_data = {checkpoint: [] for checkpoint in checkpoints}
            training_accuracies = []
            validation_accuracies = []
            train_auc_values = []
            test_auc_values = []

            for checkpoint in checkpoints:
                # Fetch intrinsic dimension data
                train_id_values = get_intrinsic_dimensions(model, dataset, "detailed-ft", mle_estimator, checkpoint, "train")
                test_id_values = get_intrinsic_dimensions(model, dataset, "detailed-ft", mle_estimator, checkpoint, "validation")

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

                # Fetch accuracy data
                train_accuracy = get_average_accuracy(model, dataset, "detailed-ft", "logit-accuracy", checkpoint, "train")
                val_accuracy = get_average_accuracy(model, dataset, "detailed-ft", "logit-accuracy", checkpoint, "validation")

                if train_accuracy is not None:
                    training_accuracies.append(train_accuracy)
                if val_accuracy is not None:
                    validation_accuracies.append(val_accuracy)

            # Create the plots
            plt.figure(figsize=(24, 6))
            palette = sns.color_palette("coolwarm", len(checkpoints))

            # Flatten the lists of intrinsic dimension values across all checkpoints
            train_id_values_flat = [value for values in train_id_data.values() for value in values]
            test_id_values_flat = [value for values in test_id_data.values() for value in values]

            # Calculate the minimum and maximum values across all checkpoints
            y_min = min(min(train_id_values_flat), min(test_id_values_flat)) - 1
            y_max = max(max(train_id_values_flat), max(test_id_values_flat)) + 1


            model_name = model.split('/')[-1].replace('Llama-', 'llama-').replace('Meta-', 'meta-').replace('-hf', '')
            # Add a title that spans the entire figure
            plt.suptitle(f'Model: {model_name}, Dataset: {dataset}', fontsize=16)

            # Graph 1: Line plot for intrinsic dimension by layer (training)
            plt.subplot(1, 4, 1)
            for idx, (checkpoint, id_values) in enumerate(train_id_data.items()):
                if id_values:
                    color = palette[idx]
                    plt.plot(range(1, len(id_values) + 1), id_values, label=f'{checkpoint}', color=color)
            plt.title(f'Intrinsic Dimension Curves for Checkpoints on Training Set')
            plt.xlabel('Layer Index')
            plt.ylabel('Intrinsic Dimension')
            plt.ylim(y_min, y_max)
            plt.legend(title='# Training Steps', loc='center right', bbox_to_anchor=(-0.1, 0.5))

            # Graph 2: Line plot for intrinsic dimension by layer (validation)
            plt.subplot(1, 4, 2)
            for idx, (checkpoint, id_values) in enumerate(test_id_data.items()):
                if id_values:
                    color = palette[idx]
                    plt.plot(range(1, len(id_values) + 1), id_values, label=f'{checkpoint}', color=color)
            plt.title(f'Intrinsic Dimension Curves for Checkpoints on Validation Set')
            plt.xlabel('Layer Index')
            plt.ylabel('Intrinsic Dimension')
            plt.ylim(y_min, y_max)


            # Graph 3: Line plot for training and validation accuracy
            plt.subplot(1, 4, 3)
            plt.plot(checkpoints, training_accuracies, color='green', linestyle='-', marker='o', label='Training Accuracy')  # Green solid line with dots
            plt.plot(checkpoints, validation_accuracies, color='orange', linestyle='-', marker='o', label='Validation Accuracy')  # Orange solid line with dots
            plt.title(f'Accuracy by Checkpoint')
            plt.xlabel('Checkpoint')
            plt.ylabel('Accuracy')
            plt.legend()

            # Graph 4: Line plot for training and validation AUC
            plt.subplot(1, 4, 4)
            plt.plot(checkpoints, train_auc_values, color='green', linestyle='-', marker='o', label='Training AUC')  # Green solid line with dots
            plt.plot(checkpoints, test_auc_values, color='orange', linestyle='-', marker='o', label='Validation AUC')  # Orange solid line with dots
            plt.title(f'Normalized AUC of ID Curve by Checkpoint')
            plt.xlabel('Checkpoint')
            plt.ylabel('Normalized AUC')
            plt.legend()

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the suptitle

            # Save the plot
            plot_file = Path(results_dir) / f"sft-{model_name}-{dataset}.pdf"
            plt.savefig(plot_file)
            plt.close()





def generate_combined_ft_dataset_figures_llama_3_8b():
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    model = "meta-llama/Meta-Llama-3-8B"
    datasets = ["qnli", "commonsense_qa", "mmlu", "sst2", "cola", "ag_news", "mnli", "qqp"]
    checkpoints = [0, 62, 124, 186, 248, 310, 372, 434, 496, 558, 620, 682, 744, 806, 868, 930]

    results_dir = "results_and_figures/combined_ft_dataset/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for dataset in datasets:
        # Prepare data for the plots
        train_id_data_mle = {checkpoint: [] for checkpoint in checkpoints}
        test_id_data_mle = {checkpoint: [] for checkpoint in checkpoints}
        train_id_data_twonn = {checkpoint: [] for checkpoint in checkpoints}
        test_id_data_twonn = {checkpoint: [] for checkpoint in checkpoints}
        training_accuracies = []
        validation_accuracies = []
        train_auc_values_mle = []
        test_auc_values_mle = []
        train_auc_values_twonn = []
        test_auc_values_twonn = []

        for checkpoint in checkpoints:
            # Fetch intrinsic dimension data using mle_50
            train_id_values_mle = get_intrinsic_dimensions(model, dataset, "detailed-ft", "mle_50", checkpoint, "train")
            test_id_values_mle = get_intrinsic_dimensions(model, dataset, "detailed-ft", "mle_50", checkpoint, "validation")

            # Fetch intrinsic dimension data using two_nn
            train_id_values_twonn = get_intrinsic_dimensions(model, dataset, "detailed-ft", "twonn", checkpoint, "train")
            test_id_values_twonn = get_intrinsic_dimensions(model, dataset, "detailed-ft", "twonn", checkpoint, "validation")

            if train_id_values_mle is not None:
                train_id_data_mle[checkpoint] = train_id_values_mle
                # Calculate AUC for training ID values using mle_50
                layers = len(train_id_values_mle)
                x_values = np.arange(1, layers + 1)
                x_normalized = (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values))
                train_auc_value_mle = auc(x_normalized, train_id_values_mle)
                train_auc_values_mle.append(train_auc_value_mle)

            if test_id_values_mle is not None:
                test_id_data_mle[checkpoint] = test_id_values_mle
                # Calculate AUC for test ID values using mle_50
                layers = len(test_id_values_mle)
                x_values = np.arange(1, layers + 1)
                x_normalized = (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values))
                test_auc_value_mle = auc(x_normalized, test_id_values_mle)
                test_auc_values_mle.append(test_auc_value_mle)

            if train_id_values_twonn is not None:
                train_id_data_twonn[checkpoint] = train_id_values_twonn
                # Calculate AUC for training ID values using two_nn
                layers = len(train_id_values_twonn)
                x_values = np.arange(1, layers + 1)
                x_normalized = (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values))
                train_auc_value_twonn = auc(x_normalized, train_id_values_twonn)
                train_auc_values_twonn.append(train_auc_value_twonn)

            if test_id_values_twonn is not None:
                test_id_data_twonn[checkpoint] = test_id_values_twonn
                # Calculate AUC for test ID values using two_nn
                layers = len(test_id_values_twonn)
                x_values = np.arange(1, layers + 1)
                x_normalized = (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values))
                test_auc_value_twonn = auc(x_normalized, test_id_values_twonn)
                test_auc_values_twonn.append(test_auc_value_twonn)

            # Fetch accuracy data
            train_accuracy = get_average_accuracy(model, dataset, "detailed-ft", "logit-accuracy", checkpoint, "train")
            val_accuracy = get_average_accuracy(model, dataset, "detailed-ft", "logit-accuracy", checkpoint, "validation")

            if train_accuracy is not None:
                training_accuracies.append(train_accuracy)
            if val_accuracy is not None:
                validation_accuracies.append(val_accuracy)

        # Create the plots
        plt.figure(figsize=(18, 24))
        palette = sns.color_palette("coolwarm", len(checkpoints))


        # Graph 1: Line plot for intrinsic dimension by layer (mle_50)
        plt.subplot(4, 2, 1)
        for idx, (checkpoint, id_values) in enumerate(train_id_data_mle.items()):
            if id_values:
                color = palette[idx]
                plt.plot(range(1, len(id_values) + 1), id_values, label=f'{checkpoint}', color=color)
        plt.title(f'Intrinsic Dimension by Layer (MLE)\nModel: {model}, Dataset: {dataset} - Training')
        plt.xlabel('Layer Index')
        plt.ylabel('Intrinsic Dimension')
        plt.legend(title='# Training Steps', loc='upper left')

        # Graph 2: Line plot for intrinsic dimension by layer (two_nn)
        plt.subplot(4, 2, 2)
        for idx, (checkpoint, id_values) in enumerate(train_id_data_twonn.items()):
            if id_values:
                color = palette[idx]
                plt.plot(range(1, len(id_values) + 1), id_values, label=f'{checkpoint}', color=color)
        plt.title(f'Intrinsic Dimension by Layer (TwoNN)\nModel: {model}, Dataset: {dataset} - Training')
        plt.xlabel('Layer Index')
        plt.ylabel('Intrinsic Dimension')
        plt.legend(title='# Training Steps', loc='upper left')



        # Graph 3: Line plot for intrinsic dimension by layer (mle_50)
        plt.subplot(4, 2, 3)
        for idx, (checkpoint, id_values) in enumerate(test_id_data_mle.items()):
            if id_values:
                color = palette[idx]
                plt.plot(range(1, len(id_values) + 1), id_values, label=f'{checkpoint}', color=color)
        
        plt.title(f'Intrinsic Dimension by Layer (MLE)\nModel: {model}, Dataset: {dataset} - Validation')
        plt.xlabel('Layer Index')
        plt.ylabel('Intrinsic Dimension')
        plt.legend(title='# Training Steps', loc='upper left')



        # Graph 4: Line plot for intrinsic dimension by layer (two_nn)
        plt.subplot(4, 2, 4)
        for idx, (checkpoint, id_values) in enumerate(test_id_data_twonn.items()):
            if id_values:
                color = palette[idx]
                plt.plot(range(1, len(id_values) + 1), id_values, label=f'{checkpoint}', color=color)
        
        plt.title(f'Intrinsic Dimension by Layer (TwoNN)\nModel: {model}, Dataset: {dataset} - Validation')
        plt.xlabel('Layer Index')
        plt.ylabel('Intrinsic Dimension')
        plt.legend(title='# Training Steps', loc='upper left')



        # Graph 5: Line plot for training and validation accuracy
        plt.subplot(4, 2, 7)
        plt.plot(checkpoints, training_accuracies, color='green', linestyle='-', marker='o', label='Training Accuracy')
        plt.plot(checkpoints, validation_accuracies, color='orange', linestyle='-', marker='o', label='Validation Accuracy')
        plt.title(f'Accuracy by Checkpoint\nModel: {model}, Dataset: {dataset}')
        plt.xlabel('Checkpoint')
        plt.ylabel('Accuracy')
        plt.legend()


        # Graph 6: Line plot for training and validation AUC (mle_50)
        plt.subplot(4, 2, 5)
        plt.plot(checkpoints, train_auc_values_mle, color='green', linestyle='-', marker='o', label='Training AUC (MLE)')
        plt.plot(checkpoints, test_auc_values_mle, color='orange', linestyle='-', marker='o', label='Validation AUC (MLE)')
        plt.title(f'AUC of ID Curve by Checkpoint (MLE)\nModel: {model}, Dataset: {dataset}')
        plt.xlabel('Checkpoint')
        plt.ylabel('AUC')
        plt.legend()

        # Graph 7: Line plot for training and validation AUC (two_nn)
        plt.subplot(4, 2, 6)
        plt.plot(checkpoints, train_auc_values_twonn, color='green', linestyle='-', marker='o', label='Training AUC (TwoNN)')
        plt.plot(checkpoints, test_auc_values_twonn, color='orange', linestyle='-', marker='o', label='Validation AUC (TwoNN)')
        plt.title(f'AUC of ID Curve by Checkpoint (TwoNN)\nModel: {model}, Dataset: {dataset}')
        plt.xlabel('Checkpoint')
        plt.ylabel('AUC')
        plt.legend()

        plt.tight_layout()

        # Save the plot
        model_name = model.split('/')[-1].replace('Llama-', 'llama-').replace('Meta-', 'meta-').replace('-hf', '')
        plot_file = Path(results_dir) / f"combined-sft-{model_name}-{dataset}.png"
        plt.savefig(plot_file)
        plt.close()

        print(f"Combined figure saved to {plot_file}")




def plot_intrinsic_dimension_by_lora_rank(model="meta-llama/Meta-Llama-3-8B", dataset="mmlu", estimator="twonn"):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns


    # Define lora ranks and checkpoints
    lora_ranks = [4, 16, 32, 64, 128]
    lora_ranks = [4, 32, 64, 128]


    checkpoints = [0, 62, 124, 186, 248, 310, 372, 434, 496, 558, 620, 682, 744, 806, 868, 930]
    checkpoints = [62, 496, 930]

    # estimator = "mle_50"

    # Prepare data for the plots
    id_data_by_lora_rank_train = {lora_rank: {checkpoint: [] for checkpoint in checkpoints} for lora_rank in lora_ranks}
    id_data_by_lora_rank_val = {lora_rank: {checkpoint: [] for checkpoint in checkpoints} for lora_rank in lora_ranks}
    auc_data_train = {lora_rank: [] for lora_rank in lora_ranks}
    auc_data_val = {lora_rank: [] for lora_rank in lora_ranks}
    accuracy_data_train = {lora_rank: [] for lora_rank in lora_ranks}
    accuracy_data_val = {lora_rank: [] for lora_rank in lora_ranks}

    for lora_rank in lora_ranks:
        for checkpoint in checkpoints:
            id_values_train = get_intrinsic_dimensions(model, dataset, "detailed-ft", estimator, checkpoint, "train", lora_rank)
            id_values_val = get_intrinsic_dimensions(model, dataset, "detailed-ft", estimator, checkpoint, "validation", lora_rank)
            
            if id_values_train is not None:
                id_data_by_lora_rank_train[lora_rank][checkpoint] = id_values_train
                # Calculate AUC for training ID values
                layers = len(id_values_train)
                x_values = np.arange(1, layers + 1)
                x_normalized = (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values))
                auc_data_train[lora_rank].append(auc(x_normalized, id_values_train))
            
            if id_values_val is not None:
                id_data_by_lora_rank_val[lora_rank][checkpoint] = id_values_val
                # Calculate AUC for validation ID values
                layers = len(id_values_val)
                x_values = np.arange(1, layers + 1)
                x_normalized = (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values))
                auc_data_val[lora_rank].append(auc(x_normalized, id_values_val))
            
            # Fetch accuracy data
            train_accuracy = get_average_accuracy(model, dataset, "detailed-ft", "logit-accuracy", checkpoint, "train", lora_rank)
            val_accuracy = get_average_accuracy(model, dataset, "detailed-ft", "logit-accuracy", checkpoint, "validation", lora_rank)

            if train_accuracy is not None:
                accuracy_data_train[lora_rank].append(train_accuracy)
            if val_accuracy is not None:
                accuracy_data_val[lora_rank].append(val_accuracy)

    # Create the plots
    plt.figure(figsize=(25, 20))
    palette = sns.color_palette("coolwarm", len(checkpoints))

    for idx, lora_rank in enumerate(lora_ranks):
        # Plot for intrinsic dimension by layer (train)
        plt.subplot(4, 5, idx + 1)
        for checkpoint, id_values in id_data_by_lora_rank_train[lora_rank].items():
            if id_values:
                color = palette[checkpoints.index(checkpoint)]
                plt.plot(range(1, len(id_values) + 1), id_values, label=f'Checkpoint {checkpoint}', color=color)
        plt.title(f'ID vs. Checkpoint (Train)\nLora Rank: {lora_rank}')
        plt.xlabel('Layer Index')
        plt.ylabel('Intrinsic Dimension')

        # Plot for intrinsic dimension by layer (validation)
        plt.subplot(4, 5, idx + 6)
        for checkpoint, id_values in id_data_by_lora_rank_val[lora_rank].items():
            if id_values:
                color = palette[checkpoints.index(checkpoint)]
                plt.plot(range(1, len(id_values) + 1), id_values, label=f'Checkpoint {checkpoint}', color=color)
        plt.title(f'ID vs. Checkpoint (Validation)\nLora Rank: {lora_rank}')
        plt.xlabel('Layer Index')
        plt.ylabel('Intrinsic Dimension')

        # Plot for AUC (train and validation)
        plt.subplot(4, 5, idx + 11)
        plt.plot(checkpoints, auc_data_train[lora_rank], linestyle='-', marker='o', label='Train AUC')
        plt.plot(checkpoints, auc_data_val[lora_rank], linestyle='--', marker='o', label='Validation AUC')
        plt.title(f'AUC by Checkpoint\nLora Rank: {lora_rank}')
        plt.xlabel('Checkpoint')
        plt.ylabel('AUC')
        plt.legend()

        # Plot for accuracy (train and validation)
        plt.subplot(4, 5, idx + 16)
        plt.plot(checkpoints, accuracy_data_train[lora_rank], 'g-', marker='o', label='Training Accuracy')
        plt.plot(checkpoints, accuracy_data_val[lora_rank], 'm-', marker='o', label='Validation Accuracy')
        plt.title(f'Accuracy by Checkpoint\nLora Rank: {lora_rank}')
        plt.xlabel('Checkpoint')
        plt.ylabel('Accuracy')
        plt.legend()

    plt.tight_layout()

    # Save the plot
    results_dir = "results_and_figures"
    os.makedirs(results_dir, exist_ok=True)
    plot_file = Path(results_dir) / f"intrinsic_dimension_by_lora_rank_{estimator}_{dataset}.png"
    plt.savefig(plot_file)
    plt.close()

    print(f"Plot saved to {plot_file}")





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



def generate_all_icl_figures():
    """
    Generate ICL figures for specified models and datasets, saving each figure with a filename based on the model and dataset.
    """
    # set_plot_theme()

    # models = ["meta-llama/Llama-2-13b-hf", "meta-llama/Meta-Llama-3-8B"]
    models = ["meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-2-13b-hf"]

    # datasets = ["ag_news-deduped", "qnli-deduped", "qqp-deduped"]
    datasets = ["mmlu", "commonsense_qa", "qnli"]

    k_values = [0, 1, 2, 5, 10, 12, 14, 16, 18, 20]
    results_dir = "results_and_figures"
    os.makedirs(results_dir, exist_ok=True)

    # mle_estimator = "mle_50"
    mle_estimator = "twonn"

    for model in models:
        model_name = model.split('/')[-1].replace('Llama-', 'llama-').replace('Meta-', 'meta-').replace('-hf', '')

        for dataset in datasets:
            # Determine the filename based on the model and dataset
            model_name = model.split('/')[-1].replace('Llama-', 'llama-').replace('Meta-', 'meta-').replace('-hf', '')

            if "-deduped" in dataset:
                dataset_name = dataset.split('-deduped')[0]
            else:
                dataset_name = dataset
            plot_file = Path(results_dir) / f"{model_name}-{dataset}.pdf"

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

                fig.suptitle(f'Model: {model_name}, Dataset: {dataset_name}', fontsize=16)

                # Bar plot for accuracy
                accuracy_df = pd.DataFrame(accuracy_data)
                sns.barplot(x='k', y='Accuracy', data=accuracy_df, ax=axes[0], palette='tab10')
                axes[0].set_title(f'k vs Accuracy')
                axes[0].set_xlabel('k')
                axes[0].set_ylabel('Accuracy')

                # Line plot for ID pattern
                # distinct_palette = sns.color_palette('hsv', len(k_values))  # Use a distinct color palette
                for idx, k in enumerate(k_values):
                    experiment_type = f"icl-{k}"
                    id_values = get_intrinsic_dimensions(model, dataset, experiment_type, mle_estimator)
                    if id_values is not None:
                        axes[1].plot(range(1, len(id_values) + 1), id_values, label=f'k={k}')

                axes[1].set_title(f'Intrinsic Dimension by Layer')
                axes[1].set_xlabel('Layer Index')
                axes[1].set_ylabel('Intrinsic Dimension')
                axes[1].legend(title='k')

                # Line plot for AUC of ID curve with dots on each point
                auc_df = pd.DataFrame(auc_data)
                sns.lineplot(x='k', y='AUC', data=auc_df, ax=axes[2], marker='o')
                axes[2].set_title(f'k vs Normalized AUC of ID Curve')
                axes[2].set_xlabel('k')
                axes[2].set_ylabel('Normalized AUC')
                axes[2].set_xticks(k_values)  # Set integer ticks for x-axis

                plt.tight_layout()
                plt.savefig(plot_file)
                plt.close(fig)

                print(f"ICL figure saved to {plot_file}")

            except Exception as e:
                print(f"Error processing model {model} with dataset {dataset}: {e}")


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
        "generate_single_icl_expanded",
        "generate_all_icl_expanded",
        "id_by_layer_for_checkpoints",
        "generate_combined_ft_figures",
        "combined_ft_figure", # Add this line,
        "comparison_figures",
        "auc_by_model_boxplot",
        "llama-3-8b-combined-ft-datasets",
        "plot_id_by_lora_rank"
    ]

    parser.add_argument("figure", type=str, choices=choices, help="Figure to generate")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B", help="Model name for combined figure")
    parser.add_argument("--dataset", type=str, default="mmlu", help="Dataset name for combined figure")
    args = parser.parse_args()

    # if args.figure == "generate_expanded_icl":
    #     generate_expanded_icl_figure()
    
    if args.figure == "generate_single_icl_expanded":
        generate_single_icl_figure()

    if args.figure == "generate_all_icl_expanded":
        generate_all_icl_figures()

    if args.figure == "auc_data":
        generate_auc_data()

    if args.figure == "auc_boxplot":
        generate_auc_boxplot()

    if args.figure == "auc_by_model_boxplot":
        generate_auc_by_model_boxplot()

    if args.figure == "comparison_figures":
        generate_comparison_figures()

    if args.figure == "inter_method_auc_heatmap":
        generate_inter_method_auc_heatmap()

    if args.figure == "two_column_summary":
        generate_two_column_summary()

    if args.figure == 'acc_ranking_table':
        generate_accuracy_ranking_table()

    if args.figure == 'twonn_validation':
        generate_twonn_validation()

    if args.figure == 'id_range_boxplot':
        generate_id_by_experiment_type_boxplot()

    if args.figure == "generate_detailed_ft_pdf":
        generate_detailed_ft()

    if args.figure == "id_by_layer_for_checkpoints":
        generate_id_by_layer_for_checkpoints()

    if args.figure == "generate_combined_ft_figures":
        generate_combined_ft_figures()

    if args.figure == "llama-3-8b-combined-ft-datasets":
        generate_combined_ft_dataset_figures_llama_3_8b()

    if args.figure == "combined_ft_figure":  # Add this block
        generate_combined_ft_figure_for_model_dataset(model=args.model, dataset=args.dataset)

    if args.figure == "plot_id_by_lora_rank":
        plot_intrinsic_dimension_by_lora_rank()

    if args.figure == "all":
        generate_all()

if __name__ == "__main__":
    main()