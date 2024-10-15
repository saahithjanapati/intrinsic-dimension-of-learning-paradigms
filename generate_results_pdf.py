import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import os

PATH_TO_RESULTS_DIR = Path("concise_results/")  # Ensure this is correctly set to your results directory


def get_average_accuracy(model, dataset, experiment_type, accuracy_type):
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


def get_intrinsic_dimensions(model, dataset, experiment_type, estimator):
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

    return layers



def generate_expanded_icl_pdf(model, datasets, k_shots, accuracy_type):
    experiment_types = []
    for k in k_shots:
        experiment_types.append(f"icl-{k}")
    
    




def generate_main_results_pdf(models, datasets, experiment_types, accuracy_type):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    for dataset in datasets:
        # Add a page for each dataset with landscape orientation
        pdf.add_page(orientation='L')
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, f"Results for Dataset: {dataset}", ln=True, align="C")

        # Create a figure with subplots (4 models * 3 graphs per model = 12 subplots)
        fig, axs = plt.subplots(5, 3, figsize=(25, 15))  # Increased width for larger graphs
        fig.tight_layout(pad=5.0)

        for model_idx, model in enumerate(models):
            # Accuracy graph
            accuracies = []
            for exp_type in experiment_types:
                accuracy = get_average_accuracy(model, dataset, exp_type, accuracy_type)
                if accuracy is None:
                    print(f"Accuracy data not found for {model} - {exp_type}. Skipping.")
                    accuracies.append(0)
                else:
                    accuracies.append(accuracy)

            axs[model_idx, 0].bar(experiment_types, accuracies)
            axs[model_idx, 0].set_title(f"Accuracy for {model.split('/')[-1]}")
            axs[model_idx, 0].set_xlabel("Experiment Type")
            axs[model_idx, 0].set_ylabel("Accuracy")

            # Intrinsic dimension graph with mle_50
            num_layers = get_num_layers(model)
            axs[model_idx, 1].set_prop_cycle(None)  # Reset color cycle for distinct lines
            for exp_type in experiment_types:
                intrinsic_dims_mle = get_intrinsic_dimensions(model, dataset, exp_type, "twonn")
                if intrinsic_dims_mle is None:
                    print(f"Intrinsic dimension (twonn) data not found for {model} - {exp_type}. Skipping.")
                    continue  # Skip plotting if there's an error
                
                try:
                    axs[model_idx, 1].plot(range(num_layers - 1), intrinsic_dims_mle, label=exp_type)
                except:
                    print(f"There was an error with (mle_50) data for {model} - {exp_type} - {dataset}. Skipping.")
                
            axs[model_idx, 1].set_title(f"Intrinsic Dim (mle_50) - {model.split('/')[-1]}")
            axs[model_idx, 1].set_xlabel("Layer Index")
            axs[model_idx, 1].set_ylabel("Intrinsic Dimension")
            axs[model_idx, 1].legend()

            # Intrinsic dimension graph with twonn
            axs[model_idx, 2].set_prop_cycle(None)  # Reset color cycle for distinct lines
            for exp_type in experiment_types:
                intrinsic_dims_twonn = get_intrinsic_dimensions(model, dataset, exp_type, "twonn")
                if intrinsic_dims_twonn is None:
                    print(f"Intrinsic dimension (twonn) data not found for {model} - {exp_type} - {dataset}. Skipping.")
                    continue  # Skip plotting if there's an error
                try:
                    axs[model_idx, 2].plot(range(num_layers - 1), intrinsic_dims_twonn, label=exp_type)
                except:
                    print(f"There was an error with (twonn) data for {model} - {exp_type}. Skipping.")
                    
            axs[model_idx, 2].set_title(f"Intrinsic Dim (twonn) - {model.split('/')[-1]}")
            axs[model_idx, 2].set_xlabel("Layer Index")
            axs[model_idx, 2].set_ylabel("Intrinsic Dimension")
            axs[model_idx, 2].legend()

        # Save figure to temporary image file
        plt_path = f"results_{dataset}.png"
        plt.savefig(plt_path)
        pdf.image(plt_path, x=10, y=30, w=280)  # Adjust width to fit the landscape orientation
        os.remove(plt_path)
        plt.close(fig)

    # Output the PDF to file
    output_path = "experiment_results_all_models.pdf"
    pdf.output(output_path)
    print(f"PDF generated at: {output_path}")


# Define the input parameters
models = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Meta-Llama-3-8B",
    "mistralai/Mistral-7B-v0.3",
    "meta-llama/Meta-Llama-3-70B"
]  

# Example models, add or change as needed
datasets = ["sst2", "cola", "qnli", "qqp", "mnli", "ag_news", "commonsense_qa", "mmlu"]
experiment_types = ["icl-0", "icl-1", "icl-2", "icl-5", "icl-10", "finetune 1k", "finetune 10"]
accuracy_type = "logit-accuracy"

# Generate the PDF
generate_main_results_pdf(models, datasets, experiment_types, accuracy_type)