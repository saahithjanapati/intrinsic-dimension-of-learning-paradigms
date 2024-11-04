from utils import *
import model_util
import sys
import wandb
from utils import *
import model_util
import sys
import wandb
from pathlib import Path
import json
from tqdm import tqdm
import traceback
from pathlib import Path


def generate_prompts(dataset_name, k, split_name):
    dataset = load_dataset(dataset_name)
    train_set = dataset['train']
    eval_set = dataset['validation']

    prompts, answers = [], []

    if split_name == 'train':
        eval_set = train_set

    num_eval_items = len(eval_set)

    # load icl indices
    icl_indices = load_icl_indices(k)
    for eval_idx in range(num_eval_items):
        prompt = ""
        indices = icl_indices[str(eval_idx)]

        for idx in indices:
            prompt += train_set[idx]['combined'] + "\n\n"

        query = eval_set[eval_idx]['input'] + "\n"
        answer = eval_set[eval_idx]['output']

        prompt += query
        prompts.append(prompt)
        answers.append(answer)

    return prompts, answers



def generate_label_set(dataset_name):
    """returns all unique labels in the dataset"""
    dataset = load_dataset(dataset_name)
    eval_set = dataset['validation']

    labels = set()
    for eval_idx in range(len(eval_set)):
        label = eval_set[eval_idx]['output']
        labels.add(label)
    
    return list(labels)



def run_activations(model, tokenizer, model_name, dataset_name, k, split_name):
    prompts, targets = generate_prompts(dataset_name, k=k, split_name=split_name)

    dataset = [{"input": prompts[i], "output": targets[i]} for i in range(len(prompts))]

    run = wandb.init(project="detailed-ft-activations", reinit=True)
    wandb.alert(title=f"detailed-Act-{dataset_name}", text=f"model_name_{model_name}-dataset_name_{dataset_name}")

    logit_dict = {}

    batch_size = get_batch_size(model_name, dataset_name, k)
    label_set = generate_label_set(dataset_name)
        
    for batch_start_idx in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[batch_start_idx: batch_start_idx + batch_size]
        batch_prompts = [ex["input"] for ex in batch]
                        
        try:
            set_random_seed()
            # Calculate perplexities and get hidden states for the last token
            # batch_perplexities, batch_hidden_states, top_logit_values, top_logit_indices, top_softmax_values = model_util.calculate_perplexity_and_hidden_states(model, tokenizer, batch_prompts, padding_side='left')
            batch_hidden_states, logits_for_last_token = model_util.get_hidden_states_and_logits_for_last_token(batch_prompts, model, tokenizer, verbose=False, padding_side='left')
            for x in range(len(batch_hidden_states)):
                    
                true_index = batch_start_idx + x
                curr_hidden_state = batch_hidden_states[x].cpu()

                ############################################################
                # Process logits for the last token
                logits = logits_for_last_token[x].cpu()
                probabilities = torch.softmax(logits, dim=-1)

                logit_values = {label: logits[tokenizer.encode(label, add_special_tokens=False)[0]].item() for label in label_set}
                probability_values = {label: probabilities[tokenizer.encode(label, add_special_tokens=False)[0]].item() for label in label_set}

                logit_dict[true_index] = {
                    "logits": logit_values,
                    "probabilities": probability_values
                }

                # Pick the label that yielded the highest logit
                predicted_label = max(logit_values, key=logit_values.get)
                correct_label = targets[true_index]
                is_correct = normalize_answer(predicted_label) == normalize_answer(correct_label)
                # Save the correctness information
                logit_dict[true_index]["is_correct"] = is_correct
                ############################################################
                                
                for layer_idx in range(curr_hidden_state.shape[0]):
                    save_path = Path(f"results/detailed-ft/activations/{model_name}/{dataset_name}/{split_name}/{k}-shot/layer-{layer_idx}/layer_{layer_idx}_index-{true_index}.safetensors")
                    save_path.parent.mkdir(parents=True, exist_ok = True)
                    save_tensor_to_file(curr_hidden_state[layer_idx], save_path)
                                
            
        except Exception as e:
            print(f"ERROR on batch with start idx: {batch_start_idx}")
            print(e)
            traceback.print_exc()
            continue


        indices_path = Path(f"results/detailed-ft/logits/{model_name}/{dataset_name}/{split_name}/{k}-shot/logit_data.json")
        indices_path.parent.mkdir(parents=True, exist_ok=True)
            
        with open(indices_path, 'w') as f:
            json.dump(logit_dict, f, indent=4)


def main():
    path_to_yaml = sys.argv[1]

    config = load_config(path_to_yaml)

    models = config["models"]
    datasets = config["datasets"]
    num_shots = config["num_shots"]

    run_accuracy_flag = config["run_accuracy"]
    run_activations_flag = config["run_activation"]

    lora_checkpoints = config["lora_checkpoints"]
    splits = ["train", "validation"]

    lora_r = config.get("lora_r", 64)

    for model_name in models:
        for dataset_name in datasets:
            for checkpoint_num in lora_checkpoints:


                if lora_r == 64:
                    lora_adapter_path = Path(f"finetune-outputs/detailed-ft/{model_name}/{dataset_name}/checkpoint-{checkpoint_num}/")
                    updated_model_name = model_name + "_" + f"checkpoint_{checkpoint_num}"
                
                else:
                    print(f"Running for lora_r = {lora_r}")
                    lora_adapter_path = Path(f"finetune-outputs/detailed-ft/{model_name}-lora_r_{lora_r}/{dataset_name}/checkpoint-{checkpoint_num}/")
                    updated_model_name = model_name + f"-lora_r_{lora_r}" + "_" + f"checkpoint_{checkpoint_num}"


                if checkpoint_num == 0:
                    model, tokenizer = load_model_and_tokenizer(model_name, output_hidden_states=True)

                else:
                    model, tokenizer = load_model_and_tokenizer(model_name, output_hidden_states=True, lora_adapter_path=lora_adapter_path)
                    print(f"Model {updated_model_name} loaded successfully.")

                for split_name in splits:
                    for k in num_shots:
                        if run_activations_flag:
                            print(f"Running activations for {updated_model_name} on {dataset_name}, split-{split_name} with {k}-shot.")
                            run_activations(model, tokenizer, updated_model_name, dataset_name, k, split_name)

                del model
                del tokenizer
                torch.cuda.empty_cache()
                



if __name__ == "__main__":
    main()

