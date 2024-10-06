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


def generate_prompts(dataset_name, k, num_eval_items=5000):
    dataset = load_dataset(dataset_name)
    train_set = dataset['train']
    eval_set = dataset['validation']

    prompts, answers = [], []

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



def run_accuracy(model, tokenizer, model_name, dataset_name, k, num_eval_items = 5000, max_new_tokens=6):
    prompts, targets = generate_prompts(dataset_name, k=k, num_eval_items=num_eval_items)
    dataset = [{"input": prompts[i], "output": targets[i]} for i in range(len(prompts))]

    run = wandb.init(project="ft-accuracy", reinit=True)
    wandb.alert(title=f"ft-Acc-{dataset_name}", text=f"model_name_{model_name}-dataset_name_{dataset_name}")

    batch_size = get_batch_size(model_name, dataset_name, k)

    all_generations = []
    num_correct = 0
    total_num_samples = 0

    for batch_start_idx in tqdm(range(0, len(prompts), batch_size)):
        batch = dataset[batch_start_idx: batch_start_idx + batch_size]
        
        batch_prompts = [ex["input"] for ex in batch]
        batch_targets = [ex["output"] for ex in batch]
                      
        try:
            set_random_seed()
            batch_generations = model_util.generate_new_texts(model, tokenizer, batch_prompts, max_new_tokens=max_new_tokens)
                                
            for i in range(len(batch_generations)):
                target = batch_targets[i]
                index = batch_start_idx + i
                generation = batch_generations[i]

                parsed_str, score = parse_generation(generation, [target], first_word_score)

                num_correct += int(score)
                total_num_samples += 1

                all_generations.append({
                    "prompt": batch_prompts[i],
                    "target": target,
                    "generated_output": generation,
                    "score": score,
                    "parsed_str": parsed_str,
                    "index": index,
                })
                
        except Exception as e:
            print(f"ERROR on batch with start idx: {batch_start_idx}")
            print(e)
            traceback.print_exc()
            continue
    

    accuracy = num_correct / total_num_samples
    results_path = Path(f"results/ft/accuracy_results/{model_name}/{dataset_name}/{k}-shot/acc-results.json")
    generations_path = Path(f"results/ft/accuracy_results/{model_name}/{dataset_name}/{k}-shot/generations.json")
                                
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, "w") as f:
        
        json.dump({
            "accuracy": accuracy,
            "num_correct": num_correct,
            "total_num_samples": total_num_samples
        }, f)
    
    with open(generations_path, "w") as f:
        json.dump(all_generations, f)


def run_activations(model, tokenizer, model_name, dataset_name, k, num_eval_items = 5000):
    prompts, targets = generate_prompts(dataset_name, k=k, num_eval_items=num_eval_items)

    dataset = [{"input": prompts[i], "output": targets[i]} for i in range(len(prompts))]

    run = wandb.init(project="ft-activations", reinit=True)
    wandb.alert(title=f"ft-Act-{dataset_name}", text=f"model_name_{model_name}-dataset_name_{dataset_name}")

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
                    save_path = Path(f"results/ft/activations/{model_name}/{dataset_name}/{k}-shot/layer-{layer_idx}/layer_{layer_idx}_index-{true_index}.safetensors")
                    save_path.parent.mkdir(parents=True, exist_ok = True)
                    save_tensor_to_file(curr_hidden_state[layer_idx], save_path)
                                
            
        except Exception as e:
            print(f"ERROR on batch with start idx: {batch_start_idx}")
            print(e)
            traceback.print_exc()
            continue


        indices_path = Path(f"results/ft/logits/{model_name}/{dataset_name}/{k}-shot/logit_data.json")
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

    lora_adapter_name = config["adapter_name"]


    for model_name in models:
        for dataset_name in datasets:

            if lora_adapter_name == 'final':
                lora_adapter_path = Path(f"finetune-outputs/full-ft/{model_name}/{dataset_name}/final/")

            updated_model_name = model_name + "_" + lora_adapter_name

            model, tokenizer = load_model_and_tokenizer(model_name, output_hidden_states=True, lora_adapter_path = lora_adapter_path)
            print(f"Model {updated_model_name} loaded successfully.")

            for k in num_shots:
                if run_activations_flag:
                    print(f"Running activations for {updated_model_name} on {dataset_name} with {k}-shot.")
                    run_activations(model, tokenizer, updated_model_name, dataset_name, k)
                
                if run_accuracy_flag:
                    print(f"Running accuracy for {updated_model_name} on {dataset_name} with {k}-shot.")
                    run_accuracy(model, tokenizer, updated_model_name, dataset_name, k)
        
            del model
            del tokenizer
            torch.cuda.empty_cache()
                



if __name__ == "__main__":
    main()

