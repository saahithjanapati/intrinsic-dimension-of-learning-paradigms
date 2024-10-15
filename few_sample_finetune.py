import wandb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from utils import load_config, load_dataset
import sys
from tqdm.auto import tqdm
import os, glob, json, argparse
from types import SimpleNamespace
import traceback
from pathlib import Path



def get_train_args(config, output_dir, train_dataset_length):
    """
    return training arguments from the provided config
    """
    steps_per_epoch = max(1, train_dataset_length // config.batch_size)
    save_steps = steps_per_epoch * config.save_frequency_in_epochs  # Save every 5 epochs
    logging_steps = max(1, steps_per_epoch // 4)

    training_args = TrainingArguments(
        output_dir = output_dir,
        report_to="wandb",
        per_device_train_batch_size = config.batch_size,
        per_device_eval_batch_size = config.batch_size,
        bf16=True,
        learning_rate = float(config.lr),
        lr_scheduler_type = "linear",
        warmup_ratio = 0.1,
        num_train_epochs = config.num_train_epochs,
        evaluation_strategy = "epoch",
        logging_strategy = "steps",
        logging_steps = logging_steps,
        save_strategy = "steps",
        save_steps = save_steps  # Save every 5 epochs
    )

    # # set gradient_accumulation steps, gradient_checkpointing if they exist
    # if hasattr(config, 'effective_batch_size'):
    #     training_args.gradient_accumulation_steps = config.effective_batch_size // config.batch_size

    if hasattr(config, 'gradient_checkpointing'):
        training_args.gradient_checkpointing = config.gradient_checkpointing

    if hasattr(config, 'gradient_checkpointing_kwargs'):
        training_args.gradient_checkpointing_kwargs = config.gradient_checkpointing_kwargs

    return training_args



def generate_ft_output_path(config, dataset_name, training_set_id, num_samples):
    return Path(f"finetune-outputs/few-sample/{config.model_id}/{dataset_name}/{num_samples}-samples/{training_set_id}")


def main():
    # load config
    path_to_config = sys.argv[1]

    config_dict = load_config(path_to_config)
    config = SimpleNamespace(**config_dict)


    for training_set_id in range(config.num_training_sets):
        for dataset_name in config.datasets:

            model = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                device_map = "auto",
                torch_dtype = torch.bfloat16,
                cache_dir = "/scratch/jax4zk/cache/"
            )

            lora_config = SimpleNamespace(**config.lora_config)

            peft_config = LoraConfig(
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                bias=lora_config.bias,
                task_type=lora_config.task_type,
                target_modules= lora_config.target_modules,
            )

            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()


            try:
                # load up the train and eval datasets
                dataset = load_dataset(dataset_name)
                train_dataset = Dataset.from_list(dataset['train'])
                eval_dataset = Dataset.from_list(dataset['validation'])

            except Exception as e:
                print(f"Error loading datasets: {e}")
                sys.exit(1)


            # load up the training indices
            # config.num_training_samples - how many data items you want to do few-shot fine-tuning with
            path_to_ft_indices = Path(f"data_indices/ft_indices_{config.num_training_samples}_samples.json")

            with open(path_to_ft_indices, 'r') as f:
                indices_dict = json.load(f)
                
            training_indices = indices_dict[str(training_set_id)]

            wandb.init(project='few-sample-ft', job_type="train", config=config)

            train_dataset = train_dataset.select(training_indices)
            eval_dataset = eval_dataset.select(range(min(config.max_validation_dataset_length, len(eval_dataset))))

            output_path = generate_ft_output_path(config, dataset_name, training_set_id, config.num_training_samples)
            training_args = get_train_args(config, output_path, train_dataset_length = len(train_dataset))

            trainer = SFTTrainer(
                model,
                train_dataset = train_dataset,
                eval_dataset = eval_dataset,
                max_seq_length = config.max_seq_length,
                args = training_args,
                dataset_text_field="combined",
            )

            trainer.train()

            # save training stats
            log_history = trainer.state.log_history
            stats_path = Path(training_args.output_dir) / "training_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(log_history, f)

            # save the final PEFT model
            path = Path(training_args.output_dir) / "final"
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            trainer.save_model(path)

            del model
            torch.cuda.empty_cache()



if __name__ == "__main__":
    main()