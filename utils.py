import logging
from pathlib import Path
import json
import random
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
from typing import List, Dict, Any, Tuple, Optional
from peft import AutoPeftModelForCausalLM
import re
import string
from safetensors.torch import save_file



logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    try:
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.debug(f"Configuration loaded successfully: {config}")
        return config
    except FileNotFoundError as e:
        logger.error(f"Config file {config_path} not found: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_path}: {e}")
        raise


def write_to_json(obj: Any, path: Path, indent: int = 4) -> None:
    """Writes a Python object to a JSON file."""
    try:
        logger.info(f"Writing object to {path}")
        with path.open("w") as file:
            json.dump(obj, file, indent=indent)
        logger.debug(f"Object successfully written to {path}")
    except IOError as e:
        logger.error(f"Failed to write JSON to {path}: {e}")
        raise


def get_num_layers(model_name: str) -> Optional[int]:
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
        logger.warning(f"Model name {model_name} not found in the layers dictionary")
    else:
        logger.info(f"Model {model_name} has {layers} layers")
    return layers


def load_model_and_tokenizer(
    model_name: str, 
    output_hidden_states: bool = True, 
    load_in_8bit: bool = False, 
    lora_adapter_path: Optional[str] = None, 
    output_attentions: bool = False
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """Loads a model and its tokenizer, optionally loading a LoRA adapter."""
    try:
        if lora_adapter_path:
            logger.info(f"Loading model from LoRA adapter at {lora_adapter_path}")
            model = AutoPeftModelForCausalLM.from_pretrained(
                lora_adapter_path, 
                device_map="auto", 
                cache_dir="/scratch/jax4zk/cache/", 
                load_in_8bit=load_in_8bit, 
                output_hidden_states=output_hidden_states, 
                output_attentions=output_attentions
            )
        else:
            logger.info(f"Loading model {model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                cache_dir="/scratch/jax4zk/cache/", 
                device_map="auto", 
                load_in_8bit=load_in_8bit, 
                output_hidden_states=output_hidden_states, 
                output_attentions=output_attentions
            )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/scratch/jax4zk/cache/", padding_side="left")
        logger.debug(f"Tokenizer padding side: {tokenizer.padding_side}")
        logger.info(f"Model and tokenizer for {model_name} loaded successfully")

        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model and tokenizer for {model_name}: {e}")
        raise


def set_random_seed(seed: int = 42) -> None:
    """Sets the random seed for reproducibility."""
    logger.info(f"Setting random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.debug("Random seed set successfully")


def load_dataset(dataset_name: str, max_length: Optional[int] = None) -> List[Dict[str, Any]]:
    """Loads a dataset from a JSON file."""
    dataset_path = Path('datasets/') / (dataset_name + ".json")
    try:
        logger.info(f"Loading dataset from {dataset_path}")
        with dataset_path.open('r') as file:
            dataset = json.load(file)
        
        if max_length is not None:
            dataset = dataset[:max_length]
            logger.info(f"Dataset truncated to {max_length} entries")

        logger.debug(f"Dataset loaded successfully from {dataset_path}")
        return dataset
    except FileNotFoundError as e:
        logger.error(f"Dataset file {dataset_path} not found: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {dataset_path}: {e}")
        raise


def save_tensor_to_file(tensor, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({f"hidden_state": tensor}, path)

############################################################################################################
# Method to evaluate text generations
import re
import string

def normalize_answer(s: str) -> str:
    """Lowercase text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    normalized = white_space_fix(remove_articles(remove_punc(lower(s))))
    logger.debug(f"Normalized answer: {normalized}")
    return normalized


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Only correct if the prediction matches the entire answer."""
    result = normalize_answer(prediction) == normalize_answer(ground_truth)
    logger.debug(f"Exact match result: {result}")
    return result


def first_word_score(prediction: str, ground_truth: str) -> bool:
    """Only correct if the predicted first word matches the answer's first word."""
    prediction_words = normalize_answer(prediction).split()
    ground_truth_words = normalize_answer(ground_truth).split()

    if prediction_words and ground_truth_words:
        result = prediction_words[0] == ground_truth_words[0]
    else:
        result = len(prediction_words) == len(ground_truth_words)

    logger.debug(f"First word score result: {result}")
    return result


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str]) -> bool:
    """Pick maximum score across possible answers."""
    scores_for_ground_truths = [metric_fn(prediction, ground_truth) for ground_truth in ground_truths]
    max_score = max(scores_for_ground_truths)
    logger.debug(f"Max score across ground truths: {max_score}")
    return max_score


def parse_generation(output_str: str, target: List[str], metric_fn) -> Tuple[str, bool]:
    """Parse a generated string for the target, and score using the specified metric."""
    ans_regex = re.compile("([\w. ]+)[\nQ]*")
    parsed_str = ans_regex.findall(output_str)

    if parsed_str:
        parsed_str = parsed_str[0]
        score = metric_max_over_ground_truths(metric_fn, parsed_str, target)
        logger.debug(f"Parsed string: {parsed_str}, Score: {score}")
    else:
        parsed_str = ""
        score = 0.0
        logger.debug(f"No valid string parsed from output. Default score: {score}")

    return parsed_str, score