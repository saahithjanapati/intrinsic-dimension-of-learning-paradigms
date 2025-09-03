# Intrinsic Dimension of Learning Paradigms

Repository accompanying the paper [*A Comparative Study of Learning Paradigms in Large Language Models via Intrinsic Dimension*](https://aclanthology.org/2025.repl4nlp-1.5/) (RepL4NLP 2025). The project compares how **supervised fine‑tuning (SFT)** and **in‑context learning (ICL)** shape the representation space of large language models (LLMs) through intrinsic dimension (ID).

## Highlights
- LoRA‑based fine‑tuning scripts for open LLMs (Llama‑3‑8B, Llama‑2‑13B, Llama‑2‑7B, Mistral‑7B).
- Utilities for building few‑shot prompts and running ICL with varying numbers of demonstrations.
- Tools to collect hidden states, estimate ID using the TwoNN estimator (using a GPU-optimized implementation), and compute the normalized area under the ID curve.
- Scripts that reproduce the accuracy and ID curves reported in the paper.

## Getting started
```bash
git clone https://github.com/your-user/intrinsic-dimension-of-learning-paradigms.git
cd intrinsic-dimension-of-learning-paradigms
python -m venv .venv && source .venv/bin/activate
pip install torch transformers datasets peft wandb numpy scipy scikit-learn tqdm
```
Login to [Weights & Biases](https://wandb.ai) if you wish to log experiments. Dataset splits used in the paper (1000 train / 5000 validation) are provided under `datasets/` and `data_indices/`.

## Running experiments
### Supervised fine‑tuning
Fine‑tune an LLM with LoRA:
```bash
python run_ft_exp.py yaml/ft_exp_1.yaml
```
The YAML configuration specifies the model, datasets, LoRA settings, and training schedule. Checkpoints and stats are stored under `finetune-outputs/`.

### In‑context learning
Evaluate ICL with different numbers of demonstrations:
```bash
python run_icl_exp.py yaml/icl_exp_1.yaml
```
Prompts are built using `k` examples sampled from the training split. Accuracy and hidden states are written to `results/icl/`.

### Computing intrinsic dimension
After collecting hidden states, estimate ID for each layer:
```bash
python run_compute_id.py yaml/compute_id_ft.yaml
```
`calculate_id.py` applies the TwoNN estimator and summarizes the results with the normalized area under the ID curve.

### Analysis and figures
Use `id_analysis.py`, `generate_results.py`, and `generate_concise_results.py` to aggregate accuracy/ID measurements and to recreate the figures shown in the paper (accuracy vs. shots, ID evolution during fine‑tuning, AUC heatmaps, etc.).

## Experiments from the paper
We evaluate four models (Llama‑3‑8B, Llama‑2‑13B, Llama‑2‑7B, Mistral‑7B) on eight NLP benchmarks: AG News, CoLA, CommonsenseQA, MMLU, MNLI, QNLI, QQP, and SST‑2.

- **SFT:** Models are fine‑tuned with LoRA on 1000 examples and ID is tracked across epochs. ID tends to rise in later training stages.
- **ICL:** Prompts with $k \in \{0,1,2,5,10,12,14,16,18,20\}$ demonstrations show a non‑linear relation between $k$ and ID—ID increases with a few demonstrations and then plateaus or decreases.
- **Comparison:** ICL with ≥5 demonstrations consistently induces higher‑dimensional representations than SFT despite lower accuracy, suggesting the two paradigms occupy different manifolds.

## Citation
```bibtex
@inproceedings{janapati2025intrinsic,
  title={A Comparative Study of Learning Paradigms in Large Language Models via Intrinsic Dimension},
  author={Saahith Janapati and Yangfeng Ji},
  booktitle={Proceedings of the 10th Workshop on Representation Learning for NLP (RepL4NLP-2025)},
  year={2025}
}
```

## Acknowledgements
This work was conducted with the Rivanna HPC Cluster at the University of Virginia.