# A Comparative Study of Learning Paradigms in Large Language Models via Intrinsic Dimension

Repository accompanying the paper [A Comparative Study of Learning Paradigms in Large Language Models via Intrinsic Dimension](https://aclanthology.org/2025.repl4nlp-1.5/) (RepL4NLP 2025). The code runs Supervised Fine‑Tuning (SFT) and In‑Context Learning (ICL), extracts hidden states, estimates Intrinsic Dimension (ID), and reproduces figures.

## Main Features of the Codebase
- Fine‑tune with LoRA and evaluate: Llama‑3‑8B, Llama‑2‑13B/7B, Mistral‑7B.
- ICL pipelines with configurable k‑shot prompting and optional de‑duped demos.
- Layerwise hidden‑state dumps, TwoNN and MLE ID estimators, normalized AUC.
- Figure scripts and concise result exports that match the paper’s plots.

## Environment
- Python 3.10+, PyTorch with CUDA, and GPUs (experiments were run on a NVIDIA A6000 GPU's on UVA's Rivanna HPC cluster).
- pip install: `torch transformers datasets peft wandb numpy scipy scikit-learn safetensors tqdm seaborn matplotlib fpdf`
- Optional: W&B for logging (set `WANDB_API_KEY`).


## Repository Map
- ICL
  - [run_icl_exp.py](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/blob/main/run_icl_exp.py): builds k‑shot prompts, logs accuracy, saves last‑token hidden states and logits to `results/icl/...`.
  - Configs: [`yaml/`](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/tree/main/yaml) (`icl_exp_*.yaml`: models, datasets, `num_shots`, `run_activation`, `run_accuracy`, `is_deduped`).
- SFT (1k samples)
  - [finetune.py](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/blob/main/finetune.py): trains LoRA adapters; outputs `finetune-outputs/full-ft/<model>/<dataset>/` with `final/` adapter and `training_stats.json`.
  - [run_ft_exp.py](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/blob/main/run_ft_exp.py): loads LoRA adapter (e.g., `adapter_name: final`), evaluates 0‑shot (or optional k‑shot) and writes `results/ft/...`.
  - Configs: [`yaml/finetune_full_1.yaml`](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/blob/main/yaml/finetune_full_1.yaml), [`yaml/ft_exp_1.yaml`](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/blob/main/yaml/ft_exp_1.yaml).
- SFT (detailed checkpoints)
  - [detailed_finetune.py](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/blob/main/detailed_finetune.py): like `finetune.py`, but used when sweeping checkpoints and LoRA ranks.
  - [detailed_ft_exp.py](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/blob/main/detailed_ft_exp.py): collects train/validation activations for each checkpoint into `results/detailed-ft/...`.
  - Configs: [`yaml/exp_detailed_ft.yaml`](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/blob/main/yaml/exp_detailed_ft.yaml) (checkpoint list), ID configs below.
- Few‑sample SFT (e.g., 5 or 10 train samples)
  - [few_sample_finetune.py](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/blob/main/few_sample_finetune.py): trains with indices from [`data_indices/`](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/tree/main/data_indices) `ft_indices_*.json`.
  - [run_few_ft_exp.py](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/blob/main/run_few_ft_exp.py): evaluates and saves to `results/few-sample-ft/...`.
  - Configs: [`yaml/finetune_few_sample.yaml`](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/blob/main/yaml/finetune_few_sample.yaml), [`yaml/few_sample_ft_exp.yaml`](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/blob/main/yaml/few_sample_ft_exp.yaml).
- ID estimation
  - [run_compute_id.py](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/blob/main/run_compute_id.py): reads results and computes ID per layer.
    - [`yaml/compute_id_icl.yaml`](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/blob/main/yaml/compute_id_icl.yaml), [`yaml/compute_id_ft.yaml`](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/blob/main/yaml/compute_id_ft.yaml), [`yaml/compute_id_few_ft.yaml`](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/blob/main/yaml/compute_id_few_ft.yaml), [`yaml/compute_id_detailed_ft.yaml`](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/blob/main/yaml/compute_id_detailed_ft.yaml) choose experiment type, models, datasets, k, and methods.
  - [calculate_id.py](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/blob/main/calculate_id.py): TwoNN (`twonn`) and MLE (`mle_k`) estimators.
- Results and figures
  - [generate_concise_results.py](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/blob/main/generate_concise_results.py): copies `results/` to `concise_results/` and writes `accuracy.json` (logit‑based) next to ID outputs.
  - [generate_results.py](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/blob/main/generate_results.py): CLI to build all plots (boxplots, heatmaps, two‑column summaries, SFT‑checkpoint panels).
  - [generate_results_pdf.py](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/blob/main/generate_results_pdf.py): optional PDF report with multi‑page figures.
- Utilities & data
  - [utils.py](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/blob/main/utils.py): config loading, dataset/json IO, batching, accuracy parsing, hidden‑state save helpers.
  - [model_util.py](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/blob/main/model_util.py): text generation and last‑token hidden‑state/softmax extraction.
  - [generate_data_indices.py](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/blob/main/generate_data_indices.py): regenerates ICL and few‑FT index maps.
  - Data: [`datasets/`](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/tree/main/datasets) `*.json`, [`data_indices/`](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/tree/main/data_indices) `*.json`.
  - Slurm templates: [`slurm_scripts/`](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/tree/main/slurm_scripts) `*.sbatch` mirror the local commands.

## Data Format
- Datasets live under [`datasets/`](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/tree/main/datasets) as JSON with keys `train` and `validation` (paper uses 1000/5000).
- Example item fields:
  - `input`: task prompt without the answer.
  - `output`: gold label token (e.g., `"A"`, `"yes"`, `"positive"`).
  - `combined`: prompt+answer text used for SFT and ICL demonstrations.
- ICL indices: [`data_indices/`](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/tree/main/data_indices) `icl_indices_<k>_shot.json` (default sampling) and `no_repeat_icl_indices_<k>_shot.json` (unique demonstrations across prompts).
- Few‑FT indices: [`data_indices/`](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/tree/main/data_indices) `ft_indices_<N>_samples.json` per training‑set id.

To regenerate indices:
```bash
python generate_data_indices.py
```

## Sample Workflow
- Fine‑tune (1k) then evaluate (0‑shot) and compute ID
  - Train LoRA: `python finetune.py yaml/finetune_full_1.yaml`
  - Evaluate: `python run_ft_exp.py yaml/ft_exp_1.yaml` (use `num_shots: [0]`)
  - ID: `python run_compute_id.py yaml/compute_id_ft.yaml`
- Detailed SFT checkpoints (train/val), then ID
  - Train: `python detailed_finetune.py yaml/finetune_full_1.yaml`
  - Collect activations: `python detailed_ft_exp.py yaml/exp_detailed_ft.yaml`
  - ID: `python run_compute_id.py yaml/compute_id_detailed_ft.yaml`
- ICL across k
  - Collect: `python run_icl_exp.py yaml/icl_exp_2.yaml` (set models, datasets, `num_shots` and optional `is_deduped: True`)
  - ID: `python run_compute_id.py yaml/compute_id_icl.yaml`
- Few‑sample SFT (optional)
  - Train: `python few_sample_finetune.py yaml/finetune_few_sample.yaml`
  - Evaluate + ID: `python run_few_ft_exp.py yaml/few_sample_ft_exp.yaml`; `python run_compute_id.py yaml/compute_id_few_ft.yaml`

Outputs (by default) are written under `results/`:
- Accuracy: `results/<paradigm>/accuracy_results/<model>/<dataset>/<k>-shot/acc-results.json`
- Logits + per‑sample correctness: `results/<paradigm>/logits/.../logit_data.json`
- Hidden states: `results/<paradigm>/activations/.../layer-*/layer_*_index-*.safetensors`
- ID: `results/id/<paradigm>/<model>/<dataset>/<k>-shot/{twonn|mle_<k>}.json`

To build paper figures (reads `concise_results/`):
```bash
python generate_concise_results.py                 # copies results → concise_results and writes accuracy.json
python generate_results.py auc_boxplot             # Figure 5
python generate_results.py inter_method_auc_heatmap# Figure 4
python generate_results.py two_column_summary --model meta-llama/Meta-Llama-3-8B --dataset mmlu  # Figure 1
python generate_results.py llama-3-8b-combined-ft-datasets                                      # SFT dynamics panels
```
All generated PDFs go to `results_and_figures/`.

## Accuracy Metrics
- Generation accuracy: compares first generated token vs label, saved as `acc-results.json` during `run_*_exp.py`.
- Logit accuracy: uses last‑token logits over label tokens and `normalize_answer`; written by [generate_concise_results.py](https://github.com/saahithjanapati/intrinsic-dimension-of-learning-paradigms/blob/main/generate_concise_results.py) as `accuracy.json` next to ID files. This matches the paper’s evaluation definition.


## Experiments (Paper Summary)
- Models: Llama‑3‑8B, Llama‑2‑13B/7B, Mistral‑7B.
- Datasets: AG News, CoLA, CommonsenseQA, MMLU, MNLI, QNLI, QQP, SST‑2.
- Findings echoed by the code paths above:
  - SFT: ID sometimes dips early then increases with training; later layers move the most.
  - ICL: ID rises for small k (≈5–10) then plateaus or decreases; peaks in ID align with peak accuracy in most settings.
  - Comparison: ICL (k≥5) induces higher IDs than SFT at 1k, while SFT attains higher accuracy.

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
Experiments used the Rivanna HPC Cluster at the University of Virginia.
