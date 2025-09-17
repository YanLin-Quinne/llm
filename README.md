# Propagation of Chaos for Mean-Field Langevin Dynamics and its Application to Model Ensembling

## Setup

1. Create a PyTorch-enabled environment

    ```bash
    docker run \
        -v /path/to/project/:/workspace/ \
        --name <container_name> \
        --gpus all \
        -ti \
        --ipc=host \
        pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
    ```

2. Install required packages

    ```bash
    pip install fire
    ```

## Usage

Use `merge_LoRA_param.py` to perform parameter merging of LoRA models.

1. Configure LoRA base parameter paths

    Set the list of paths containing base parameters and adapter_config.json in the `lora_path_list` within `__main__`.

    Note: Base parameters must have the `.bin` extension for successful merging.

2. Execute the Python script

    The merged parameters and updated adapter_config.json will be generated in the `--output_dir`:

    ```bash
    python merge_LoRA_param.py \
        --merge_mode "block" \
        --merge_size 64 \
        --merge_proj "['q_proj', 'k_proj', 'v_proj', 'up_proj', 'down_proj']" \
        --output_dir "./output"
    ```

## Merging Modes

`merge_LoRA_param.py` implements three distinct merging strategies:

### Block Merging

- Sequentially merges columns of matrix A and rows of matrix B from each base parameter
- The merge size must satisfy the inequality:

    (Number of base parameters) * (Base parameter rank) ≤ (merge size)

- When (Post-merge size) % (Number of base parameters) ≠ 0, parameters are supplemented sequentially from the lowest index in `lora_path_list`

### Random Merging

- Stochastically selects columns of matrix A and rows of matrix B from base parameters with replacement
- Random selection is controlled by the `--seed_list` argument
- Parameters are generated using the specified seed values

### Model Soups

- Implements the methodology from Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time [[Wortsman+ ICML 2022](https://arxiv.org/abs/2203.05482)]
- Creates an ensemble through simple averaging of LoRA base parameters

## Command-line Arguments

### merge_mode (str)

- "block"(default): Sequential parameter selection from each LoRA
- "random": Stochastic parameter selection from each LoRA
- "soups": Ensemble using Model Soups algorithm

### merge_size (int)

- Final rank of the merged LoRA
- Ignored when merge_mode is "soups"

### merge_proj (List[str])

- Projection layers to merge
- Defaults to ['q_proj', 'k_proj', 'v_proj', 'up_proj', 'down_proj']

    Note: Including lm_head will raise an error

### seed_list (List[int])

- Random seeds for stochastic selection
- Used only when merge_mode is "random"
- Defaults to [100, 200, 300, 400]

### output_dir (str)

- Directory for saving merged LoRA model
- Defaults to './output'

## Reproducibility

Results can be reproduced by creating base parameters through commonsense_reasoning and applying block merging.

## Additional Notes

### adapter_config.json Specifications

This file is essential for evaluation purposes and is automatically generated alongside merged parameters.

The system loads adapter_config.json from the first path in `lora_path_list` and updates three key parameters:

- r: Updated with post-merge parameter size
- lora_alpha: Adjusted based on the ratio of base parameter r and lora_alpha
- target_modules: Updated to reflect merged layers
