# LoRA Fine-tuning for Commonsense Reasoning Tasks

## Creating LoRA Base Parameters

This implementation enables reproduction of our paper's results through merging LoRA parameters.
The codebase primarily references the commonsense reasoning experiments from [DoRA: Weight-Decomposed Low-Rank Adaptation(ICML2024 (Oral))](https://github.com/NVlabs/DoRA).
The dataset documentation is adapted from the commonsense_reasoning .
For comprehensive details, please refer to the original DoRA commonsense_reasoning [README](https://github.com/NVlabs/DoRA/blob/main/commonsense_reasoning/README.md).

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
    apt-get update
    apt-get -y install git
    
    pip install -r requirements.txt

    ```

3. Login to huggingfece.
    ```bash
    huggingface-cli login
    ```

## Datasets

> Download the complete commonsense datasets from [here](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/dataset) and download the commonsense 170k　finetuning dataset from [here](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/commonsense_170k.json), then organize the data as follows

```bash
# Reference: DoRA/commonsense_reasoning/README.md

# Store the complete commonsense datasets
./dataset
# rest of the files
./experiment
./peft
# Finetuning commonsense dataset
./commonsense_170k.json
...
```

## LoRA Fine-tuning

The following scripts facilitate LoRA parameter generation:

### Standard Optimization Scripts
- llama2_7B_DoRA.sh
- llama3_8B_DoRA.sh

These scripts implement standard AdamW optimization. Execute as follows:

```bash
bash llama2_7B_DoRA.sh 32 64 ./output 0 42
bash llama3_8B_DoRA.sh 32 64 ./output 0 42
```

Parameters:
- Argument 1: LoRA rank
- Argument 2: LoRA alpha
- Argument 3: Output path
- Argument 4: GPU device ID
- Argument 5: Random seed

### Noisy Optimization Scripts
- llama2_7B_DoRA_NoisyAdamW.sh
- llama3_8B_DoRA_NoisyAdamW.sh

These scripts implement AdamW optimization with noise injection. Execute as follows:

```bash
bash llama2_7B_DoRA_NoisyAdamW.sh 32 64 ./output 0 42 0.001
bash llama3_8B_DoRA_NoisyAdamW.sh 32 64 ./output 0 42 0.001
```

Parameters:
- Argument 1: LoRA rank
- Argument 2: LoRA alpha
- Argument 3: Output path
- Argument 4: GPU device ID
- Argument 5: Random seed
- Argument 6: Noise temperature

## Evaluation

Evaluate LoRA and merged parameters using:

- llama2_7B_DoRA_eval.sh
- llama3_8B_DoRA_eval.sh

Execute as follows:

```bash
bash llama2_7B_DoRA_eval.sh ./output 0
bash llama3_8B_DoRA_eval.sh ./output 0
```

Parameters:
- Argument 1: Path to LoRA parameters
- Argument 2: GPU device ID

## Acknowledgement

We greatly appreciate the contributions of two remarkable repositories: [LLM-Adapter](https://github.com/AGI-Edgerunners/LLM-Adapters), [PEFT](https://github.com/huggingface/peft), [DoRA](https://github.com/NVlabs/DoRA). These projects have significantly benefited our work.
