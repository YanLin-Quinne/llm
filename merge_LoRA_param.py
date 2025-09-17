import os
import random
import json

import fire
import torch

from pathlib import Path
from typing import Dict, List, Tuple

NUM_OF_LAYER = 32

def setting_seed(seed: int = 42) -> None:
    """
    Function to set the seed value.

    Args:
        seed (int, optional): Seed value. Defaults to 42.
    """
    torch.manual_seed(seed)
    random.seed(seed)


def get_params_keys(layer: int, proj: str) -> Tuple[str, str]:
    """
    Function to return the LoRA parameter keys added to each layer.

    Args:
        layer (int): Layer index for which to retrieve keys.
        proj (str): Type of projection layer for which to retrieve keys.

    Returns:
        Tuple[str, str]: Returns LoRA matrix A and B keys (lora_A.weight_key, lora_B.weight_key)
    
    Raises:
        KeyError: When the specified proj is not supported (e.g., lm_head)
    """
    if proj in ["down_proj", "gate_proj", "up_proj"]:
        layer_type = "mlp"
    elif proj in ["k_proj", "o_proj", "q_proj", "v_proj"]:
        layer_type = "self_attn"
    else:
        raise KeyError(f"{proj=} cannot be merged with the current code.")
    
    return (f"base_model.model.model.layers.{layer}.{layer_type}.{proj}.lora_A.weight",
            f"base_model.model.model.layers.{layer}.{layer_type}.{proj}.lora_B.weight")


def block_merge_parameters(lora_params: List[Dict[str, torch.Tensor]], merge_size: int,
                           params_keys: Tuple[str, str]) -> Tuple[Tuple[str, torch.Tensor], Tuple[str, torch.Tensor]]:
    """
    Function to merge parameters by selecting from the beginning of each base parameter.

    Args:
        lora_params (List[Dict[str, torch.Tensor]]): List containing base LoRA parameters.
        merge_size (int): Final rank of the LoRA after merging.
        params_keys (Tuple[str, str]): Two keys for base matrices A and B (lora_A.weight_key, lora_B.weight_key).

    Returns:
        Tuple[Tuple[str, torch.Tensor], Tuple[str, torch.Tensor]]: 
        Keys and parameters of merged LoRA matrices A and B.
        Returns in format ((lora_A.weight_key, param_A), (lora_B.weight_key, param_B)).
    
    Raises:
        KeyError: When merge_size is too large
    """
    layer_key_A, layer_key_B = params_keys

    base_lora_rank = lora_params[0][layer_key_A].shape[0]
    chosen_params_A, chosen_params_B = [], []
    
    if merge_size > lora_params[0][layer_key_A].shape[0] * len(lora_params):
        # Error handling when trying to merge beyond the base rank
        raise KeyError(f"Not enough blocks. merge_size must be <= base_lora_rank * len(lora_params).\
                        {merge_size=}, base_lora_rank * len(lora_params)={lora_params[0][layer_key_A].shape[0]*len(lora_params)}")

    block_size = merge_size // len(lora_params)
    additional_size = merge_size % len(lora_params)

    for param in lora_params:
        for idx in range(block_size):
            chosen_params_A.append(param[layer_key_A][idx])
            chosen_params_B.append(param[layer_key_B][:,idx])
        if additional_size > 0:
            # Add one more parameter from the beginning of param_list
            chosen_params_A.append(param[layer_key_A][idx + 1])
            chosen_params_B.append(param[layer_key_B][:,idx + 1])
            additional_size -= 1

    merged_params_A = torch.stack(chosen_params_A, dim=0)
    merged_params_B = torch.stack(chosen_params_B, dim=-1)

    merged_params_A = merged_params_A * ((base_lora_rank / merge_size) ** (1 / 2))
    merged_params_B = merged_params_B * ((base_lora_rank / merge_size) ** (1 / 2))

    return ((layer_key_A, merged_params_A), (layer_key_B, merged_params_B))


def random_merge_parameters(lora_params: List[Dict[str, torch.Tensor]], merge_size: int,
                           params_keys: Tuple[str, str]) -> Tuple[Tuple[str, torch.Tensor], Tuple[str, torch.Tensor]]:
    """
    Function to merge parameters by randomly selecting from base parameters.

    Args:
        lora_params (List[Dict[str, torch.Tensor]]): List containing base LoRA parameters.
        merge_size (int): Final rank of the LoRA after merging.
        params_keys (Tuple[str, str]): Two keys for base matrices A and B (lora_A.weight_key, lora_B.weight_key).

    Returns:
        Tuple[Tuple[str, torch.Tensor], Tuple[str, torch.Tensor]]: 
        Keys and parameters of merged LoRA matrices A and B.
        Returns in format ((lora_A.weight_key, param_A), (lora_B.weight_key, param_B)).
    """
    layer_key_A, layer_key_B = params_keys
    
    size_A, size_B = lora_params[0][layer_key_A].shape[0], lora_params[0][layer_key_B].shape[1]
    base_lora_rank = size_A
    chosen_params_A, chosen_params_B = [], []
    
    for _ in range(merge_size):
        # Randomly choose which LoRA parameter and which index to select from
        index_A, row = random.randint(0, len(lora_params) - 1), random.randint(0, size_A - 1)
        index_B, cal = index_A, row
        chosen_params_A.append(lora_params[index_A][layer_key_A][row])
        chosen_params_B.append(lora_params[index_B][layer_key_B][:,cal])
    
    merged_params_A = torch.stack(chosen_params_A, dim=0)
    merged_params_B = torch.stack(chosen_params_B, dim=-1)
    
    merged_params_A = merged_params_A * ((base_lora_rank / merge_size) ** (1 / 2))
    merged_params_B = merged_params_B * ((base_lora_rank / merge_size) ** (1 / 2))
    
    return ((layer_key_A, merged_params_A), (layer_key_B, merged_params_B))


def model_soups(lora_params: List[Dict[str, torch.Tensor]],
                params_keys: Tuple[str, str]) -> Tuple[Tuple[str, torch.Tensor], Tuple[str, torch.Tensor]]:
    """
    Function to create a model soup (simple average) from LoRA parameters.

    Args:
        lora_params (List[Dict[str, torch.Tensor]]): List containing base LoRA parameters.
        params_keys (Tuple[str, str]): Two keys for base matrices A and B (lora_A.weight_key, lora_B.weight_key).

    Returns:
        Tuple[Tuple[str, torch.Tensor], Tuple[str, torch.Tensor]]: 
        Keys and parameters of merged LoRA matrices A and B.
        Returns in format ((lora_A.weight_key, param_A), (lora_B.weight_key, param_B)).
    """
    layer_key_A, layer_key_B = params_keys    
    ave_param_A, ave_param_B = None, None

    for param in lora_params:
        if ave_param_A is None:
            ave_param_A, ave_param_B = param[layer_key_A], param[layer_key_B]
        else:
            ave_param_A += param[layer_key_A]
            ave_param_B += param[layer_key_B]
    
    merged_params_A = ave_param_A / len(lora_params)
    merged_params_B = ave_param_B / len(lora_params)
        
    return ((layer_key_A, merged_params_A), (layer_key_B, merged_params_B))


def build_lora_parameters(lora_params: List[Dict[str, torch.Tensor]], merge_mode: str, merge_size: int, 
                          merge_proj: List[str]) -> Dict[str, torch.Tensor]:
    """
    Function to merge LoRA parameters using the specified mode.

    Args:
        lora_params (List[Dict[str, torch.Tensor]]): List of base LoRA parameters.
        merge_mode (str): Merge mode. Must be one of 'block', 'random', or 'soups'.
        merge_size (int): Final rank of the LoRA after merging. Not used in 'soups' mode.
        merge_proj (List[str]): List of projection layers to merge.

    Returns:
        Dict[str, torch.Tensor]: Merged LoRA parameters.

    Raises:
        KeyError: When an invalid merge_mode is specified
    """
    merged_params: Dict[str, torch.Tensor] = {}

    for layer in range(NUM_OF_LAYER):
        for proj in merge_proj:
            params_keys = get_params_keys(layer, proj)                
            if merge_mode == 'block':
                merge_A, merge_B = block_merge_parameters(lora_params, merge_size, params_keys)
            elif merge_mode == 'random':
                merge_A, merge_B = random_merge_parameters(lora_params, merge_size, params_keys)
            elif merge_mode == 'soups':
                merge_A, merge_B = model_soups(lora_params, params_keys)

            merged_params[merge_A[0]] = merge_A[1]
            merged_params[merge_B[0]] = merge_B[1]
    return merged_params


def save_merged_lora(merged_params: Dict[str, torch.Tensor], file_path: str):
    """
    Function to save merged LoRA parameters in .bin format.
    
    Args:
        merged_params (Dict[str, torch.Tensor]): Parameters to save.
        file_path (str): String specifying where to save the parameters.
    """    
    os.makedirs(file_path)
    
    file_path += '/adapter_model.bin'
    
    torch.save(merged_params, file_path)


def load_lora_params(lora_path: str) -> Dict[str, torch.Tensor]:
    """
    Function to load base LoRA parameters.

    Args:
        lora_path (str): Path where the LoRA parameters to load are located.

    Returns:
        Dict[str, torch.Tensor]: Returns the loaded parameter keys and parameters.
    """
    lora_param = torch.load(lora_path, map_location="cpu")
    return lora_param

def update_adapter_config(model_path: str, output_dir: str, merge_size: int, merge_proj: List[str]) -> None:
    """
    Function to create adapter_config.json with updated rank, alpha, and merge_proj.
    Updates alpha based on the ratio of base parameter's rank and alpha to merge_size.

    Args:
        model_path (str): Path where the base parameter's adapter_config.json is saved.
        output_dir (str): Path where the newly created adapter_config.json will be saved.
        merge_size (int): Final rank of the LoRA after merging.
        merge_proj (List[str]): List of projection layers to merge.

    Raises:
        FileNotFoundError: Error when input adapter_config.json is not found.
    """
    try:
        model_path = Path(model_path)
        input_config_path = model_path.parent / 'adapter_config.json'
                
        output_config_path = output_dir + '/adapter_config.json'
        
        if not input_config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {input_config_path}")
        
        with open(input_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Match the ratio of rank and alpha
        scale = config['lora_alpha'] // config['r']
        
        if merge_size:
            config['lora_alpha'] = merge_size * scale
            config['r'] = merge_size
        config['target_modules'] = merge_proj
        
        with open(output_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
        print(f"Configuration file saved: {output_config_path}")
        
    except json.JSONDecodeError:
        print("Failed to parse JSON file.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def save_baseparameter_path(lora_path_list: List[str], output_dir: str) -> None:
    """
    Function to save the paths of base parameters used in merging to a text file.

    Args:
        lora_path_list (List[str]): Path where the LoRA parameters to load are located.
        output_dir (str): Path where base_parameter_path.txt will be saved.
    """
    output_path = output_dir + '/base_parameter_path.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lora_path_list))


def main(merge_mode: str = 'block', # options: block, random, soups
         merge_size: int | None = None, 
         merge_proj: List[str] = ['q_proj', 'k_proj', 'v_proj', 'up_proj', 'down_proj'],
         seed_list: List[int] = [100, 200, 300, 400], # only random
         output_dir: str = './output') -> None:
    """
    Function to generate a new LoRA model by merging multiple LoRA models.

    Args:
        merge_mode (str):
            Specifies the merging method. Choose one of the following:
            - "block": Select parameters sequentially from each LoRA parameter
            - "random": Select parameters randomly from each LoRA parameter
            - "soups": Merge LoRA parameters using Model Soups algorithm
            Defaults to "block"
        merge_size (int):
            Final rank of the LoRA after merging.
            Not used when merge_mode is "soups"
        merge_proj (List[str]): 
            List of projection layers to merge
            Defaults to ['q_proj', 'k_proj', 'v_proj', 'up_proj', 'down_proj']
            Specifying lm_head will result in an error
        seed_list (List[int]):
            Random seed values.
            Used when merge_mode is "random".
            Defaults to [100, 200, 300, 400].
        output_dir (str):
            Output directory for the merged LoRA model.
            Defaults to './output'

    Raises:
        KeyError: When an invalid merge_mode is specified
    """
    
    # List of paths to LoRA models to merge
    # Currently only supports .bin format files
    lora_path_list = ["PATH/adapter_model.bin",]    

    lora_param_list = []
    for path in lora_path_list:
        lora_param_list.append(load_lora_params(path))
    
    if merge_mode == "random":
        for i, seed in enumerate(seed_list):
            setting_seed(seed)
            merged_params = build_lora_parameters(lora_param_list, merge_mode, merge_size, merge_proj)
            if merged_params:
                save_merged_lora(merged_params, output_dir + f"merge{i}")
                print(f"Merged LoRA parameters have been saved to '{output_dir}merge{i}'.")
                update_adapter_config(lora_path_list[0], output_dir + f"merge{i}", merge_size, merge_proj)
                save_baseparameter_path(lora_path_list, output_dir + f"merge{i}")
            else:
                print("No merged parameters.")
    elif merge_mode == "block":
        merged_params = build_lora_parameters(lora_param_list, merge_mode, merge_size, merge_proj)
        if merged_params:
            save_merged_lora(merged_params, output_dir)
            print(f"Merged LoRA parameters have been saved to '{output_dir}'.")
            update_adapter_config(lora_path_list[0], output_dir, merge_size, merge_proj)
            save_baseparameter_path(lora_path_list, output_dir)
        else:
            print("No merged parameters.")
    elif merge_mode == "soups":
        merged_params = build_lora_parameters(lora_param_list, merge_mode, merge_size, merge_proj)
        if merged_params:
            save_merged_lora(merged_params, output_dir)
            print(f"Merged LoRA parameters have been saved to '{output_dir}'.")
            update_adapter_config(lora_path_list[0], output_dir, merge_size, merge_proj)
            save_baseparameter_path(lora_path_list, output_dir)
        else:
            print("No merged parameters.")            
    else:
        raise KeyError(f"merge_mode is not set correctly")

if __name__ == "__main__":
    fire.Fire(main)
