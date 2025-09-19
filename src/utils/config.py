import yaml
from pathlib import Path
from typing import Dict, Any, Optional

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"Configuration loaded from {config_path}")
    return config

def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Configuration saved to {save_path}")

def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries"""
    merged = {}
    
    for config in configs:
        for key, value in config.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = merge_configs(merged[key], value)
            else:
                merged[key] = value
    
    return merged

def load_model_config(config_dir: str = "config") -> Dict[str, Any]:
    """Load model configuration with defaults"""
    config_dir = Path(config_dir)
    
    # Default model configuration - only AkiaHRMConfig parameters
    default_config = {
        'vocab_size': 32000,
        'd_model': 512,
        'n_layers_high': 4,
        'n_layers_low': 8,
        'n_heads': 8,
        'd_ff': 1536,
        'max_sequence_length': 4096,
        'dropout': 0.1,
        'layer_norm_epsilon': 1e-5,
        'reasoning_steps': 8,
        'halt_threshold': 0.85,
        'use_flash_attention': True,
        'high_level_timescale': 2,
        'cross_hierarchy_dim': 192,
        'reasoning_head_dim': 96
    }
    
    # Valid AkiaHRMConfig parameters
    valid_params = set(default_config.keys())
    
    # Try to load from file
    model_config_path = config_dir / "model_config.yaml"
    if model_config_path.exists():
        file_config = load_config(model_config_path)
        if 'model' in file_config:
            file_config = file_config['model']
        
        # Filter to only valid parameters
        filtered_config = {k: v for k, v in file_config.items() if k in valid_params}
        merged_config = merge_configs(default_config, filtered_config)
        
        # Ensure critical float values are actually floats
        if 'layer_norm_epsilon' in merged_config:
            merged_config['layer_norm_epsilon'] = float(merged_config['layer_norm_epsilon'])
        if 'dropout' in merged_config:
            merged_config['dropout'] = float(merged_config['dropout'])
        if 'halt_threshold' in merged_config:
            merged_config['halt_threshold'] = float(merged_config['halt_threshold'])
            
        return merged_config
    else:
        print("Model config file not found, using defaults")
        return default_config

def load_training_config(config_dir: str = "config") -> Dict[str, Any]:
    """Load training configuration with defaults"""
    config_dir = Path(config_dir)
    
    # Default training configuration
    default_config = {
        'experiment_name': 'akia-hrm-training',
        'batch_size': 8,
        'gradient_accumulation_steps': 4,
        'learning_rate': 1e-4,
        'min_learning_rate': 1e-6,
        'weight_decay': 0.01,
        'beta1': 0.9,
        'beta2': 0.95,
        'epsilon': 1e-8,
        'max_epochs': 25,
        'warmup_steps': 500,
        'eval_steps': 250,
        'save_steps': 500,
        'logging_steps': 50,
        'mixed_precision': True,
        'train_split': 0.9,
        'val_split': 0.1,
        'reasoning_steps_train': 6,
        'reasoning_steps_eval': 8,
        'use_wandb': True,
        'wandb_project': 'akia-hrm',
        'dataloader_num_workers': 2,
        'pin_memory': True
    }
    
    # Try to load from file
    training_config_path = config_dir / "training_config.yaml"
    if training_config_path.exists():
        file_config = load_config(training_config_path)
        if 'training' in file_config:
            file_config = file_config['training']
        return merge_configs(default_config, file_config)
    else:
        print("Training config file not found, using defaults")
        return default_config

def create_default_configs(config_dir: str = "config"):
    """Create default configuration files"""
    config_dir = Path(config_dir)
    config_dir.mkdir(exist_ok=True)
    
    # Model configuration
    model_config = {
        'model': {
            'name': 'akia-hrm-27m',
            'version': '1.4',
            'vocab_size': 32000,
            'd_model': 512,
            'n_layers_high': 4,
            'n_layers_low': 8,
            'n_heads': 8,
            'd_ff': 1536,
            'max_sequence_length': 4096,
            'dropout': 0.1,
            'reasoning_steps': 8,
            'halt_threshold': 0.85,
            'high_level_timescale': 2,
            'cross_hierarchy_dim': 192,
            'reasoning_head_dim': 96
        }
    }
    
    # Training configuration
    training_config = {
        'training': {
            'experiment_name': 'akia-hrm-training',
            'data_path': 'data/processed/akia_training_data.pkl',
            'batch_size': 8,
            'gradient_accumulation_steps': 4,
            'learning_rate': 1e-4,
            'min_learning_rate': 1e-6,
            'weight_decay': 0.01,
            'max_epochs': 25,
            'eval_steps': 250,
            'save_steps': 500,
            'mixed_precision': True,
            'use_wandb': True,
            'wandb_project': 'akia-hrm'
        }
    }
    
    # Save configurations
    save_config(model_config, config_dir / "model_config.yaml")
    save_config(training_config, config_dir / "training_config.yaml")
    
    print(f"Default configurations created in {config_dir}")
