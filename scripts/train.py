#!/usr/bin/env python3
"""
Training script for Akia HRM
Usage: python scripts/train.py --data-path data/processed/akia_training_data.pkl
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from model.hrm_architecture import AkiaHRM, AkiaHRMConfig
from training.trainer import AkiaTrainer
from training.dataset import AkiaDataset
from utils.tokenizer import create_tokenizer_from_data, SimpleTokenizer
from utils.config import load_model_config, load_training_config

def main():
    parser = argparse.ArgumentParser(description="Train Akia HRM model")
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/akia_training_data.pkl",
        help="Path to training data"
    )
    
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Directory containing configuration files"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints"
    )
    
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to saved tokenizer (will create if not exists)"
    )
    
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    parser.add_argument(
        "--wandb-disabled",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (smaller dataset, frequent logging)"
    )
    
    args = parser.parse_args()
    
    # Setup W&B credentials automatically
    try:
        from setup_wandb import setup_wandb
        if not args.wandb_disabled:
            wandb_success = setup_wandb()
            if not wandb_success:
                print("⚠️  W&B setup failed, continuing without logging...")
                args.wandb_disabled = True
    except ImportError:
        print("⚠️  W&B setup script not found, you may need to authenticate manually")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 Starting Akia HRM Training")
    print("=" * 50)
    print(f"Data path: {args.data_path}")
    print(f"Config directory: {args.config_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Load configurations
    print("\n📋 Loading configurations...")
    model_config = load_model_config(args.config_dir)
    training_config = load_training_config(args.config_dir)
    
    # Override config for debug mode
    if args.debug:
        print("🐛 Debug mode enabled")
        training_config.update({
            'max_epochs': 2,
            'eval_steps': 10,
            'save_steps': 20,
            'logging_steps': 5,
            'batch_size': 2,
            'gradient_accumulation_steps': 2
        })
    
    # Disable wandb if requested
    if args.wandb_disabled:
        training_config['use_wandb'] = False
    
    # Create or load tokenizer
    print("\n🔤 Setting up tokenizer...")
    if args.tokenizer_path and Path(args.tokenizer_path).exists():
        print(f"Loading tokenizer from {args.tokenizer_path}")
        tokenizer = SimpleTokenizer.from_pretrained(args.tokenizer_path)
    else:
        print("Creating new tokenizer from training data...")
        tokenizer_save_path = args.tokenizer_path or "tokenizer_vocab.json"
        tokenizer = create_tokenizer_from_data(
            args.data_path, 
            vocab_size=model_config['vocab_size'],
            save_path=tokenizer_save_path
        )
    
    # Update model config with actual vocab size
    actual_vocab_size = tokenizer.get_vocab_size()
    model_config['vocab_size'] = actual_vocab_size
    print(f"📝 Updated vocab_size to actual tokenizer size: {actual_vocab_size}")
    
    # Initialize model
    print("\n🧠 Initializing Akia HRM model...")
    config = AkiaHRMConfig(**model_config)
    model = AkiaHRM(config)
    
    # Load dataset
    print(f"\n📊 Loading dataset from {args.data_path}...")
    dataset = AkiaDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=training_config.get('max_sequence_length', 1024)
    )
    
    # Debug mode: use smaller subset
    if args.debug:
        subset_size = min(100, len(dataset))
        dataset.data = dataset.data[:subset_size]
        print(f"Debug mode: using {subset_size} samples")
    
    # Initialize trainer
    print("\n🏋️ Initializing trainer...")
    trainer = AkiaTrainer(model, training_config, tokenizer)
    
    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"📂 Resuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)
    
    # Start training
    print("\n🎯 Starting training process...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Training samples: {len(dataset)}")
    print(f"Effective batch size: {training_config['batch_size'] * training_config['gradient_accumulation_steps']}")
    print(f"Max epochs: {training_config['max_epochs']}")
    
    try:
        trainer.train(dataset)
        print("\n🎉 Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        # Save emergency checkpoint
        emergency_path = os.path.join(args.output_dir, "emergency_checkpoint.pt")
        trainer.save_checkpoint(emergency_path)
        print(f"Emergency checkpoint saved: {emergency_path}")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        # Save emergency checkpoint
        emergency_path = os.path.join(args.output_dir, "error_checkpoint.pt")
        trainer.save_checkpoint(emergency_path)
        print(f"Error checkpoint saved: {emergency_path}")
        raise
    
    finally:
        print(f"\n📁 Checkpoints saved in: {args.output_dir}")
        print("Training session ended.")

if __name__ == "__main__":
    main()
