import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import time
import math
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm.auto import tqdm

class AkiaTrainer:
    """Main trainer for Akia HRM model"""
    
    def __init__(self, model, config: Dict, tokenizer):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Setup mixed precision
        self.use_amp = config.get('mixed_precision', True)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        print(f"Trainer initialized on device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
    
    def setup_optimizer_and_scheduler(self, train_dataloader):
        """Setup optimizer and learning rate scheduler"""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in ['bias', 'norm', 'embedding']):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer_grouped_parameters = [
            {'params': decay_params, 'weight_decay': self.config.get('weight_decay', 0.01)},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.get('learning_rate', 1e-4),
            betas=(self.config.get('beta1', 0.9), self.config.get('beta2', 0.95)),
            eps=self.config.get('epsilon', 1e-8)
        )
        
        # Setup scheduler
        total_steps = len(train_dataloader) * self.config.get('max_epochs', 25)
        warmup_steps = self.config.get('warmup_steps', 500)
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.config.get('min_learning_rate', 1e-6)
        )
        
        print(f"Optimizer and scheduler setup complete")
        print(f"Total training steps: {total_steps}")
    
    def prepare_data(self, dataset):
        """Prepare train and validation data"""
        # Split dataset
        train_size = int(self.config.get('train_split', 0.9) * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        from .dataset import SimpleCollator
        collator = SimpleCollator(pad_token_id=0)
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=True,
            collate_fn=collator,
            num_workers=self.config.get('dataloader_num_workers', 2),
            pin_memory=self.config.get('pin_memory', True)
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=False,
            collate_fn=collator,
            num_workers=self.config.get('dataloader_num_workers', 2),
            pin_memory=self.config.get('pin_memory', True)
        )
        
        print(f"Data prepared: {len(train_dataset)} train, {len(val_dataset)} val samples")
        return train_dataloader, val_dataloader
    
    def training_step(self, batch) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass with mixed precision
        if self.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(
                    input_ids=input_ids, 
                    labels=labels, 
                    reasoning_steps=self.config.get('reasoning_steps_train', 6)
                )
                loss = outputs['total_loss'] / self.config.get('gradient_accumulation_steps', 4)
        else:
            outputs = self.model(
                input_ids=input_ids, 
                labels=labels, 
                reasoning_steps=self.config.get('reasoning_steps_train', 6)
            )
            loss = outputs['total_loss'] / self.config.get('gradient_accumulation_steps', 4)
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Return metrics
        metrics = {
            'loss': outputs.get('loss', 0.0).item(),
            'total_loss': outputs.get('total_loss', 0.0).item(),
            'consistency_loss': outputs.get('consistency_loss', 0.0).item() if 'consistency_loss' in outputs else 0.0,
            'halt_regularization': outputs.get('halt_regularization', 0.0).item(),
            'reasoning_steps': outputs.get('reasoning_steps_taken', 0),
            'perplexity': math.exp(outputs.get('loss', 0.0).item()) if outputs.get('loss', 0.0).item() < 10 else float('inf')
        }
        
        return metrics
    
    def validation_step(self, batch) -> Dict[str, float]:
        """Single validation step"""
        self.model.eval()
        
        with torch.no_grad():
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=input_ids, 
                        labels=labels, 
                        reasoning_steps=self.config.get('reasoning_steps_eval', 8)
                    )
            else:
                outputs = self.model(
                    input_ids=input_ids, 
                    labels=labels, 
                    reasoning_steps=self.config.get('reasoning_steps_eval', 8)
                )
        
        metrics = {
            'val_loss': outputs.get('loss', 0.0).item(),
            'val_total_loss': outputs.get('total_loss', 0.0).item(),
            'val_reasoning_steps': outputs.get('reasoning_steps_taken', 0),
            'val_perplexity': math.exp(outputs.get('loss', 0.0).item()) if outputs.get('loss', 0.0).item() < 10 else float('inf')
        }
        
        return metrics
    
    def save_checkpoint(self, save_path: str, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, save_path)
        
        if is_best:
            best_path = str(Path(save_path).parent / 'best_model.pt')
            torch.save(checkpoint, best_path)
        
        print(f"Checkpoint saved: {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Checkpoint loaded: {checkpoint_path}")
    
    def train(self, dataset):
        """Main training loop"""
        print("Starting Akia HRM training...")
        
        # Prepare data
        train_dataloader, val_dataloader = self.prepare_data(dataset)
        
        # Setup optimizer and scheduler
        self.setup_optimizer_and_scheduler(train_dataloader)
        
        # Initialize wandb if enabled
        if self.config.get('use_wandb', True):
            wandb.init(
                project=self.config.get('wandb_project', 'akia-hrm'),
                name=self.config.get('experiment_name', 'akia-training'),
                config=self.config
            )
        
        # Training loop
        max_epochs = self.config.get('max_epochs', 25)
        gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 4)
        eval_steps = self.config.get('eval_steps', 250)
        save_steps = self.config.get('save_steps', 500)
        logging_steps = self.config.get('logging_steps', 50)
        
        for epoch in range(max_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            num_batches = 0
            
            # Training
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{max_epochs}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Training step
                metrics = self.training_step(batch)
                epoch_loss += metrics['total_loss']
                num_batches += 1
                
                # Gradient accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                
                # Logging
                if self.global_step % logging_steps == 0:
                    metrics['learning_rate'] = self.scheduler.get_last_lr()[0]
                    metrics['epoch'] = epoch
                    metrics['global_step'] = self.global_step
                    
                    if self.config.get('use_wandb', True):
                        wandb.log(metrics)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{metrics['total_loss']:.4f}",
                    'ppl': f"{metrics['perplexity']:.2f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
                    'steps': f"{metrics['reasoning_steps']}"
                })
                
                # Validation
                if self.global_step % eval_steps == 0:
                    val_metrics = self.validate(val_dataloader)
                    
                    if self.config.get('use_wandb', True):
                        wandb.log(val_metrics)
                    
                    # Save best model
                    if val_metrics['val_loss'] < self.best_loss:
                        self.best_loss = val_metrics['val_loss']
                        self.save_checkpoint(f"checkpoints/best_model_step_{self.global_step}.pt", is_best=True)
                
                # Save checkpoint
                if self.global_step % save_steps == 0:
                    self.save_checkpoint(f"checkpoints/checkpoint_step_{self.global_step}.pt")
            
            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Final validation for epoch
            val_metrics = self.validate(val_dataloader)
            val_metrics['epoch'] = epoch
            
            if self.config.get('use_wandb', True):
                wandb.log(val_metrics)
        
        # Save final model
        self.save_checkpoint("checkpoints/final_model.pt")
        print("Training completed!")
        
        if self.config.get('use_wandb', True):
            wandb.finish()
    
    def validate(self, val_dataloader) -> Dict[str, float]:
        """Run validation"""
        self.model.eval()
        
        total_val_loss = 0.0
        total_val_steps = 0
        all_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation", leave=False):
                metrics = self.validation_step(batch)
                all_metrics.append(metrics)
                total_val_loss += metrics['val_total_loss']
                total_val_steps += 1
        
        # Aggregate metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        
        print(f"Validation - Loss: {avg_metrics['val_loss']:.4f}, Perplexity: {avg_metrics['val_perplexity']:.2f}")
        
        return avg_metrics
