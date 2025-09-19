import torch
from torch.utils.data import Dataset
import pickle
import json
from typing import List, Dict, Any
from pathlib import Path

class AkiaDataset(Dataset):
    """Dataset class for Akia HRM training data"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        data_path = Path(data_path)
        if data_path.suffix == '.pkl':
            with open(data_path, 'rb') as f:
                self.data = pickle.load(f)
        elif data_path.suffix == '.json':
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        print(f"Loaded {len(self.data)} training samples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Get input and target sequences
        input_sequence = sample.get('input_sequence', '')
        target_sequence = sample.get('target_sequence', '')
        
        # Combine for autoregressive training
        if input_sequence and target_sequence:
            full_text = input_sequence + " " + target_sequence
        elif target_sequence:
            full_text = target_sequence
        else:
            full_text = input_sequence
        
        # Tokenize
        tokens = self.tokenizer.encode(full_text)
        
        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        # Create input_ids and labels for causal language modeling
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'domain': sample.get('domain', 'unknown'),
            'difficulty': sample.get('difficulty', 'medium'),
            'sample_id': sample.get('id', f'sample_{idx}')
        }

class SimpleCollator:
    """Simple data collator for batching"""
    
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # Get maximum length in batch
        max_len = max(len(item['input_ids']) for item in batch)
        
        # Pad sequences
        input_ids = []
        labels = []
        attention_mask = []
        
        for item in batch:
            input_seq = item['input_ids']
            label_seq = item['labels']
            
            # Pad input_ids and labels
            pad_length = max_len - len(input_seq)
            if pad_length > 0:
                input_seq = torch.cat([input_seq, torch.full((pad_length,), self.pad_token_id)])
                label_seq = torch.cat([label_seq, torch.full((pad_length,), -100)])  # -100 is ignored in loss
            
            # Create attention mask (1 for real tokens, 0 for padding)
            attn_mask = torch.ones_like(input_seq)
            if pad_length > 0:
                attn_mask[-pad_length:] = 0
            
            input_ids.append(input_seq)
            labels.append(label_seq)
            attention_mask.append(attn_mask)
        
        return {
            'input_ids': torch.stack(input_ids),
            'labels': torch.stack(labels),
            'attention_mask': torch.stack(attention_mask)
        }
