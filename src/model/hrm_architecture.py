import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, Embedding, Dropout
import math
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

@dataclass
class AkiaHRMConfig:
    """Configuration for Akia HRM model"""
    vocab_size: int = 32000
    d_model: int = 512
    n_layers_high: int = 4  # High-level planning layers
    n_layers_low: int = 8   # Low-level execution layers
    n_heads: int = 8
    d_ff: int = 1536
    max_sequence_length: int = 4096
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    reasoning_steps: int = 8
    halt_threshold: float = 0.85
    use_flash_attention: bool = True
    
    # Hierarchical specific parameters
    high_level_timescale: int = 2
    cross_hierarchy_dim: int = 192
    reasoning_head_dim: int = 96

class RotaryPositionalEncoding(nn.Module):
    """Rotary Positional Encoding"""
    
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        self.d_model = d_model
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary positional embedding"""
    q_rot = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).flatten(-2)
    k_rot = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).flatten(-2)
    
    q_embed = q * cos + q_rot * sin
    k_embed = k * cos + k_rot * sin
    return q_embed, k_embed

class HierarchicalAttention(nn.Module):
    """Multi-head attention with hierarchical communication"""
    
    def __init__(self, config: AkiaHRMConfig, is_high_level: bool = False):
        super().__init__()
        self.config = config
        self.is_high_level = is_high_level
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        
        self.q_proj = Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = Linear(config.d_model, config.d_model, bias=False)
        
        if not is_high_level:
            self.cross_hierarchy_proj = Linear(config.cross_hierarchy_dim, config.d_model, bias=False)
            self.hierarchy_gate = Linear(config.d_model * 2, config.d_model)
        
        self.dropout = Dropout(config.dropout)
        self.rotary_emb = RotaryPositionalEncoding(self.head_dim)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        hierarchy_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary_emb(seq_len, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            # Use -65504.0 instead of -1e9 for float16 compatibility
            mask_value = -65504.0 if scores.dtype == torch.float16 else -1e9
            scores = scores.masked_fill(mask == 0, mask_value)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        output = self.out_proj(attn_output)
        
        if not self.is_high_level and hierarchy_context is not None:
            hierarchy_proj = self.cross_hierarchy_proj(hierarchy_context)
            combined = torch.cat([output, hierarchy_proj], dim=-1)
            gate = torch.sigmoid(self.hierarchy_gate(combined))
            output = gate * output + (1 - gate) * hierarchy_proj
        
        return output

class HierarchicalTransformerLayer(nn.Module):
    """Single hierarchical transformer layer"""
    
    def __init__(self, config: AkiaHRMConfig, is_high_level: bool = False):
        super().__init__()
        self.config = config
        self.is_high_level = is_high_level
        
        self.attention = HierarchicalAttention(config, is_high_level)
        self.norm1 = LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.norm2 = LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        
        self.ffn = nn.Sequential(
            Linear(config.d_model, config.d_ff),
            nn.GELU(),
            Dropout(config.dropout),
            Linear(config.d_ff, config.d_model),
            Dropout(config.dropout)
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        hierarchy_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        attn_output = self.attention(
            self.norm1(x), 
            mask=mask, 
            hierarchy_context=hierarchy_context
        )
        x = x + attn_output
        
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output
        
        return x

class ReasoningHaltModule(nn.Module):
    """Module to determine when to halt reasoning"""
    
    def __init__(self, config: AkiaHRMConfig):
        super().__init__()
        self.config = config
        self.halt_predictor = nn.Sequential(
            Linear(config.d_model, config.reasoning_head_dim),
            nn.GELU(),
            Linear(config.reasoning_head_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        halt_scores = self.halt_predictor(x[:, -1:, :])
        return halt_scores.squeeze(-1)

class AkiaHRM(nn.Module):
    """Akia Hierarchical Reasoning Model - 27M Parameters"""
    
    def __init__(self, config: AkiaHRMConfig):
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.token_embedding = Embedding(config.vocab_size, config.d_model)
        self.embedding_dropout = Dropout(config.dropout)
        
        # High-level planning module
        self.high_level_layers = nn.ModuleList([
            HierarchicalTransformerLayer(config, is_high_level=True)
            for _ in range(config.n_layers_high)
        ])
        
        # Low-level execution module
        self.low_level_layers = nn.ModuleList([
            HierarchicalTransformerLayer(config, is_high_level=False)
            for _ in range(config.n_layers_low)
        ])
        
        # Cross-hierarchy projection
        self.hierarchy_proj = Linear(config.d_model, config.cross_hierarchy_dim)
        
        # Reasoning and halt mechanisms
        self.reasoning_halt = ReasoningHaltModule(config)
        
        # Output head
        self.output_norm = LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.lm_head = Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Verify parameter count
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Akia HRM initialized with {total_params:,} parameters")
        
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        reasoning_steps: Optional[int] = None,
        return_reasoning_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        x = self.token_embedding(input_ids)
        x = self.embedding_dropout(x)
        
        # Create causal mask
        mask = self.create_causal_mask(seq_len, device)
        
        # High-level planning
        high_level_states = []
        high_x = x
        for layer in self.high_level_layers:
            high_x = layer(high_x, mask=mask)
            high_level_states.append(high_x)
        
        # Project high-level context
        hierarchy_context = self.hierarchy_proj(high_x)
        
        # Low-level execution with hierarchical guidance
        reasoning_states = []
        low_x = x
        
        if reasoning_steps is None:
            reasoning_steps = self.config.reasoning_steps
        
        for step in range(reasoning_steps):
            step_states = []
            
            # Update hierarchy context periodically
            if step % self.config.high_level_timescale == 0 and step > 0:
                hierarchy_context = self.hierarchy_proj(high_level_states[-1])
            
            # Low-level processing
            for i, layer in enumerate(self.low_level_layers):
                low_x = layer(low_x, mask=mask, hierarchy_context=hierarchy_context)
                step_states.append(low_x.clone())
            
            reasoning_states.append(step_states)
            
            # Check halt condition
            halt_prob = self.reasoning_halt(low_x)
            if torch.max(halt_prob) > self.config.halt_threshold and step >= 2:
                break
        
        # Final output
        output = self.output_norm(low_x)
        logits = self.lm_head(output)
        
        result = {
            'logits': logits,
            'reasoning_steps_taken': step + 1,
            'halt_probabilities': halt_prob
        }
        
        # Calculate loss if labels provided
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
            result['loss'] = loss
            
            # Additional losses for training stability
            if len(reasoning_states) > 1:
                consistency_loss = 0.0
                for i in range(1, len(reasoning_states)):
                    for j in range(len(reasoning_states[i])):
                        curr_state = reasoning_states[i][j]
                        prev_state = reasoning_states[i-1][j]
                        consistency_loss += F.mse_loss(curr_state, prev_state)
                consistency_loss /= (len(reasoning_states) - 1) * len(reasoning_states[0])
                result['consistency_loss'] = consistency_loss * 0.01
            
            halt_reg = torch.mean(halt_prob) * 0.001
            result['halt_regularization'] = halt_reg
            
            total_loss = loss
            if 'consistency_loss' in result:
                total_loss += result['consistency_loss']
            total_loss += halt_reg
            result['total_loss'] = total_loss
        
        if return_reasoning_states:
            result['reasoning_states'] = reasoning_states
            result['high_level_states'] = high_level_states
        
        return result
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 512,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        reasoning_steps: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Generate text using the HRM model"""
        
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        generated_ids = input_ids.clone()
        reasoning_info = []
        
        with torch.no_grad():
            for step in range(max_length):
                outputs = self.forward(
                    generated_ids, 
                    reasoning_steps=reasoning_steps,
                    return_reasoning_states=False
                )
                
                next_token_logits = outputs['logits'][:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for token_id in set(generated_ids[i].tolist()):
                            if next_token_logits[i, token_id] < 0:
                                next_token_logits[i, token_id] *= repetition_penalty
                            else:
                                next_token_logits[i, token_id] /= repetition_penalty
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    mask_value = -65504.0 if next_token_logits.dtype == torch.float16 else -float('Inf')
                    next_token_logits[indices_to_remove] = mask_value
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    mask_value = -65504.0 if next_token_logits.dtype == torch.float16 else -float('Inf')
                    next_token_logits[indices_to_remove] = mask_value
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # Store reasoning information
                reasoning_info.append({
                    'reasoning_steps': outputs['reasoning_steps_taken'],
                    'halt_probability': outputs['halt_probabilities'].cpu().numpy(),
                    'token_probability': probs.gather(1, next_token).cpu().numpy()
                })
                
                # Check for end of sequence
                if torch.all(next_token == 2):  # Assuming 2 is EOS token
                    break
        
        return {
            'generated_ids': generated_ids,
            'reasoning_info': reasoning_info
        }
    
    @classmethod
    def from_pretrained(cls, model_path: str) -> 'AkiaHRM':
        """Load a pretrained model"""
        checkpoint = torch.load(model_path, map_location='cpu')
        config = AkiaHRMConfig(**checkpoint['config'])
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def save_pretrained(self, save_path: str):
        """Save the model"""
        checkpoint = {
            'config': {
                'vocab_size': self.config.vocab_size,
                'd_model': self.config.d_model,
                'n_layers_high': self.config.n_layers_high,
                'n_layers_low': self.config.n_layers_low,
                'n_heads': self.config.n_heads,
                'd_ff': self.config.d_ff,
                'max_sequence_length': self.config.max_sequence_length,
                'dropout': self.config.dropout,
                'layer_norm_epsilon': self.config.layer_norm_epsilon,
                'reasoning_steps': self.config.reasoning_steps,
                'halt_threshold': self.config.halt_threshold,
                'high_level_timescale': self.config.high_level_timescale,
                'cross_hierarchy_dim': self.config.cross_hierarchy_dim,
                'reasoning_head_dim': self.config.reasoning_head_dim
            },
            'model_state_dict': self.state_dict()
        }
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")

def count_parameters(model):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    breakdown = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if params > 0:
                breakdown[name] = params
    
    return total, breakdown
