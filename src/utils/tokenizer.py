from typing import List, Dict, Optional
import json
import re
from pathlib import Path

class SimpleTokenizer:
    """Simple word-level tokenizer for Akia HRM"""
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_built = False
        
        # Special tokens
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3,
            '<reasoning>': 4,
            '</reasoning>': 5,
            '<planning>': 6,
            '</planning>': 7
        }
        
        # Initialize with special tokens
        self.word_to_id.update(self.special_tokens)
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
    
    def build_vocab(self, texts: List[str], min_freq: int = 2):
        """Build vocabulary from texts"""
        word_counts = {}
        
        print(f"Building vocabulary from {len(texts)} texts...")
        
        for text in texts:
            # Simple tokenization
            words = self._tokenize_text(text)
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Filter by minimum frequency
        filtered_words = {word: count for word, count in word_counts.items() 
                         if count >= min_freq and word not in self.special_tokens}
        
        # Sort by frequency
        sorted_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)
        
        # Add to vocabulary (leave space for special tokens)
        available_slots = self.vocab_size - len(self.special_tokens)
        for word, _ in sorted_words[:available_slots]:
            if word not in self.word_to_id:
                self.word_to_id[word] = len(self.word_to_id)
        
        # Create reverse mapping
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.vocab_built = True
        
        print(f"Vocabulary built with {len(self.word_to_id)} tokens")
        print(f"Coverage: {len(filtered_words)} unique words, using {len(self.word_to_id)} tokens")
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Simple text tokenization"""
        # Convert to lowercase and split on whitespace/punctuation
        text = text.lower()
        # Keep alphanumeric and basic punctuation
        text = re.sub(r'[^\w\s\.,!?\-]', ' ', text)
        # Split and filter empty strings
        words = [word.strip() for word in text.split() if word.strip()]
        return words
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        
        words = self._tokenize_text(text)
        token_ids = [self.special_tokens['<bos>']]  # Start token
        
        for word in words:
            token_id = self.word_to_id.get(word, self.special_tokens['<unk>'])
            token_ids.append(token_id)
        
        token_ids.append(self.special_tokens['<eos>'])  # End token
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        words = []
        for token_id in token_ids:
            word = self.id_to_word.get(token_id, '<unk>')
            if word not in ['<pad>', '<bos>', '<eos>']:
                words.append(word)
        return ' '.join(words)
    
    def save_vocab(self, save_path: str):
        """Save vocabulary to file"""
        vocab_data = {
            'word_to_id': self.word_to_id,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        
        print(f"Vocabulary saved to {save_path}")
    
    def load_vocab(self, vocab_path: str):
        """Load vocabulary from file"""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.word_to_id = vocab_data['word_to_id']
        self.vocab_size = vocab_data['vocab_size']
        self.special_tokens = vocab_data['special_tokens']
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.vocab_built = True
        
        print(f"Vocabulary loaded from {vocab_path}")
        print(f"Vocabulary size: {len(self.word_to_id)}")
    
    @classmethod
    def from_pretrained(cls, vocab_path: str) -> 'SimpleTokenizer':
        """Create tokenizer from saved vocabulary"""
        tokenizer = cls()
        tokenizer.load_vocab(vocab_path)
        return tokenizer
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.word_to_id) if self.vocab_built else self.vocab_size
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token mappings"""
        return self.special_tokens.copy()

def create_tokenizer_from_data(data_path: str, vocab_size: int = 32000, save_path: Optional[str] = None) -> SimpleTokenizer:
    """Create and train tokenizer from training data"""
    import pickle
    
    # Load training data
    with open(data_path, 'rb') as f:
        training_data = pickle.load(f)
    
    # Extract all text
    texts = []
    for sample in training_data:
        if 'input_sequence' in sample:
            texts.append(sample['input_sequence'])
        if 'target_sequence' in sample:
            texts.append(sample['target_sequence'])
    
    print(f"Extracted {len(texts)} text samples for vocabulary building")
    
    # Create and train tokenizer
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    tokenizer.build_vocab(texts)
    
    # Save if path provided
    if save_path:
        tokenizer.save_vocab(save_path)
    
    return tokenizer
