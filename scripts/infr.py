#!/usr/bin/env python3
"""
Standalone inference script for Akia HRM
Usage:
  python inference_standalone.py --model-path checkpoints/final_model.pt --prompt "Your question here"
"""
from pathlib import Path
import sys
# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
#!/usr/bin/env python3
"""
Standalone inference script for Akia HRM model
Usage example:
python inference.py --model-path checkpoints/final_model.pt --prompt "What is AI?"
"""

import argparse
import torch
from model.hrm_architecture import AkiaHRM, AkiaHRMConfig
from utils.tokenizer import SimpleTokenizer  # Adjust import as needed

def load_model_and_tokenizer(model_path: str, tokenizer_path: str = "tokenizer_vocab.json"):
    print(f"Loading model from checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')

    # Load config saved in checkpoint and recreate config object
    config_dict = checkpoint.get('config', {})
    config = AkiaHRMConfig(**config_dict)

    model = AkiaHRM(config)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Handle DataParallel 'module.' prefix if present
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
        
    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = SimpleTokenizer.from_pretrained(tokenizer_path)

    return model, tokenizer, device

def generate_response(model, tokenizer, device, prompt, max_length=256, temperature=0.8, top_k=50, top_p=0.9, reasoning_steps=8):
    input_tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_tokens], dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            reasoning_steps=reasoning_steps,
        )
    generated_ids = outputs['generated_ids'][0]
    response = tokenizer.decode(generated_ids.cpu().tolist())

    # Remove prompt part from response if present
    prompt_decoded = tokenizer.decode(input_tokens)
    if response.startswith(prompt_decoded):
        response = response[len(prompt_decoded):].strip()

    return response

def main():
    parser = argparse.ArgumentParser(description="Inference for Akia HRM")
    parser.add_argument('--model-path', required=True, help="Path to model checkpoint")
    parser.add_argument('--tokenizer-path', default='tokenizer_vocab.json', help="Path to tokenizer vocab json")
    parser.add_argument('--prompt', required=True, help="Prompt text for generation")
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--reasoning-steps', type=int, default=8)

    args = parser.parse_args()

    model, tokenizer, device = load_model_and_tokenizer(args.model_path, args.tokenizer_path)
    print(f"\nPrompt: {args.prompt}\n")
    response = generate_response(
        model, tokenizer, device, args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        reasoning_steps=args.reasoning_steps,
    )
    print(f"Model response:\n{response}\n")

if __name__ == '__main__':
    main()
