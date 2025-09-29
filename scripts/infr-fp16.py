#!/usr/bin/env python3
"""
Standalone inference script for Akia HRM (handles FP16 + mixed checkpoint formats)

Usage:
  python infr-fp16.py --model-path checkpoints/final_model.pt --prompt "Your question here"
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import torch
from pathlib import Path
from model.hrm_architecture import AkiaHRM, AkiaHRMConfig
from utils.tokenizer import SimpleTokenizer  # adjust path if needed

def load_model_and_tokenizer(model_path, tokenizer_path=None, config_path=None):
    import json
    from pathlib import Path

    # If config path not provided, look for config.json next to model
    if config_path is None:
        config_path = Path(model_path).parent / "config.json"

    # Load config dict manually
        # Try to load config.json, else use default
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        model_config = AkiaHRMConfig(**config)
    else:
        print("⚠️ Config file not found, using default AkiaHRMConfig")
        model_config = AkiaHRMConfig()  # default params


    # Initialize config object
    config = AkiaHRMConfig(**config_dict)

    # Build model
    model = AkiaHRM(config)

    # Load weights
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)

    # Load tokenizer
    if tokenizer_path is not None:
        tokenizer = SimpleTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = SimpleTokenizer()  # or your default creation method

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, device, config


@torch.inference_mode()
def generate_response(model,
                      tokenizer,
                      device,
                      prompt,
                      max_length=256,
                      temperature=0.8,
                      top_k=50,
                      top_p=0.9,
                      reasoning_steps=8):
    """Generate text from model given a prompt"""
    input_tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_tokens], dtype=torch.long).to(device)

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

    # Remove prompt part if duplicated in output
    prompt_decoded = tokenizer.decode(input_tokens)
    if response.startswith(prompt_decoded):
        response = response[len(prompt_decoded):].strip()

    return response


def main():
    parser = argparse.ArgumentParser(description="Inference for Akia HRM (FP16-ready)")
    parser.add_argument('--model-path', required=True, help="Path to model checkpoint")
    parser.add_argument('--tokenizer-path', default='tokenizer_vocab.json', help="Path to tokenizer vocab json")
    parser.add_argument('--prompt', required=True, help="Prompt text for generation")
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--reasoning-steps', type=int, default=8)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model + tokenizer (don’t pass device here)
    model, tokenizer, _, config = load_model_and_tokenizer(
        args.model_path,
        tokenizer_path=args.tokenizer_path
    )

    model.to(device)

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
