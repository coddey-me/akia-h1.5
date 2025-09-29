#!/usr/bin/env python3
"""
Standalone inference script for Akia HRM (handles FP16 + mixed checkpoint formats)

Usage:
  python infr-fp16.py --model-path checkpoints/final_model.pt --prompt "Your question here"
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import os, json, torch
from model.hrm_architecture import AkiaHRM, AkiaHRMConfig
from utils.tokenizer import SimpleTokenizer

def load_model_and_tokenizer(model_path, tokenizer_path, device):
    """Load model + tokenizer with optional config.json fallback"""
    # Determine config path based on model_path folder
    config_path = os.path.join(os.path.dirname(model_path), "config.json")

    if os.path.exists(config_path):
        # Load config from file
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        print(f"Loaded config from {config_path}")
        model_config = AkiaHRMConfig(**config_dict)
    else:
        # Fallback to default AkiaHRMConfig
        print("⚠️ Config file not found, using default AkiaHRMConfig")
        model_config = AkiaHRMConfig()

    # Load tokenizer
    tokenizer = SimpleTokenizer.from_pretrained(tokenizer_path)

    # Build model
    model = AkiaHRM(model_config)
    state_dict = torch.load(model_path, map_location=device)

    # Handle whether you saved as full checkpoint or just model.state_dict
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"], strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)

    model.to(device)

    # (Optional) FP16
    try:
        model = model.half()
    except Exception as e:
        print(f"FP16 conversion failed: {e}")

    model.eval()
    return model, tokenizer, device, model_config



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
