#!/usr/bin/env python3
"""
Inference script for Akia HRM
Usage: python scripts/inference.py --model-path checkpoints/best_model.pt --prompt "Your question here"
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from model.hrm_architecture import AkiaHRM
from utils.tokenizer import SimpleTokenizer

def load_model_and_tokenizer(model_path: str, tokenizer_path: str = None):
    """Load trained model and tokenizer"""
    print(f"Loading model from: {model_path}")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AkiaHRM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Load tokenizer
    if tokenizer_path is None:
        tokenizer_path = "tokenizer_vocab.json"
    
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = SimpleTokenizer.from_pretrained(tokenizer_path)
    
    return model, tokenizer, device

def generate_response(
    model, 
    tokenizer, 
    device, 
    prompt: str,
    max_length: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    reasoning_steps: int = 8
):
    """Generate response for a given prompt"""
    
    print(f"\n🤔 Thinking with {reasoning_steps} reasoning steps...")
    
    # Encode prompt
    input_tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_tokens], dtype=torch.long).to(device)
    
    print(f"Input tokens: {len(input_tokens)}")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            reasoning_steps=reasoning_steps
        )
    
    # Decode response
    generated_ids = outputs['generated_ids'][0]
    response = tokenizer.decode(generated_ids.cpu().tolist())
    
    # Extract only the new part (after the prompt)
    prompt_decoded = tokenizer.decode(input_tokens)
    if response.startswith(prompt_decoded):
        response = response[len(prompt_decoded):].strip()
    
    # Get reasoning information
    reasoning_info = outputs['reasoning_info']
    avg_reasoning_steps = sum(info['reasoning_steps'] for info in reasoning_info) / len(reasoning_info)
    avg_halt_prob = sum(info['halt_probability'][0] for info in reasoning_info) / len(reasoning_info)
    
    return response, {
        'avg_reasoning_steps': avg_reasoning_steps,
        'avg_halt_probability': avg_halt_prob,
        'total_tokens_generated': len(generated_ids) - len(input_tokens)
    }

def interactive_mode(model, tokenizer, device):
    """Run interactive conversation mode"""
    print("\n🤖 Akia HRM Interactive Mode")
    print("=" * 40)
    print("Type your questions and press Enter. Type 'quit' or 'exit' to stop.")
    print("Type 'help' for commands.")
    print()
    
    # Default settings
    settings = {
        'max_length': 256,
        'temperature': 0.8,
        'top_k': 50,
        'top_p': 0.9,
        'reasoning_steps': 8
    }
    
    while True:
        try:
            # Get user input
            prompt = input("👤 You: ").strip()
            
            if not prompt:
                continue
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if prompt.lower() == 'help':
                print("\n📖 Commands:")
                print("  help - Show this help")
                print("  settings - Show current settings")
                print("  set <setting> <value> - Change setting")
                print("  quit/exit - Exit interactive mode")
                print("\n⚙️ Available settings:")
                for key, value in settings.items():
                    print(f"  {key}: {value}")
                print()
                continue
            
            if prompt.lower() == 'settings':
                print("\n⚙️ Current settings:")
                for key, value in settings.items():
                    print(f"  {key}: {value}")
                print()
                continue
            
            if prompt.lower().startswith('set '):
                try:
                    _, setting, value = prompt.split(' ', 2)
                    if setting in settings:
                        if setting in ['max_length', 'top_k', 'reasoning_steps']:
                            settings[setting] = int(value)
                        else:
                            settings[setting] = float(value)
                        print(f"✅ {setting} set to {settings[setting]}")
                    else:
                        print(f"❌ Unknown setting: {setting}")
                except:
                    print("❌ Invalid command format. Use: set <setting> <value>")
                continue
            
            # Generate response
            print("🧠 Akia:", end=" ", flush=True)
            
            response, info = generate_response(
                model, tokenizer, device, prompt, **settings
            )
            
            print(response)
            
            # Show reasoning info
            print(f"   💭 Reasoning: {info['avg_reasoning_steps']:.1f} steps, "
                  f"Halt: {info['avg_halt_probability']:.2f}, "
                  f"Tokens: {info['total_tokens_generated']}")
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description="Run inference with Akia HRM")
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="tokenizer_vocab.json",
        help="Path to tokenizer vocabulary file"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt for inference (if not provided, enters interactive mode)"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum generation length"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling"
    )
    
    parser.add_argument(
        "--reasoning-steps",
        type=int,
        default=8,
        help="Number of reasoning steps"
    )
    
    args = parser.parse_args()
    
    print("🧠 Akia HRM Inference")
    print("=" * 30)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Load model and tokenizer
    try:
        model, tokenizer, device = load_model_and_tokenizer(
            args.model_path, 
            args.tokenizer_path
        )
        print("✅ Model and tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Single prompt mode
    if args.prompt:
        print(f"\n👤 Prompt: {args.prompt}")
        
        response, info = generate_response(
            model, tokenizer, device, args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            reasoning_steps=args.reasoning_steps
        )
        
        print(f"🧠 Akia: {response}")
        print(f"\n📊 Generation Info:")
        print(f"   Reasoning steps: {info['avg_reasoning_steps']:.1f}")
        print(f"   Halt probability: {info['avg_halt_probability']:.3f}")
        print(f"   Tokens generated: {info['total_tokens_generated']}")
    
    # Interactive mode
    else:
        interactive_mode(model, tokenizer, device)

if __name__ == "__main__":
    main()
