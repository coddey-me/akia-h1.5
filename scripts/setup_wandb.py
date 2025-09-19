#!/usr/bin/env python3
"""
Automatic Weights & Biases setup for Akia HRM
Put your W&B credentials here and they'll be set up automatically
"""

import os
import wandb

def setup_wandb():
    """Setup W&B with credentials"""
    
    # ============================================
    # PUT YOUR W&B CREDENTIALS HERE
    # ============================================
    WANDB_API_KEY = "54cb45aff62ea6d2210b53c6bb899782245e1c37"  # Replace with your actual API key
    WANDB_ENTITY = "lacesseapp-lacesse-ventures"       # Replace with your W&B username (optional)
    
    # ============================================
    
    # Check if API key is set
    if WANDB_API_KEY == "YOUR_API_KEY_HERE":
        print("❌ Please set your WANDB_API_KEY in scripts/setup_wandb.py")
        print("   Get your API key from: https://wandb.ai/authorize")
        return False
    
    # Set environment variables
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    
    if WANDB_ENTITY != "your-username":
        os.environ["WANDB_ENTITY"] = WANDB_ENTITY
    
    # Test login
    try:
        wandb.login(key=WANDB_API_KEY)
        print("✅ W&B authentication successful!")
        return True
    except Exception as e:
        print(f"❌ W&B authentication failed: {e}")
        return False

if __name__ == "__main__":
    setup_wandb()
