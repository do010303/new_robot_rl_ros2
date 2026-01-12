#!/usr/bin/env python3
"""
Step 1: Export PyTorch model to ONNX
Run this first, then run onnx_to_tflite.py

Usage:
    python3 pytorch_to_onnx.py --model ../checkpoints/sac_gazebo/actor_sac_best.pth
"""

import os
import sys
import argparse

# Model dimensions (match training)
STATE_DIM = 16
ACTION_DIM = 6
MAX_ACTION = 1.5708  # ±90° in radians


def export_to_onnx(model_path: str, agent_type: str = 'sac'):
    """Export PyTorch model to ONNX format"""
    
    print("="*70)
    print("🔄 Step 1: PyTorch → ONNX Export")
    print("="*70)
    
    import torch
    import torch.nn as nn
    
    # Handle .pth extension
    if model_path.endswith('.pth'):
        model_base = model_path[:-4]
    else:
        model_base = model_path
        model_path = model_path + '.pth'
    
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model not found at {model_path}")
        return None
    
    print(f"\n📦 Loading PyTorch model: {model_path}")
    print(f"   Agent type: {agent_type.upper()}")
    
    # Define actor network
    if agent_type == 'sac':
        class ActorNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(STATE_DIM, 256)
                self.l2 = nn.Linear(256, 256)
                self.mean_linear = nn.Linear(256, ACTION_DIM)
            
            def forward(self, x):
                x = torch.relu(self.l1(x))
                x = torch.relu(self.l2(x))
                return torch.tanh(self.mean_linear(x)) * MAX_ACTION
    else:
        class ActorNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(STATE_DIM, 400)
                self.l2 = nn.Linear(400, 300)
                self.l3 = nn.Linear(300, ACTION_DIM)
            
            def forward(self, x):
                x = torch.relu(self.l1(x))
                x = torch.relu(self.l2(x))
                return torch.tanh(self.l3(x)) * MAX_ACTION
    
    # Load weights
    actor = ActorNetwork()
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        actor.load_state_dict(state_dict, strict=False)
        actor.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ ERROR: Could not load weights: {e}")
        return None
    
    # Export to ONNX
    onnx_path = f"{model_base}.onnx"
    print(f"\n⚙️  Exporting to ONNX: {onnx_path}")
    
    dummy_input = torch.randn(1, STATE_DIM)
    
    torch.onnx.export(
        actor,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"✅ ONNX export successful!")
    print(f"\n📝 Next step:")
    print(f"   python3 onnx_to_tflite.py --model {onnx_path}")
    
    return onnx_path


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch to ONNX')
    parser.add_argument('--model', '-m', type=str, required=True, help='PyTorch model path')
    parser.add_argument('--agent', type=str, default='sac', choices=['sac', 'td3'])
    
    args = parser.parse_args()
    export_to_onnx(args.model, args.agent)


if __name__ == '__main__':
    main()
