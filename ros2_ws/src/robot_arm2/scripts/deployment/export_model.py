#!/usr/bin/env python3
"""
Export Trained RL Model for Raspberry Pi Deployment

Two export options:
1. ONNX format (recommended, works well on Pi with onnxruntime)
2. TorchScript format (if ONNX fails)

Usage:
    # Export SAC model to ONNX (default)
    python3 export_model.py --model ../checkpoints/sac_gazebo/actor_sac_best.pth
    
    # Export TD3 model
    python3 export_model.py --model ../checkpoints/td3/actor_td3_best.pth --agent td3
    
    # Export to TorchScript instead
    python3 export_model.py --model ../checkpoints/sac_gazebo/actor_sac_best.pth --format torchscript

Model Specs (6DOF Robot - Direct Joint Control):
    - State dim: 18 (joints(6) + robot_xyz(3) + target_xyz(3) + dist(4) + vel(2))
    - Action dim: 6 (joint angle deltas)
"""

import os
import sys
import argparse
import numpy as np

# Add parent directory for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)


def export_to_onnx(model_path: str, output_path: str = None,
                   state_dim: int = 18, action_dim: int = 6, agent_type: str = 'sac'):
    """
    Export PyTorch model to ONNX format
    """
    import torch
    import torch.nn as nn
    
    print("=" * 70)
    print("🔄 ONNX Model Export")
    print("=" * 70)
    
    # Handle .pth extension
    if model_path.endswith('.pth'):
        model_base = model_path[:-4]
    else:
        model_base = model_path
        model_path = model_path + '.pth'
    
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model file not found: {model_path}")
        return False
    
    print(f"\n📦 Model: {model_path}")
    print(f"   Agent type: {agent_type.upper()}")
    print(f"   State dim: {state_dim}")
    print(f"   Action dim: {action_dim}")
    
    # Recreate actor network architecture
    print(f"\n🏗️  Creating actor network...")
    
    if agent_type == 'sac':
        # SAC GaussianActor - simplified for inference
        class ActorNetwork(nn.Module):
            def __init__(self, state_dim, action_dim, max_action=0.1):
                super().__init__()
                self.max_action = max_action
                self.l1 = nn.Linear(state_dim, 256)
                self.l2 = nn.Linear(256, 256)
                self.mean_linear = nn.Linear(256, action_dim)
            
            def forward(self, x):
                x = torch.relu(self.l1(x))
                x = torch.relu(self.l2(x))
                mean = self.mean_linear(x)
                return torch.tanh(mean) * self.max_action
    else:
        # TD3 Actor
        class ActorNetwork(nn.Module):
            def __init__(self, state_dim, action_dim, max_action=0.1):
                super().__init__()
                self.max_action = max_action
                self.l1 = nn.Linear(state_dim, 400)
                self.l2 = nn.Linear(400, 300)
                self.l3 = nn.Linear(300, action_dim)
            
            def forward(self, x):
                x = torch.relu(self.l1(x))
                x = torch.relu(self.l2(x))
                return torch.tanh(self.l3(x)) * self.max_action
    
    actor = ActorNetwork(state_dim, action_dim, max_action=0.1)
    
    # Load weights
    print(f"📂 Loading weights...")
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Try to load (may have extra keys like log_std_linear for SAC)
    try:
        actor.load_state_dict(state_dict, strict=False)
        print("✅ Weights loaded (some keys may be skipped for inference)")
    except Exception as e:
        print(f"⚠️  Partial load: {e}")
    
    actor.eval()
    actor.cpu()
    
    # Export to ONNX
    if output_path is None:
        output_path = f"{model_base}.onnx"
    
    print(f"\n⚙️  Exporting to ONNX...")
    
    dummy_input = torch.randn(1, state_dim)
    
    torch.onnx.export(
        actor,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['state'],
        output_names=['action'],
        dynamic_axes={'state': {0: 'batch'}, 'action': {0: 'batch'}}
    )
    
    print(f"✅ Exported to: {output_path}")
    
    # Verify ONNX model
    print(f"\n🧪 Verifying ONNX model...")
    
    try:
        import onnx
        import onnxruntime as ort
        
        # Check model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model is valid")
        
        # Test inference
        session = ort.InferenceSession(output_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        test_input = np.random.randn(1, state_dim).astype(np.float32)
        result = session.run([output_name], {input_name: test_input})[0]
        
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {result.shape}")
        print(f"   Output: {result[0]}")
        print(f"   Output range: [{result.min():.4f}, {result.max():.4f}]")
        print("✅ Inference test passed!")
        
    except Exception as e:
        print(f"⚠️  Verification warning: {e}")
    
    # Statistics
    original_size = os.path.getsize(model_path)
    onnx_size = os.path.getsize(output_path)
    
    print(f"\n📊 Statistics:")
    print(f"   Original (.pth): {original_size / 1024:.1f} KB")
    print(f"   ONNX: {onnx_size / 1024:.1f} KB")
    
    print("\n" + "=" * 70)
    print("✅ EXPORT COMPLETE!")
    print("=" * 70)
    print(f"\n📦 Model ready: {output_path}")
    print(f"\n📝 On Pi, use ONNX Runtime:")
    print(f"   pip3 install onnxruntime")
    print(f"   python3 deploy_on_pi.py --model {os.path.basename(output_path)}")
    
    return True


def export_to_torchscript(model_path: str, output_path: str = None,
                          state_dim: int = 18, action_dim: int = 6, agent_type: str = 'sac'):
    """
    Export PyTorch model to TorchScript format (jit.trace)
    """
    import torch
    import torch.nn as nn
    
    print("=" * 70)
    print("🔄 TorchScript Model Export")
    print("=" * 70)
    
    # Handle .pth extension
    if model_path.endswith('.pth'):
        model_base = model_path[:-4]
    else:
        model_base = model_path
        model_path = model_path + '.pth'
    
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model file not found: {model_path}")
        return False
    
    print(f"\n📦 Model: {model_path}")
    
    # Create network
    if agent_type == 'sac':
        class ActorNetwork(nn.Module):
            def __init__(self, state_dim, action_dim, max_action=0.1):
                super().__init__()
                self.max_action = max_action
                self.l1 = nn.Linear(state_dim, 256)
                self.l2 = nn.Linear(256, 256)
                self.mean_linear = nn.Linear(256, action_dim)
            
            def forward(self, x):
                x = torch.relu(self.l1(x))
                x = torch.relu(self.l2(x))
                return torch.tanh(self.mean_linear(x)) * self.max_action
    else:
        class ActorNetwork(nn.Module):
            def __init__(self, state_dim, action_dim, max_action=0.1):
                super().__init__()
                self.max_action = max_action
                self.l1 = nn.Linear(state_dim, 400)
                self.l2 = nn.Linear(400, 300)
                self.l3 = nn.Linear(300, action_dim)
            
            def forward(self, x):
                x = torch.relu(self.l1(x))
                x = torch.relu(self.l2(x))
                return torch.tanh(self.l3(x)) * self.max_action
    
    actor = ActorNetwork(state_dim, action_dim, max_action=0.1)
    state_dict = torch.load(model_path, map_location='cpu')
    actor.load_state_dict(state_dict, strict=False)
    actor.eval()
    
    # Trace and save
    if output_path is None:
        output_path = f"{model_base}_traced.pt"
    
    print(f"\n⚙️  Tracing model...")
    
    dummy_input = torch.randn(1, state_dim)
    traced = torch.jit.trace(actor, dummy_input)
    traced.save(output_path)
    
    print(f"✅ Saved to: {output_path}")
    
    # Test
    print(f"\n🧪 Testing...")
    loaded = torch.jit.load(output_path)
    test_input = torch.randn(1, state_dim)
    output = loaded(test_input)
    print(f"   Output: {output[0].detach().numpy()}")
    print("✅ Test passed!")
    
    original_size = os.path.getsize(model_path)
    traced_size = os.path.getsize(output_path)
    print(f"\n📊 Size: {original_size/1024:.1f} KB → {traced_size/1024:.1f} KB")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Export trained RL model for Raspberry Pi',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export to ONNX (recommended)
  python3 export_model.py --model ../checkpoints/sac_gazebo/actor_sac_best.pth
  
  # Export TD3 model  
  python3 export_model.py --model ../checkpoints/td3/actor_td3_best.pth --agent td3
  
  # Export to TorchScript
  python3 export_model.py --model ../checkpoints/sac_gazebo/actor_sac_best.pth --format torchscript
        """
    )
    
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Path to PyTorch model (.pth file)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file path')
    parser.add_argument('--agent', type=str, default='sac', choices=['sac', 'td3'],
                        help='Agent type: sac or td3 (default: sac)')
    parser.add_argument('--format', '-f', type=str, default='onnx', choices=['onnx', 'torchscript'],
                        help='Export format: onnx or torchscript (default: onnx)')
    parser.add_argument('--state-dim', type=int, default=18,
                        help='State dimension (default: 18)')
    parser.add_argument('--action-dim', type=int, default=6,
                        help='Action dimension (default: 6)')
    
    args = parser.parse_args()
    
    if args.format == 'onnx':
        success = export_to_onnx(
            model_path=args.model,
            output_path=args.output,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            agent_type=args.agent
        )
    else:
        success = export_to_torchscript(
            model_path=args.model,
            output_path=args.output,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            agent_type=args.agent
        )
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
