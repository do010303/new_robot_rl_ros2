import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnx
from onnx import shape_inference
from onnxruntime.quantization import quantize_dynamic, QuantType

# Add parent directory for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Model dimensions (Drawing task)
STATE_DIM = 18
ACTION_DIM = 3
JOINT_DIM = 6

class InferenceActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256)):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden_dims[0])
        self.l2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.mean_linear = nn.Linear(hidden_dims[1], action_dim)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return torch.tanh(self.mean_linear(x))

class InferenceNeuralIK(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=512, output_dim=6):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        normalized = self.network(x)
        return normalized * 1.5708

def export_and_quantize(model_name, pth_path, onnx_path, model, input_dim):
    print(f"\n📦 Processing {model_name}...")
    
    if not os.path.exists(pth_path):
        print(f"❌ ERROR: {pth_path} not found")
        return False
        
    try:
        checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        
        if not filtered_dict:
            print(f"❌ ERROR: No matching keys found")
            return False
            
        model.load_state_dict(filtered_dict, strict=False)
        model.eval()
        print(f"✅ Weights loaded")
    except Exception as e:
        print(f"❌ ERROR: Failed to load: {e}")
        return False
        
    dummy_input = torch.randn(1, input_dim)
    
    try:
        # Export WITHOUT tracing first to avoid ScriptModule issues
        torch.onnx.export(
            model, dummy_input, onnx_path,
            export_params=True,
            opset_version=15, # Try 15 which is widely supported
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
        )
        
        # EXPLICIT SHAPE INFERENCE
        onnx_model = onnx.load(onnx_path)
        inferred_model = shape_inference.infer_shapes(onnx_model)
        onnx.save(inferred_model, onnx_path)
        
        print(f"✅ Exported and Shape-Inferred ONNX: {onnx_path}")
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False
    
    # Quantize
    quant_path = onnx_path.replace(".onnx", "_quant.onnx")
    try:
        quantize_dynamic(
            onnx_path, 
            quant_path, 
            weight_type=QuantType.QInt8
        )
        print(f"✅ Quantized (INT8) ONNX: {quant_path}")
    except Exception as e:
        print(f"❌ Quantization failed: {e}")
        return False
    
    return True

def main():
    print("="*70)
    print("🎨 DRAWING RL -> QUANTIZED ONNX EXPORT (v5)")
    print("="*70)
    
    checkpoints_dir = os.path.join(parent_dir, 'checkpoints')
    output_dir = os.path.join(script_dir, 'onnx_models')
    os.makedirs(output_dir, exist_ok=True)
    
    actor_pth = os.path.join(checkpoints_dir, 'sac_drawing_neuralIK', 'actor_sac_best.pth')
    export_and_quantize("Actor", actor_pth, os.path.join(output_dir, 'actor_drawing.onnx'), 
                         InferenceActor(STATE_DIM, ACTION_DIM), STATE_DIM)
    
    nik_pth = os.path.join(checkpoints_dir, 'neural_ik.pth')
    export_and_quantize("Neural IK", nik_pth, os.path.join(output_dir, 'neural_ik.onnx'), 
                         InferenceNeuralIK(3, 512, 6), 3)
    
    print("\n" + "="*70)
    print("✅ EXPORT COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
