#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch
import torch.nn as nn

# Add paths to sys.path
scripts_dir = "/home/ducanh/new_rl_ros2/ros2_ws/src/visual_servoing/scripts"
sys.path.insert(0, scripts_dir)
sys.path.insert(0, os.path.join(scripts_dir, "rl"))

from agents.sac_agent import GaussianActor
from rl.neural_ik import NeuralIKNetwork

class ActorONNXWrapper(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor
        
    def forward(self, state):
        mean, log_std = self.actor(state)
        return torch.tanh(mean)

class NeuralIKONNXWrapper(nn.Module):
    def __init__(self, model, pos_min, pos_max):
        super().__init__()
        self.model = model
        self.register_buffer('pos_min', torch.tensor(pos_min, dtype=torch.float32))
        self.register_buffer('pos_max', torch.tensor(pos_max, dtype=torch.float32))
        
    def forward(self, pos):
        pos_norm = 2 * (pos - self.pos_min) / (self.pos_max - self.pos_min + 1e-8) - 1
        return self.model(pos_norm)

def main():
    onnx_dir = "/home/ducanh/new_rl_ros2/wicom_roboarm/onnx_models"
    os.makedirs(onnx_dir, exist_ok=True)
    
    # 1. Export Actor
    actor_checkpoint = os.path.join(scripts_dir, "checkpoints", "sac_drawing_neuralIK", "actor_sac_best.pth")
    if not os.path.exists(actor_checkpoint):
        print(f"Error: Actor checkpoint not found at {actor_checkpoint}")
        sys.exit(1)
        
    print(f"Loading Actor checkpoint from {actor_checkpoint}...")
    actor = GaussianActor(state_dim=18, action_dim=3, max_action=np.array([1.0, 1.0, 1.0]))
    actor.load_state_dict(torch.load(actor_checkpoint, map_location="cpu", weights_only=False))
    actor.eval()
    
    actor_wrapper = ActorONNXWrapper(actor)
    dummy_state = torch.zeros(1, 18)
    actor_onnx_path = os.path.join(onnx_dir, "actor_drawing.onnx")
    
    print(f"Exporting Actor to ONNX format at {actor_onnx_path}...")
    torch.onnx.export(
        actor_wrapper,
        dummy_state,
        actor_onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print("Actor export complete!")
    
    # 2. Export Neural IK
    nik_checkpoint = os.path.join(scripts_dir, "checkpoints", "neural_ik.pth")
    if not os.path.exists(nik_checkpoint):
        print(f"Error: Neural IK checkpoint not found at {nik_checkpoint}")
        sys.exit(1)
        
    print(f"Loading Neural IK checkpoint from {nik_checkpoint}...")
    checkpoint = torch.load(nik_checkpoint, map_location="cpu", weights_only=False)
    
    nik_net = NeuralIKNetwork()
    nik_net.load_state_dict(checkpoint['model_state_dict'])
    nik_net.eval()
    
    pos_min = checkpoint['pos_min']
    pos_max = checkpoint['pos_max']
    
    nik_wrapper = NeuralIKONNXWrapper(nik_net, pos_min, pos_max)
    dummy_pos = torch.zeros(1, 3)
    nik_onnx_path = os.path.join(onnx_dir, "neural_ik.onnx")
    
    print(f"Exporting Neural IK to ONNX format at {nik_onnx_path}...")
    torch.onnx.export(
        nik_wrapper,
        dummy_pos,
        nik_onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print("Neural IK export complete!")
    
if __name__ == "__main__":
    main()
