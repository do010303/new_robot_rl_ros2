#!/usr/bin/env python3
import os
import sys
import onnx
from onnx import shape_inference
import onnxruntime.quantization.quant_utils as _qu
from onnxruntime.quantization import quantize_dynamic, QuantType

# Patch ONNX Runtime quantization utility to clear value_info before shape inference
def patched_load_model_with_shape_infer(model_path):
    model = onnx.load(str(model_path))
    model.graph.ClearField("value_info")
    inferred_model = shape_inference.infer_shapes(model)
    return inferred_model

_qu.load_model_with_shape_infer = patched_load_model_with_shape_infer

def main():
    onnx_dir = "/home/ducanh/new_rl_ros2/wicom_roboarm/onnx_models"
    
    actor_in = os.path.join(onnx_dir, "actor_drawing.onnx")
    actor_out = os.path.join(onnx_dir, "actor_drawing_quantized.onnx")
    
    nik_in = os.path.join(onnx_dir, "neural_ik.onnx")
    nik_out = os.path.join(onnx_dir, "neural_ik_quantized.onnx")
    
    if not os.path.exists(actor_in):
        print(f"Error: {actor_in} does not exist. Run export_onnx.py first.")
        sys.exit(1)
    if not os.path.exists(nik_in):
        print(f"Error: {nik_in} does not exist. Run export_onnx.py first.")
        sys.exit(1)
        
    print(f"Quantizing {actor_in} -> {actor_out}...")
    quantize_dynamic(
        model_input=actor_in,
        model_output=actor_out,
        weight_type=QuantType.QInt8
    )
    print("Actor quantization complete!")
    
    print(f"Quantizing {nik_in} -> {nik_out}...")
    quantize_dynamic(
        model_input=nik_in,
        model_output=nik_out,
        weight_type=QuantType.QInt8
    )
    print("Neural IK quantization complete!")

if __name__ == "__main__":
    main()
