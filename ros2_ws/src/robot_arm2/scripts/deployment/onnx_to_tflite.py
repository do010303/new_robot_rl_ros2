#!/usr/bin/env python3
"""
Step 2: Convert ONNX model to TFLite
Run this after pytorch_to_onnx.py

Usage:
    python3 onnx_to_tflite.py --model ../checkpoints/sac_gazebo/actor_sac_best.onnx
    python3 onnx_to_tflite.py --model ../checkpoints/sac_gazebo/actor_sac_best.onnx --quantize
"""

# IMPORTANT: Import TensorFlow FIRST
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import argparse
import tempfile
import shutil


def convert_to_tflite(onnx_path: str, quantize: bool = False):
    """Convert ONNX model to TFLite format"""
    
    print("="*70)
    print("🔄 Step 2: ONNX → TFLite Conversion")
    print("="*70)
    
    if not onnx_path.endswith('.onnx'):
        onnx_path = onnx_path + '.onnx'
    
    if not os.path.exists(onnx_path):
        print(f"❌ ERROR: ONNX model not found at {onnx_path}")
        return None
    
    print(f"\n📦 Loading ONNX model: {onnx_path}")
    print(f"   Quantization: {'ENABLED' if quantize else 'DISABLED'}")
    
    try:
        import onnx
        from onnx_tf.backend import prepare
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # FORCE IR VERSION to 9 to satisfy onnx checker/onnx-tf
        if onnx_model.ir_version > 9:
            print(f"⚠️  Downgrading IR version from {onnx_model.ir_version} to 9...")
            onnx_model.ir_version = 9
            
        print("✅ ONNX model loaded")
        
        # Convert to TensorFlow
        print("\n⚙️  Converting ONNX → TensorFlow...")
        tf_rep = prepare(onnx_model)
        
        # Save to temp directory
        temp_dir = tempfile.mkdtemp()
        saved_model_path = os.path.join(temp_dir, "saved_model")
        tf_rep.export_graph(saved_model_path)
        print("✅ TensorFlow conversion successful")
        
        # Convert to TFLite
        print("\n⚙️  Converting TensorFlow → TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        print("✅ TFLite conversion successful")
        
        # Save TFLite model
        base_path = onnx_path.replace('.onnx', '')
        suffix = "_quantized" if quantize else ""
        tflite_path = f"{base_path}{suffix}.tflite"
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        # Stats
        onnx_size = os.path.getsize(onnx_path)
        tflite_size = os.path.getsize(tflite_path)
        
        print("\n" + "="*70)
        print("✅ CONVERSION COMPLETE!")
        print("="*70)
        print(f"\n📊 Statistics:")
        print(f"   ONNX size:   {onnx_size / 1024:.1f} KB")
        print(f"   TFLite size: {tflite_size / 1024:.1f} KB")
        print(f"   Reduction:   {100*(1-tflite_size/onnx_size):.1f}%")
        print(f"\n📦 Output: {tflite_path}")
        print(f"\n📝 Deploy to Pi:")
        print(f"   ./deploy_to_pi.sh")
        
        return tflite_path
        
    except ImportError as e:
        print(f"❌ ERROR: Missing package: {e}")
        print("   pip install onnx onnx-tf tensorflow")
        return None
    except Exception as e:
        print(f"❌ ERROR: Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Convert ONNX to TFLite')
    parser.add_argument('--model', '-m', type=str, required=True, help='ONNX model path')
    parser.add_argument('--quantize', '-q', action='store_true', help='Apply quantization')
    
    args = parser.parse_args()
    convert_to_tflite(args.model, args.quantize)


if __name__ == '__main__':
    main()
