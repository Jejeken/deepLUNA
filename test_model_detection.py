"""
Quick test for model detection in CXR v3
=======================================
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from model_management_v3.model_config import ModelConfig

def test_model_detection():
    """Test if models are detected correctly"""
    print("=" * 60)
    print("MODEL DETECTION TEST")
    print("=" * 60)

    try:
        # Create model config
        config = ModelConfig()

        print(f"Base path: {config.base_path}")
        print(f"Segmentation path: {config.seg_model_path}")
        print(f"Classification path: {config.clf_model_path}")
        print()

        # Check if directories exist
        print("DIRECTORY EXISTENCE CHECK:")
        print(f"Segmentation dir exists: {os.path.exists(config.seg_model_path)}")
        print(f"Classification dir exists: {os.path.exists(config.clf_model_path)}")
        print()

        # List actual files in directories
        if os.path.exists(config.seg_model_path):
            print("SEGMENTATION DIRECTORY CONTENTS:")
            for file in os.listdir(config.seg_model_path):
                file_path = os.path.join(config.seg_model_path, file)
                size = os.path.getsize(file_path) / (1024*1024)  # MB
                print(f"  {file} ({size:.1f} MB)")
        print()

        if os.path.exists(config.clf_model_path):
            print("CLASSIFICATION DIRECTORY CONTENTS:")
            for file in os.listdir(config.clf_model_path):
                file_path = os.path.join(config.clf_model_path, file)
                size = os.path.getsize(file_path) / (1024*1024)  # MB
                print(f"  {file} ({size:.1f} MB)")
        print()

        # Get detected models
        seg_models = config.get_model_names('segmentation')
        clf_models = config.get_model_names('classification')

        print("DETECTED MODELS:")
        print(f"Segmentation models: {seg_models}")
        print(f"Classification models: {clf_models}")
        print()

        # Test defaults
        default_seg = config.get_default_model('segmentation')
        default_clf = config.get_default_model('classification')

        print("DEFAULT MODELS:")
        print(f"Default segmentation: {default_seg.name if default_seg else 'None'}")
        print(f"Default classification: {default_clf.name if default_clf else 'None'}")
        print()

        print("TEST RESULT:")
        if seg_models and clf_models:
            print("✅ SUCCESS: Both model types detected")
            return True
        else:
            print("❌ FAILURE: Missing models")
            return False

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_model_detection()
    print(f"\nOverall result: {'PASSED' if success else 'FAILED'}")