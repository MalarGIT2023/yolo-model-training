"""
YOLOv11 Emotion Detection Model Training Script
===============================================

This script trains a YOLOv11 Nano model for facial emotion recognition.

The training process:
1. Loads pre-trained YOLOv11 Nano weights (transfer learning)
2. Fine-tunes on emotion detection dataset
3. Generates trained model weights optimized for emotion classification

Prerequisites:
- Roboflow dataset downloaded (see ../roboflow/ROBOFLOW_GUIDE.md)
- YOLOv11 base model (yolo11n.pt) in this directory
- Dependencies installed: pip install -r ultralytics-requirements.txt

Output:
- Trained model: runs/detect/train/weights/best.pt
- Training logs: runs/detect/train/
- Results visualization: runs/detect/train/results.csv

For detailed training configuration, see README.md in this directory
"""

from ultralytics import YOLO

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

# Load pre-trained YOLOv11 Nano model as base for transfer learning
# yolo11n.pt = YOLOv11 Nano (lightweight, fast, good for edge devices)
# Other options: yolo11s.pt (small), yolo11m.pt (medium), yolo11l.pt (large)
# Nano model is optimized for Raspberry Pi deployment
print("Loading YOLOv11 Nano base model for transfer learning...")
model = YOLO("yolo11n.pt")

# ============================================================================
# MODEL TRAINING CONFIGURATION
# ============================================================================

print("Starting training on emotion detection dataset...")
print("-" * 70)

# Train the model with emotion detection dataset
model.train(
    # ======== DATASET CONFIGURATION ========
    data="datasets/data.yaml",           # Path to dataset config file
                                         # Must contain:
                                         # - path: dataset root directory
                                         # - train: training images path
                                         # - val: validation images path
                                         # - nc: number of classes (10 emotions)
                                         # - names: emotion class names
    
    # ======== TRAINING PARAMETERS ========
    epochs=200,                          # Number of training epochs (full dataset passes)
                                         # Higher = more training time but potentially better accuracy
                                         # Typical range: 100-300
    
    imgsz=320,                          # Input image size (width x height)
                                         # Options: 320, 416, 512, 640, etc.
                                         # Larger = better accuracy but slower training
                                         # Smaller = faster training but lower accuracy
    
    batch=10,                           # Batch size (images per training step)
                                         # Depends on available GPU/CPU memory
                                         # Larger batches = more stable training but slower
    
    device="cpu",                       # Device to use for training
                                         # Options: "cpu", "0" (GPU:0), "0,1" (multi-GPU)
                                         # CPU is slower but works everywhere
                                         # GPU training recommended for faster results
    
    name="emotionsbest.pt",             # Name of training run
                                         # Output saved to: runs/detect/emotionsbest.pt/
    
    workers=4,                          # Number of workers for data loading
                                         # Higher = faster data loading (if CPU allows)
                                         # Typical: 2-8 depending on CPU cores
    
    lr0=0.001,                          # Initial learning rate (0.001 = 0.1%)
                                         # Affects training speed and convergence
                                         # Lower = slower but stable training
                                         # Higher = faster but may diverge
    
    augment=True,                       # Enable data augmentation
                                         # Applies: rotation, scaling, flipping, etc.
                                         # Improves model robustness and reduces overfitting
)

# ============================================================================
# TRAINING COMPLETE
# ============================================================================

print("-" * 70)
print("✓ Training completed successfully!")
print("\nOutput:")
print("  - Best model: runs/detect/emotionsbest.pt/weights/best.pt")
print("  - Training results: runs/detect/emotionsbest.pt/results.csv")
print("  - Training logs: runs/detect/emotionsbest.pt/")
print("\nNext steps:")
print("  1. Review training results: runs/detect/emotionsbest.pt/results.png")
print("  2. Test the model: python test_model.py")
print("  3. Deploy to Raspberry Pi: Copy weights/best.pt to deployment folder")

