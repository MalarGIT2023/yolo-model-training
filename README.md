# YOLOv11 Emotion Detection Model Training

**Mission Tomorrow Career Exploration Event hosted by Chamber RVA**  
*Presented to 11,000+ eighth graders in Richmond*  
*Volunteered for IEEE Region3 Richmond*

---

This directory contains scripts and configuration for training YOLOv11 models to recognize facial emotions. This is **Step 2** of the emotion detection workflow.

## What This Project Does

The `model-training.py` script teaches a YOLOv11 model to recognize emotions by using **transfer learning** — starting with a pre-trained model and fine-tuning it with emotion detection data.

### The Machine Learning Process

Think of training a model like teaching a student:

```
Step 1: Start with a Smart Student
   └─ YOLOv11 Nano (yolo11n.pt)
      - Already knows how to detect objects
      - Pre-trained on 1M+ images
      - Ready to specialize in emotions

Step 2: Show Lots of Examples
   └─ Roboflow emotion dataset
      - Thousands of labeled emotion images
      - Different faces, ages, ethnicities
      - Each labeled: happy, sad, angry, etc.

Step 3: Let Them Practice & Learn
   └─ Train for 200 epochs (repetitions)
      - Model adjusts its "brain" weights
      - Gets better at recognizing patterns
      - Learns what makes faces happy vs sad

Step 4: Test & Save the Best
   └─ runs/detect/train/weights/best.pt
      - Most accurate version saved
      - Ready to use in real-world apps!
      - Only 6.5 MB (fits on Raspberry Pi)
```

### Why Transfer Learning?

Training from scratch would take weeks on a regular computer. Transfer learning:
- Reuses knowledge from 1M+ images
- Trains in minutes/hours (not days)
- Needs fewer examples (thousands vs millions)
- Works great for specialized tasks

### Career Connection: Machine Learning Engineer

This process is what **ML Engineers** do:
- Prepare datasets
- Train models
- Evaluate performance
- Deploy to production
- Monitor and improve

## Quick Start

### Prerequisites

1. **Get Your Roboflow Dataset**
   
   First, prepare the emotion detection dataset using the separate **roboflow-dataset-manager** project:
   
   ```bash
   cd ../roboflow-dataset-manager
   python dataset-download-roboflow.py
   # Downloads emotion detection data in YOLOv11 format
   ```
   
   This creates a `datasets/` folder with:
   ```
   datasets/
   ├── data.yaml           # Dataset configuration
   ├── train/images/       # Training images
   ├── train/labels/       # Training annotations
   ├── valid/images/       # Validation images
   ├── valid/labels/       # Validation annotations
   └── test/images/        # Test images
   ```
   
   📖 **For detailed dataset setup**: See [../roboflow-dataset-manager/README.md](../roboflow-dataset-manager/README.md)

2. **Dependencies**
   ```bash
   pip install -r ultralytics-requirements.txt
   ```

3. **Base Model**
   - `yolo11n.pt` should already be in this directory
   - If missing, it will auto-download on first run

### Run Training

```bash
python model-training.py
```

**Expected output:**
```
Loading YOLOv11 Nano base model for transfer learning...
Starting training on emotion detection dataset...
----------------------------------------------------------------------
...training progress...
----------------------------------------------------------------------
✓ Training completed successfully!

Output:
  - Best model: runs/detect/emotionsbest.pt/weights/best.pt
  - Training results: runs/detect/emotionsbest.pt/results.csv
```

**Training time estimates:**
- CPU (typical): 2-4 hours
- GPU (NVIDIA): 30-60 minutes
- Raspberry Pi: Not recommended (use pre-trained model instead)

## Configuration Parameters Explained

### `data="./datasets/data.yaml"`
- Path to dataset configuration file downloaded via roboflow-dataset-manager
- Points to your emotion detection dataset from Roboflow
- This YAML file contains:
  - `path:` dataset root directory
  - `train:` path to training images
  - `val:` path to validation images
  - `nc:` 10 (number of emotion classes)
  - `names:` list of emotion class names (e.g., happy, sad, angry, etc.)

**Important**: Make sure to run the roboflow-dataset-manager script first to generate this file!

### `epochs=200`
- Number of complete passes through the training dataset
- Each epoch:
  - Model sees all training images once
  - Weights updated based on performance
  - Validation performed to check progress

**Guidelines:**
- `100-150` epochs: Quick training (1-2 hours), decent accuracy
- `200` epochs: Balanced (2-4 hours), good accuracy
- `300+` epochs: Long training (4+ hours), may overfit

### `imgsz=320`
- Input image size (320×320 pixels)
- Trade-off between speed and accuracy
- Options:
  - `320`: Fast, lower accuracy (good for RPi deployment)
  - `512`: Medium (good balance)
  - `640`: Slower, higher accuracy (best for accuracy)

### `batch=10`
- Number of images processed per training step
- Depends on available memory
- If "out of memory" error: reduce to 8, 5, or 2
- Larger batches train faster but need more memory

### `device="cpu"`
- Where computation happens
- Options:
  - `"cpu"`: CPU-only (slow but works everywhere)
  - `"0"`: GPU 0 (fast, if NVIDIA CUDA available)
  - `"0,1"`: Multiple GPUs (fastest)
  
**Recommendation:**
- For quick testing: Use CPU
- For production: Use GPU (10x faster)

### `name="emotionsbest.pt"`
- Name of training run
- Output folder: `runs/detect/emotionsbest.pt/`
- Change name for different training runs:
  ```python
  name="emotionsbest_v2.pt"  # For second training run
  ```

### `workers=4`
- Parallel workers for loading data
- Depends on CPU cores
- If your CPU has 4 cores: use `workers=4`
- If slower: reduce to `workers=2`

### `lr0=0.001`
- Learning rate (how much weights change per step)
- Lower (0.0001): Slow, stable training
- Higher (0.01): Fast, may diverge/fail
- Default (0.001): Recommended balance

### `augment=True`
- Data augmentation (modify images during training)
- Includes: rotation, scaling, color changes, etc.
- Benefits:
  - More diverse training examples
  - Better model generalization
  - Reduces overfitting
  - Recommended: Always `True`

## Output Structure

After training completes:

```
runs/detect/emotionsbest.pt/
├── weights/
│   ├── best.pt              ← Use this for deployment
│   └── last.pt
├── results.csv              ← Training metrics
├── results.png              ← Training charts
├── confusion_matrix.png     ← Per-class performance
└── val_batch*.jpg           ← Example predictions
```

### Best Model (`best.pt`)
- Best weights from all 200 epochs
- Highest validation accuracy
- Ready to deploy to Raspberry Pi
- File size: ~6.5 MB

## Monitoring Training Progress

### Option 1: Live Terminal Output
Training progress shown in console with metrics:
```
Epoch 1/200: loss=2.34, val_loss=2.12, accuracy=0.45
Epoch 2/200: loss=2.10, val_loss=1.98, accuracy=0.58
...
```

### Option 2: View Results Chart
```bash
# After training completes
open runs/detect/emotionsbest.pt/results.png
# or
display runs/detect/emotionsbest.pt/results.png  # Linux
```

### Option 3: Parse Results CSV
```bash
cat runs/detect/emotionsbest.pt/results.csv
```

## Advanced Training Configurations

### Scenario 1: Quick Testing (30 minutes)
```python
model.train(
    data="../roboflow-dataset-manager/datasets/data.yaml",
    epochs=50,              # Fewer epochs
    imgsz=320,
    batch=10,
    device="cpu"
)
```

### Scenario 2: High Accuracy (4+ hours)
```python
model.train(
    data="../roboflow-dataset-manager/datasets/data.yaml",
    epochs=300,             # More epochs
    imgsz=512,              # Larger images
    batch=16,
    device="0"              # Use GPU
)
```

### Scenario 3: Multiple GPU Training
```python
model.train(
    data="../roboflow-dataset-manager/datasets/data.yaml",
    epochs=200,
    imgsz=512,
    batch=32,               # Larger batch (uses multiple GPUs)
    device="0,1"            # Use GPU 0 and 1
)
```

### Scenario 4: Lightweight for RPi Deployment
```python
model.train(
    data="../roboflow-dataset-manager/datasets/data.yaml",
    epochs=150,
    imgsz=320,              # Small images
    batch=8,
    device="cpu"
)
```

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size or image size
```python
batch=5         # Instead of 10
imgsz=320       # Instead of 512
```

### Issue: "Dataset not found"
**Solution**: Ensure dataset is downloaded from Roboflow
```bash
# Check dataset exists in roboflow-dataset-manager
ls ../roboflow-dataset-manager/datasets/data.yaml
ls ../roboflow-dataset-manager/datasets/train/images/
ls ../roboflow-dataset-manager/datasets/train/labels/

# If not found, run:
cd ../roboflow-dataset-manager
python dataset-download-roboflow.py
cd ../yolo-model-training
```

### Issue: "Training too slow"
**Solution**: Use GPU or reduce settings
```python
device="0"      # Use GPU if available
imgsz=320       # Smaller images
batch=10        # OK batch size
epochs=100      # Fewer epochs for testing
```

### Issue: "Model accuracy not improving"
**Solutions**:
1. Increase training time: `epochs=300`
2. Use larger images: `imgsz=512`
3. Verify dataset quality (check annotations)
4. Try different learning rate: `lr0=0.0005`

### Issue: "runs/detect directory too large"
**Solution**: Clean up old training runs
```bash
rm -rf runs/detect/emotionsbest.pt*/
# Keep only the best version
```

## Using the Trained Model

### Option 1: Direct Python Usage
```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/detect/emotionsbest.pt/weights/best.pt')

# Predict on image
results = model('image.jpg')

# Display results
results[0].show()
```

### Option 2: Deploy to Raspberry Pi
```bash
# Copy trained model to deployment
cp runs/detect/emotionsbest.pt/weights/best.pt \
   ../../face-emotion-detection-yolo/yolo-trained-models/emotionnanomodel.pt
```

### Option 3: Export to ONNX (Deployment)
```python
from ultralytics import YOLO

model = YOLO('runs/detect/emotionsbest.pt/weights/best.pt')

# Export to ONNX format (more portable)
model.export(format='onnx')
# Output: best.onnx (can use on different platforms)
```

## Performance Metrics

### Typical Results (After 200 Epochs)
- **Precision**: 0.85-0.92 (% of detections that are correct)
- **Recall**: 0.80-0.88 (% of objects detected)
- **mAP50**: 0.87-0.94 (overall accuracy at IOU 0.5)
- **Inference Speed**: 50-100ms per image (CPU)

### Interpreting results.csv
```
epoch,train/loss,val/loss,metrics/precision,metrics/recall,metrics/mAP50
0,2.34,2.12,0.45,0.52,0.38
1,2.10,1.98,0.58,0.65,0.51
...
199,0.12,0.18,0.89,0.86,0.87
```

- Lower loss = better training
- Higher precision/recall = better detections

## Understanding Your Dataset

The dataset downloaded from Roboflow is automatically organized in YOLOv11 format:

**Training Data**: Used to teach the model
- Located in: `../roboflow-dataset-manager/datasets/train/`
- ~70-80% of total images
- Typically 1000+ labeled emotion images

**Validation Data**: Used to check progress during training
- Located in: `../roboflow-dataset-manager/datasets/valid/`
- ~10-15% of total images
- Monitor to prevent overfitting

**Test Data**: Used to evaluate final model
- Located in: `../roboflow-dataset-manager/datasets/test/`
- ~10-15% of total images
- Never seen during training

**Switching Datasets**: To use a different Roboflow dataset
1. Edit `dataset-download-roboflow.py` in roboflow-dataset-manager
2. Change workspace, project, or version
3. Re-run the download script
4. Training will automatically use the new dataset

## Next Steps After Training

1. **Evaluate Results**
   - Review `results.png` chart
   - Check `confusion_matrix.png`
   - Analyze metrics in `results.csv`

2. **Test on New Images**
   - Create simple test script
   - Verify emotion predictions
   - Check confidence scores

3. **Deploy Model**
   - Copy to Raspberry Pi
   - Update deployment scripts
   - Run real-time inference

4. **Iterate if Needed**
   - Collect more diverse data
   - Retrain with new data
   - Compare model versions

## Model Comparison

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| Nano (11n) | 6.5 MB | Fast | Good |
| Small (11s) | 13 MB | Medium | Better |
| Medium (11m) | 33 MB | Slower | Best |
| Large (11l) | 55 MB | Slowest | Excellent |

**Recommendation for RPi**: Use Nano (11n) - best balance of size and speed

## Resources

- **YOLOv11 Docs**: https://docs.ultralytics.com/models/yolov11/
- **Ultralytics Training Guide**: https://docs.ultralytics.com/modes/train/
- **Roboflow Integration**: See [../roboflow-dataset-manager/README.md](../roboflow-dataset-manager/README.md)
- **Transfer Learning**: https://docs.ultralytics.com/modes/train/#transfer-learning

## Security & Best Practices

1. **Backup trained models**: `cp runs/detect/emotionsbest.pt/weights/best.pt backup/`
2. **Track experiments**: Different `name` parameter for each run
3. **Version datasets**: Keep separate version directories
4. **Monitor resources**: Check CPU/GPU usage during training
5. **Save results**: Archive training logs before next training

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) file for full details.

**MIT License Summary**: You are free to use, modify, and distribute this software for any purpose, provided you include the original license and copyright notice.

## Credits & Acknowledgments

**Created for**: IEEE Mission Tomorrow Career Exploration Event  
**Event**: Hosted by Chamber RVA for 11,000+ eighth graders in Richmond  
**Presented by**: IEEE Region 3 Richmond

**External Dependencies**:
- **Ultralytics YOLOv11** — Object detection and transfer learning framework
- **PyTorch** — Deep learning framework
- **Roboflow Python SDK** — Dataset downloading and management
- **OpenCV** — Computer vision library

**Special Thanks**:
- IEEE Region 3 Richmond for volunteering
- Chamber RVA for organizing Mission Tomorrow
- All educators supporting STEM education

---

**Last Updated**: November 2025  
**Model**: YOLOv11 Nano  
**Framework**: Ultralytics  
**Use Case**: Facial Emotion Detection
