# EfficientNet-B7 (V1)

## Overview

**EfficientNet-B7** is a video deepfake detection model that employs a frame-by-frame analysis approach using face detection and classification. The architecture combines **MTCNN** (Multi-task Cascaded Convolutional Networks) for robust face detection with an **EfficientNet-B7** encoder for deepfake classification.

**Model Type:** Video Analysis  
**Primary Use:** Frame-level face-based deepfake detection  
**Architecture:** MTCNN + EfficientNet-B7 + Custom Aggregation  

### Key Characteristics

- **Face-Centric Detection**: Uses MTCNN to detect and extract faces from every frame
- **Per-Face Classification**: Each detected face is independently analyzed
- **Temporal Aggregation**: Employs a confidence-based strategy to aggregate predictions across all frames
- **Isotropic Preprocessing**: Maintains aspect ratio during face crop resizing
- **Visualization Generation**: Optional frame-by-frame suspicion graph overlay

---

## How It Works

### Pipeline Overview

```text
Input Video
    ↓
Frame Extraction
    ↓
MTCNN Face Detection (per frame)
    ↓
Face Crop Extraction (with padding)
    ↓
Isotropic Resize + Padding (380x380)
    ↓
Normalization (ImageNet stats)
    ↓
EfficientNet-B7 Encoder
    ↓
Global Average Pooling
    ↓
Classification Head (Sigmoid)
    ↓
Per-Face Probability Scores
    ↓
Confidence-Based Aggregation
    ↓
Final Prediction (REAL/FAKE)
```

### Detection Strategy

1. **Face Detection Phase**
   - MTCNN processes each frame with thresholds `[0.7, 0.8, 0.8]`
   - Detects all faces in the frame (multi-face support)
   - Returns bounding boxes for each detected face

2. **Face Preprocessing Phase**
   - Crops face with 33% padding (horizontal and vertical)
   - Applies isotropic resize to 380×380 (maintains aspect ratio)
   - Pads with black pixels to square dimensions
   - Normalizes using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

3. **Classification Phase**
   - Forward pass through EfficientNet-B7 encoder
   - Global average pooling reduces spatial dimensions
   - Single-node linear classifier outputs logit
   - Sigmoid activation produces probability (0=REAL, 1=FAKE)

4. **Aggregation Phase**
   - Collects all face predictions across all frames
   - Applies **Confident Strategy** aggregation:
     - If >40% of faces score >0.8 AND >11 faces: average high-confidence fakes
     - If >90% of faces score <0.2: average low-confidence reals
     - Otherwise: simple mean of all predictions

---

## Architecture Details

### Component Breakdown

#### 1. MTCNN Face Detector

```python
MTCNN(
    margin=0,
    thresholds=[0.7, 0.8, 0.8],  # P-Net, R-Net, O-Net
    device="cuda" or "cpu"
)
```

- **Purpose**: Detect and localize faces in frames
- **Stages**: Proposal Network → Refinement Network → Output Network
- **Output**: Bounding boxes `[xmin, ymin, xmax, ymax]` for each face

#### 2. EfficientNet-B7 Encoder

```python
timm.create_model(
    "tf_efficientnet_b7.ns_jft_in1k",
    pretrained=True
)
```

**Architecture Specifications:**

- **Framework**: TIMM (PyTorch Image Models)
- **Variant**: `tf_efficientnet_b7.ns_jft_in1k` (TensorFlow-style, JFT-300M pretrained)
- **Input Resolution**: 380×380×3
- **Feature Dimension**: 2560 (final encoder output channels)
- **Parameters**: ~66M (EfficientNet-B7 backbone)

**Layer Structure:**

```text
Conv2D(3 → 64, 3×3, stride=2)       # Stem
    ↓
MBConv Blocks (7 stages)
    Stage 1: 64 → 32    (MBConv1, k3×3)
    Stage 2: 32 → 48    (MBConv6, k3×3)
    Stage 3: 48 → 80    (MBConv6, k5×5)
    Stage 4: 80 → 160   (MBConv6, k3×3)
    Stage 5: 160 → 224  (MBConv6, k5×5)
    Stage 6: 224 → 384  (MBConv6, k5×5)
    Stage 7: 384 → 640  (MBConv6, k3×3)
    ↓
Conv2D(640 → 2560, 1×1)              # Head
    ↓
Output: [B, 2560, H/32, W/32]
```

#### 3. Classification Head

```python
DeepFakeClassifier(
    encoder="tf_efficientnet_b7.ns_jft_in1k",
    dropout_rate=0.0
)
```

**Forward Pass:**

```python
x = encoder.forward_features(x)       # [B, 2560, 12, 12]
x = AdaptiveAvgPool2d((1, 1))(x)      # [B, 2560, 1, 1]
x = Flatten()(x)                       # [B, 2560]
x = Dropout(0.0)(x)                    # [B, 2560] (no dropout)
logit = Linear(2560, 1)(x)             # [B, 1]
```

**Output:**

- Single logit value (no sigmoid in forward pass)
- Sigmoid applied during inference: `prob_fake = torch.sigmoid(logit)`

### Preprocessing Pipeline

#### Isotropic Resize + Padding

```python
def _preprocess_face(img: np.ndarray, size: int = 380) -> np.ndarray:
    h, w = img.shape[:2]
    scale = size / max(w, h)  # Preserve aspect ratio
    
    # Upscale: use cubic interpolation
    # Downscale: use area-based interpolation
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=interp)
    
    # Pad to square 380x380
    h, w = img.shape[:2]
    padded_img = np.zeros((size, size, 3), dtype=np.uint8)  # Black padding
    start_w, start_h = (size - w) // 2, (size - h) // 2
    padded_img[start_h:start_h + h, start_w:start_w + w] = img
    
    return padded_img  # [380, 380, 3]
```

#### Face Extraction with Padding

```python
for bbox in batch_boxes:
    xmin, ymin, xmax, ymax = [int(b) for b in bbox]
    w, h = xmax - xmin, ymax - ymin
    
    # Add 33% padding
    p_h, p_w = h // 3, w // 3
    
    # Crop with padding (clamped to frame boundaries)
    crop = frame[
        max(ymin - p_h, 0):ymax + p_h,
        max(xmin - p_w, 0):xmax + p_w
    ]
```

### Aggregation Strategy

#### Confident Strategy Algorithm

```python
def _confident_strategy(preds: List[float], threshold: float = 0.8) -> float:
    if not preds:
        return 0.0
    
    preds = np.array(preds)
    
    # Rule 1: High-confidence fakes
    fakes = np.count_nonzero(preds > threshold)
    if fakes > len(preds) / 2.5 and fakes > 11:
        return np.mean(preds[preds > threshold])
    
    # Rule 2: High-confidence reals
    if np.count_nonzero(preds < 0.2) > 0.9 * len(preds):
        return np.mean(preds[preds < 0.2])
    
    # Rule 3: Mixed/uncertain → simple average
    return np.mean(preds)
```

**Logic Breakdown:**

1. **High-Confidence Fake Rule**:
   - Condition: `(fakes > total/2.5) AND (fakes > 11)`
   - Example: If 100 faces detected, need >40 scoring >0.8
   - Action: Average only the high-scoring predictions
   - Rationale: Confident fakes dominate the decision

2. **High-Confidence Real Rule**:
   - Condition: `>90% of faces score <0.2`
   - Action: Average only the low-scoring predictions
   - Rationale: Overwhelming real evidence

3. **Fallback Rule**:
   - Condition: Neither above rule triggered
   - Action: Simple mean of all predictions
   - Rationale: Uncertain case, use balanced average

---

## Input/Output Specifications

### Input Requirements

| Property | Value |
|----------|-------|
| **Media Type** | Video file (MP4, AVI, MOV, etc.) |
| **Frame Requirements** | Any resolution (processed per-frame) |
| **Face Requirements** | Detectable faces (MTCNN-compatible) |
| **Preprocessing** | Automatic (MTCNN + isotropic resize) |

### Output Schema

```python
VideoAnalysisResult(
    prediction: str,              # "REAL" or "FAKE"
    confidence: float,             # 0.0 to 1.0
    processing_time: float,        # Seconds
    note: Optional[str],           # Warning if no faces detected
    frame_count: int,              # Total frames in video
    frames_analyzed: int,          # Frames successfully processed
    frame_predictions: List[FramePrediction],  # Per-frame details
    metrics: Dict[str, Any],       # Additional metrics
    visualization_path: Optional[str]  # Path to visualization video
)
```

#### Frame Prediction Structure

```python
FramePrediction(
    index: int,                    # Frame number (0-indexed)
    score: float,                  # Frame-level score (max of faces)
    prediction: str                # "REAL" or "FAKE" (per-frame)
)
```

#### Metrics Dictionary

```python
metrics = {
    "total_faces_detected": int,          # Across all frames
    "average_face_score": float,          # Mean of all face predictions
    "suspicious_frames_count": int,       # Frames with score > 0.5
}
```

---

## Strengths and Limitations

### Strengths

1. **Multi-Face Support**
   - Processes all detected faces in each frame
   - Handles group videos or multi-person scenes

2. **Robust Face Detection**
   - MTCNN provides reliable face localization
   - Three-stage cascade (P-Net, R-Net, O-Net) ensures accuracy

3. **Aspect Ratio Preservation**
   - Isotropic resize prevents distortion
   - Maintains facial feature proportions

4. **Adaptive Aggregation**
   - Confident Strategy handles mixed predictions
   - Balances high-confidence and uncertain cases

5. **Strong Encoder**
   - EfficientNet-B7 trained on JFT-300M dataset
   - 2560-dimensional feature space

6. **Temporal Analysis**
   - Processes every frame (no sampling)
   - Detects manipulation across time

### Limitations

1. **No Temporal Modeling**
   - Each frame analyzed independently
   - No LSTM or temporal aggregation of features
   - Cannot detect cross-frame inconsistencies

2. **Face Detection Dependency**
   - Requires visible, detectable faces
   - Fails if MTCNN cannot detect faces
   - No fallback for face-free scenarios

3. **Computational Cost**
   - EfficientNet-B7 is computationally expensive (~66M params)
   - 380×380 input resolution increases processing time
   - All frames processed (no adaptive sampling)

4. **Aggregation Heuristics**
   - Confidence thresholds (0.8, 0.2) are hand-tuned
   - May not generalize to all deepfake types
   - Simple averaging in uncertain cases

5. **No Audio Analysis**
   - Video-only model
   - Cannot detect audio-visual desynchronization

6. **Single-Modality**
   - Does not incorporate temporal context
   - No cross-frame attention mechanism

---

## Technical Deep Dive

### EfficientNet-B7 Architecture

#### Compound Scaling

EfficientNet-B7 uses **compound scaling** to balance depth, width, and resolution:

```text
Scaling Coefficients (B7):
- Depth Coefficient (α): 2.0
- Width Coefficient (β): 1.8
- Resolution Coefficient (γ): 2.0

Formula:
depth = α^φ
width = β^φ
resolution = γ^φ

where φ = 1.5 for B7
```

**Result:**

- **Depth**: 7 stages with deeper blocks
- **Width**: 2560 final channels (vs. 1280 in B0)
- **Resolution**: 380×380 input (vs. 224×224 in B0)

#### MBConv Block

**Mobile Inverted Bottleneck Convolution (MBConv):**

```python
# MBConv6 block (expansion ratio = 6)
x_input = x  # [B, C_in, H, W]

# 1. Expansion (1×1 Conv)
x = Conv2D(C_in → 6*C_in, 1×1)(x)
x = BatchNorm()(x)
x = Swish()(x)

# 2. Depthwise Convolution (k×k)
x = DepthwiseConv2D(6*C_in, k×k, padding='same')(x)
x = BatchNorm()(x)
x = Swish()(x)

# 3. Squeeze-and-Excitation (SE)
se = GlobalAvgPool()(x)  # [B, 6*C_in]
se = Dense(C_in/4)(se)
se = Swish()(se)
se = Dense(6*C_in)(se)
se = Sigmoid()(se)
x = x * se  # Channel-wise gating

# 4. Projection (1×1 Conv)
x = Conv2D(6*C_in → C_out, 1×1)(x)
x = BatchNorm()(x)

# 5. Skip Connection (if C_in == C_out and stride == 1)
if C_in == C_out and stride == 1:
    x = x + x_input  # Residual connection
```

**Key Components:**

- **Expansion**: Increases channels by factor of 6
- **Depthwise Conv**: Spatial filtering (parameter-efficient)
- **SE Block**: Channel attention mechanism
- **Projection**: Reduce channels back to output size
- **Skip Connection**: Residual learning (when possible)

#### Stage Configuration (B7)

| Stage | Input Channels | Output Channels | Blocks | Kernel | Expansion | Stride |
|-------|----------------|-----------------|--------|--------|-----------|--------|
| 1     | 64             | 32              | 4      | 3×3    | 1         | 1      |
| 2     | 32             | 48              | 8      | 3×3    | 6         | 2      |
| 3     | 48             | 80              | 8      | 5×5    | 6         | 2      |
| 4     | 80             | 160             | 10     | 3×3    | 6         | 2      |
| 5     | 160            | 224             | 10     | 5×5    | 6         | 1      |
| 6     | 224            | 384             | 14     | 5×5    | 6         | 2      |
| 7     | 384            | 640             | 4      | 3×3    | 6         | 1      |

**Total MBConv Blocks**: 4 + 8 + 8 + 10 + 10 + 14 + 4 = **58 blocks**

### MTCNN Architecture

#### Three-Stage Cascade

**Stage 1 - Proposal Network (P-Net)**:

```python
Conv2D(3 → 10, 3×3)
MaxPool2D(2×2)
Conv2D(10 → 16, 3×3)
Conv2D(16 → 32, 3×3)

# Two heads:
# - Face classification: Conv2D(32 → 2, 1×1) → Softmax
# - Bounding box regression: Conv2D(32 → 4, 1×1)
```

- **Input**: Image pyramid (multiple scales)
- **Output**: Coarse face proposals
- **Threshold**: 0.7

**Stage 2 - Refinement Network (R-Net)**:

```python
Conv2D(3 → 28, 3×3)
MaxPool2D(3×3)
Conv2D(28 → 48, 3×3)
MaxPool2D(3×3)
Conv2D(48 → 64, 2×2)
Flatten()
Dense(576 → 128)

# Two heads:
# - Face classification: Dense(128 → 2) → Softmax
# - Bounding box regression: Dense(128 → 4)
```

- **Input**: Proposals from P-Net (resized to 24×24)
- **Output**: Refined bounding boxes
- **Threshold**: 0.8

**Stage 3 - Output Network (O-Net)**:

```python
Conv2D(3 → 32, 3×3)
MaxPool2D(3×3)
Conv2D(32 → 64, 3×3)
MaxPool2D(3×3)
Conv2D(64 → 64, 3×3)
MaxPool2D(2×2)
Conv2D(64 → 128, 2×2)
Flatten()
Dense(1152 → 256)

# Three heads:
# - Face classification: Dense(256 → 2) → Softmax
# - Bounding box regression: Dense(256 → 4)
# - Facial landmarks: Dense(256 → 10)  # 5 keypoints (unused here)
```

- **Input**: Refined proposals (resized to 48×48)
- **Output**: Final bounding boxes with landmarks
- **Threshold**: 0.8

### Visualization Generation

#### Frame-by-Frame Graph Overlay

```python
def _generate_visualization(
    media_path: str,
    frame_scores: List[float],
    total_frames: int
) -> str:
    # Read video
    cap = cv2.VideoCapture(media_path)
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Matplotlib setup
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(6, 2.5))
    
    for i in range(total_frames):
        ret, frame = cap.read()
        score = frame_scores[i]
        
        # 1. Update plot
        ax.clear()
        ax.fill_between(range(i + 1), frame_scores[:i + 1], 
                         color="#FF4136", alpha=0.4)  # Red fill
        ax.plot(range(i + 1), frame_scores[:i + 1], 
                color="#FF851B", linewidth=2)  # Orange line
        ax.axhline(y=0.5, color="white", linestyle="--", alpha=0.7)  # Threshold
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Frame Suspicion Analysis")
        
        # 2. Convert plot to image
        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plot_img_bgr = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
        
        # 3. Resize and overlay plot on frame
        plot_h, plot_w, _ = plot_img_bgr.shape
        new_plot_h = int(frame_height * 0.3)  # 30% of frame height
        new_plot_w = int(new_plot_h * (plot_w / plot_h))
        resized_plot = cv2.resize(plot_img_bgr, (new_plot_w, new_plot_h))
        
        # Overlay in top-right corner (with boundary checks)
        x_offset = frame_width - new_plot_w - 10
        y_offset = 10
        frame[y_offset:y_offset + new_plot_h, 
              x_offset:x_offset + new_plot_w] = resized_plot
        
        # 4. Add suspicion bar at bottom
        score_color = np.array([0, 0, 255]) * score + \
                      np.array([0, 255, 0]) * (1 - score)  # Red→Green gradient
        bar_height = 40
        cv2.rectangle(frame, (0, frame_height - bar_height), 
                     (frame_width, frame_height), 
                     tuple(map(int, score_color)), -1)
        cv2.putText(frame, f"Frame Suspicion: {score:.2f}", 
                   (15, frame_height - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()
    return output_path
```

**Visualization Components:**

1. **Temporal Graph**: Shows suspicion scores over time (top-right overlay)
2. **Suspicion Bar**: Color-coded bar at bottom (green=real, red=fake)
3. **Score Text**: Numerical score overlaid on bar

### Memory and Performance Considerations

#### Model Size

```python
EfficientNet-B7 Parameters:
- Encoder: ~66M parameters
- Classification Head: 2560 × 1 = 2.6K parameters
- Total: ~66M parameters

Memory Footprint (FP32):
- Model Weights: 66M × 4 bytes = 264 MB
- Activations (per image, 380×380):
  - Input: 380 × 380 × 3 = 433 KB
  - Stage 7 output: 12 × 12 × 2560 = 1.47 MB
  - Total (approx): ~100 MB per image
```

#### Computational Complexity

**FLOPs (per 380×380 image):**

```text
EfficientNet-B7: ~37 GFLOPs per forward pass

For a 30 FPS video with 1 face per frame:
- 30 frames/sec × 37 GFLOPs = 1.11 TFLOPs/sec
- On RTX 3090 (~35 TFLOPs FP32): Real-time capable
```

**Processing Time Breakdown (example):**

```text
For 300-frame video (10 sec @ 30 FPS):
- MTCNN face detection: ~2-3 sec
- EfficientNet-B7 inference: ~5-7 sec
- Visualization generation (optional): ~8-10 sec
- Total: ~15-20 sec (with visualization)
```

### Integration Example

```python
from src.config import EfficientNetB7Config
from src.ml.detectors.efficientnet_detector import EfficientNetB7Detector

# 1. Load configuration
config = EfficientNetB7Config(
    class_name="EfficientNetB7Detector",
    model_name="EFFICIENTNET-B7-V1",
    model_path="models/EfficientNet-B7-v1",
    encoder="tf_efficientnet_b7.ns_jft_in1k",
    input_size=380,
    isAudio=False,
    isImage=False,
    isVideo=True
)

# 2. Initialize detector
detector = EfficientNetB7Detector(config)
detector.load()

# 3. Run analysis
result = detector.analyze(
    media_path="video.mp4",
    generate_visualizations=True,  # Enable visualization
    video_id="abc123",
    user_id="user456"
)

# 4. Access results
print(f"Prediction: {result.prediction}")
print(f"Confidence: {result.confidence:.3f}")
print(f"Total Faces: {result.metrics['total_faces_detected']}")
print(f"Suspicious Frames: {result.metrics['suspicious_frames_count']}")

# 5. Frame-level analysis
for frame_pred in result.frame_predictions[:10]:
    print(f"Frame {frame_pred.index}: {frame_pred.score:.3f} ({frame_pred.prediction})")

# 6. Visualization path (if generated)
if result.visualization_path:
    print(f"Visualization saved: {result.visualization_path}")
```

### Configuration Parameters

```yaml
EFFICIENTNET-B7-V1:
  class_name: "EfficientNetB7Detector"
  model_path: "models/EfficientNet-B7-v1"
  encoder: "tf_efficientnet_b7.ns_jft_in1k"
  input_size: 380
  isAudio: false
  isImage: false
  isVideo: true
```

**Parameter Descriptions:**

- **encoder**: TIMM model identifier
  - `tf_efficientnet_b7.ns_jft_in1k`: TensorFlow-style EfficientNet-B7 pretrained on JFT-300M with Noisy Student
  
- **input_size**: Face crop resolution (380×380)
  - Matches EfficientNet-B7 expected input resolution
  
- **model_path**: Path to trained classifier weights
  - Contains state dict for `DeepFakeClassifier`

### Event Publishing (Progress Tracking)

```python
# Frame analysis progress (every 5 frames)
ProgressEvent(
    media_id=video_id,
    user_id=user_id,
    event="FRAME_ANALYSIS_PROGRESS",
    message="Analyzed 50/300 frames, detected 75 faces",
    data=EventData(
        model_name="EFFICIENTNET-B7-V1",
        progress=50,
        total=300,
        details={
            "phase": "frame_processing",
            "faces_detected_so_far": 75,
            "current_frame_faces": 2
        }
    )
)

# Visualization progress (every 50 frames)
ProgressEvent(
    media_id=video_id,
    user_id=user_id,
    event="FRAME_ANALYSIS_PROGRESS",
    message="Generating visualization: 150/300 frames processed",
    data=EventData(
        model_name="EFFICIENTNET-B7-V1",
        progress=150,
        total=300,
        details={"phase": "visualization"}
    )
)

# Analysis completion
ProgressEvent(
    media_id=video_id,
    user_id=user_id,
    event="ANALYSIS_COMPLETE",
    message="Analysis completed: FAKE (confidence: 0.876)",
    data=EventData(
        model_name="EFFICIENTNET-B7-V1",
        progress=300,
        total=300,
        details={
            "prediction": "FAKE",
            "confidence": 0.876,
            "processing_time": 18.5,
            "total_frames": 300,
            "faces_detected": 275
        }
    )
)
```

---

## Summary

EfficientNet-B7 (V1) is a **frame-by-frame face-based video deepfake detector** that combines the robust face detection capabilities of MTCNN with the powerful feature extraction of EfficientNet-B7. Its strength lies in multi-face support, aspect-ratio-preserving preprocessing, and an adaptive confidence-based aggregation strategy.

**Best Use Cases:**

- Videos with clear, visible faces
- Multi-person scenes
- High-resolution videos
- Scenarios requiring frame-level suspicion tracking

**Not Suitable For:**

- Face-free videos
- Heavily compressed or low-quality videos
- Real-time applications (due to computational cost)
- Audio-visual manipulation detection

The model provides comprehensive frame-by-frame predictions, temporal suspicion graphs (via visualization), and detailed metrics for downstream analysis.
