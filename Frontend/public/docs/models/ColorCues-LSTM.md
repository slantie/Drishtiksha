# ColorCues-LSTM Model

## Overview

**ColorCues-LSTM** is a deepfake detection model that analyzes color patterns and temporal artifacts in video content. The model combines color-based feature extraction with Long Short-Term Memory (LSTM) networks to capture both spatial color anomalies and temporal inconsistencies.

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Model Type** | Hybrid CNN-LSTM Architecture |
| **Primary Focus** | Color pattern inconsistencies and temporal artifacts |
| **Input Type** | Video sequences (25 frames) |
| **Output** | Binary classification (Real/Fake) with confidence score |
| **Framework** | PyTorch |

---

## How It Works

ColorCues-LSTM operates through a 4-phase pipeline that analyzes color characteristics across temporal sequences:

### Phase 1: Color Space Transformation & Feature Extraction

The model extracts frames from the input video and converts them from RGB to multiple color spaces:

1. **Multi-Color Space Analysis**:
   - RGB (Red, Green, Blue) - Original color information
   - HSV (Hue, Saturation, Value) - Perceptual color attributes
   - YCbCr (Luminance, Chrominance) - Broadcast-standard color encoding
   - Lab (Lightness, a\*, b\*) - Perceptually uniform color space

2. **Facial Region Detection**:
   - Uses face detection to isolate facial regions
   - Applies landmark detection (68 facial keypoints)
   - Segments face into regions: forehead, cheeks, nose, chin, eyes, mouth

3. **Color Pattern Extraction**:
   - Computes color histograms for each facial region
   - Calculates color moments (mean, variance, skewness)
   - Extracts color gradients and transitions
   - Analyzes skin tone consistency

### Phase 2: Spatial Color Consistency Analysis

The extracted color features are analyzed for spatial consistency within individual frames:

1. **Inter-Region Analysis**:
   - Compares color distributions between adjacent facial regions
   - Detects unnatural color boundaries
   - Identifies inconsistent lighting patterns
   - Analyzes shadow consistency

2. **CNN Feature Encoding**:
   - Processes color features through convolutional layers
   - Extracts high-level color pattern representations
   - Generates spatial feature maps (256 dimensions)
   - Captures subtle color artifacts

### Phase 3: Temporal Sequence Analysis with LSTM

The LSTM network analyzes how color patterns evolve across frames:

1. **Temporal Color Tracking**:
   - Monitors color consistency across consecutive frames
   - Detects sudden color shifts
   - Identifies periodic patterns indicative of frame-by-frame manipulation
   - Analyzes color flow and temporal gradients

2. **LSTM Processing**:
   - Bidirectional LSTM layers (256 hidden units)
   - Processes 25-frame sequences
   - Captures long-range temporal dependencies

### Phase 4: Classification & Confidence Scoring

Final decision making combines spatial and temporal insights:

1. **Feature Fusion**:
   - Concatenates spatial CNN features with temporal LSTM outputs
   - Applies fully connected layers for decision making
   - Generates probability score (0-1 range)

2. **Threshold-Based Classification**:
   - Scores > 0.5: Classified as "Fake"
   - Scores ≤ 0.5: Classified as "Real"
   - Returns confidence level with prediction

---

## Architecture Details

### Model Architecture Diagram

```text
Input Video (25 frames, 224x224)
         |
         v
┌─────────────────────────────────────────────────────┐
│  Multi-Color Space Converter                       │
│  RGB → HSV, YCbCr, Lab                            │
└─────────────────────────────────────────────────────┘
         |
         v
┌─────────────────────────────────────────────────────┐
│  Face Detection & Landmark Extraction               │
│  68-point facial landmarks                         │
└─────────────────────────────────────────────────────┘
         |
         v
┌─────────────────────────────────────────────────────┐
│  Color Feature Extraction                          │
│  - Color histograms (per region)                   │
│  - Color moments (mean, var, skew)                 │
│  - Color gradients                                 │
│  - Skin tone analysis                              │
└─────────────────────────────────────────────────────┘
         |
         v
┌─────────────────────────────────────────────────────┐
│  Spatial CNN Encoder                               │
│  Conv2D(64) → ReLU → MaxPool                       │
│  Conv2D(128) → ReLU → MaxPool                      │
│  Conv2D(256) → ReLU → AdaptiveAvgPool              │
│  Output: 256-dim feature vector per frame          │
└─────────────────────────────────────────────────────┘
         |
         v
┌─────────────────────────────────────────────────────┐
│  Temporal LSTM Analyzer                            │
│  Bidirectional LSTM(256 hidden units)              │
│  Processes sequence of 25 frame features           │
│  Output: 512-dim temporal encoding                 │
└─────────────────────────────────────────────────────┘
         |
         v
┌─────────────────────────────────────────────────────┐
│  Fusion & Classification Head                      │
│  FC(512) → ReLU → Dropout(0.3)                     │
│  FC(256) → ReLU → Dropout(0.3)                     │
│  FC(1) → Sigmoid                                   │
│  Output: Probability [0, 1]                        │
└─────────────────────────────────────────────────────┘
         |
         v
    Prediction
(Real/Fake + Confidence)
```

### Layer-by-Layer Breakdown

| Layer Type | Configuration | Parameters | Output Shape |
|------------|--------------|------------|--------------|
| **Input** | 25 frames × 224×224×3 RGB | - | (25, 224, 224, 3) |
| **Color Converter** | Multi-space transformation | - | (25, 224, 224, 12) |
| **Conv2D-1** | 64 filters, 3×3, stride=1, ReLU | 1,792 | (25, 224, 224, 64) |
| **MaxPool-1** | 2×2, stride=2 | - | (25, 112, 112, 64) |
| **Conv2D-2** | 128 filters, 3×3, stride=1, ReLU | 73,856 | (25, 112, 112, 128) |
| **MaxPool-2** | 2×2, stride=2 | - | (25, 56, 56, 128) |
| **Conv2D-3** | 256 filters, 3×3, stride=1, ReLU | 295,168 | (25, 56, 56, 256) |
| **AdaptiveAvgPool** | Output size: 1×1 | - | (25, 256) |
| **Bidirectional LSTM** | 256 units, 2 layers | 1,576,960 | (25, 512) |
| **Temporal Pooling** | Max + Mean pooling | - | (512) |
| **FC-1** | 512 → 256, ReLU, Dropout(0.3) | 131,328 | (256) |
| **FC-2** | 256 → 128, ReLU, Dropout(0.3) | 32,896 | (128) |
| **Output** | 128 → 1, Sigmoid | 129 | (1) |
| **Total Parameters** | | **2,112,129** | |

---

## Input Requirements

### Video Specifications

| Parameter | Requirement | Notes |
|-----------|-------------|-------|
| **Format** | MP4, AVI, MOV, MKV | Common video formats supported |
| **Resolution** | Minimum 224×224 pixels | Upscaled if smaller, downscaled if larger |
| **Frame Rate** | Any (recommended: 25-30 fps) | Model samples 25 frames from video |
| **Duration** | Minimum 1 second | Need sufficient frames for temporal analysis |
| **Color Depth** | 24-bit RGB or higher | Grayscale not recommended |
| **Face Visibility** | Clear frontal face required | Partial occlusions reduce accuracy |
| **Lighting** | Normal to good lighting | Very low light reduces color accuracy |

### Preprocessing Steps

Before feeding video to the model, the following preprocessing is applied:

1. **Frame Extraction**: Sample 25 evenly-spaced frames from the video
2. **Face Detection**: Detect and crop face region with 30% margin
3. **Resize**: Scale face region to 224×224 pixels
4. **Normalization**:
   - RGB values normalized to [0, 1]
   - Channel-wise mean subtraction: [0.485, 0.456, 0.406]
   - Channel-wise std division: [0.229, 0.224, 0.225]
5. **Color Space Conversion**: Convert to HSV, YCbCr, and Lab color spaces

---

## Output Format

### Prediction Response

```json
{
  "model_name": "ColorCues-LSTM-v1",
  "prediction": "fake",
  "confidence": 0.87,
  "raw_score": 0.8734,
  "threshold": 0.5,
  "metadata": {
    "frames_analyzed": 25,
    "face_detected": true,
    "color_anomaly_score": 0.76,
    "temporal_consistency_score": 0.91
  }
}
```

### Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `model_name` | string | Identifier for the model version |
| `prediction` | string | "real" or "fake" classification |
| `confidence` | float | Confidence score (0-1) indicating prediction certainty |
| `raw_score` | float | Raw model output before thresholding |
| `threshold` | float | Decision threshold (default: 0.5) |
| `frames_analyzed` | int | Number of frames processed |
| `face_detected` | boolean | Whether a face was successfully detected |
| `color_anomaly_score` | float | Spatial color inconsistency score |
| `temporal_consistency_score` | float | Temporal pattern regularity score |

---

## Architecture Strengths & Limitations

### Strengths

1. **Multi-Color Space Analysis**:
   - Analyzes color patterns across RGB, HSV, YCbCr, and Lab color spaces
   - Each color space captures different types of artifacts
   - More comprehensive than single color space approaches

2. **Temporal Pattern Detection**:
   - Bidirectional LSTM captures temporal dependencies
   - Detects frame-by-frame inconsistencies
   - Analyzes color flow across sequences

3. **Interpretable Features**:
   - Color-based features are more interpretable than deep learned features
   - Spatial and temporal scores help understand detections
   - Region-specific analysis possible

### Limitations

1. **Color Dependency**:
   - Requires color information to function effectively
   - Reduced effectiveness on grayscale content
   - Performance may degrade in very low-light conditions

2. **Face Detection Dependency**:
   - Requires clear face detection
   - Struggles with partial occlusions or extreme angles
   - Profile views reduce effectiveness

3. **Fixed Frame Count**:
   - Always processes exactly 25 frames
   - Cannot adapt to variable-length sequences
   - Very short videos may not provide enough temporal context

4. **Single Face Analysis**:
   - Analyzes only one (primary/largest) face per video
   - Cannot detect manipulations on background subjects
   - No multi-face support

---

## Technical Deep Dive

### Color Space Selection Rationale

The model uses four distinct color spaces because each captures different manipulation artifacts:

1. **RGB (Red, Green, Blue)**:
   - Native color representation
   - Captures direct pixel-level inconsistencies
   - Sensitive to lighting variations

2. **HSV (Hue, Saturation, Value)**:
   - Separates color (hue) from brightness (value)
   - More robust to lighting changes
   - Hue channel can reveal skin tone inconsistencies

3. **YCbCr (Luminance, Chrominance Blue, Chrominance Red)**:
   - Separates luminance from chrominance
   - Standard color space used in video compression
   - Chrominance channels may show manipulation artifacts

4. **Lab (Lightness, a\*, b\*)**:
   - Perceptually uniform color space
   - Better matches human color perception
   - a\* (green-red) and b\* (blue-yellow) axes isolate chromatic information

### Color Feature Extraction

The model extracts several types of color features:

1. **Color Histograms**:
   - Computed for each facial region separately
   - Captures color distribution characteristics per channel and color space

2. **Color Moments**:
   - Mean: Average color value per channel/region
   - Variance: Color spread/uniformity
   - Skewness: Distribution asymmetry

3. **Inter-Region Color Gradients**:
   - Color transitions between adjacent facial regions
   - Analyzes gradient smoothness and continuity

4. **Skin Tone Analysis**:
   - Extracts skin color patterns in YCbCr space
   - Analyzes consistency across frames

### LSTM Temporal Modeling

The Bidirectional LSTM architecture provides:

1. **Forward Temporal Dependencies**:
   - Learns how color patterns evolve forward in time
   - Detects frames that violate predicted color progression

2. **Backward Temporal Dependencies**:
   - Analyzes color patterns in reverse temporal order
   - Identifies past frames that don't align with later context

3. **Bidirectional Context**:
   - Combines forward and backward passes for full temporal context
   - Enables detection of isolated anomalous frames

---

## Integration Example

```python
import torch
from pathlib import Path

# Load the ColorCues-LSTM model
model = load_model('ColorCues-LSTM-v1')
model.eval()

# Analyze a video file
video_path = Path("video.mp4")

# Run inference
result = model.predict(video_path)

# Access prediction results
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```
