# EyeBlink-CNN-LSTM Model

**Model Category**: Video Analysis  
**Model Type**: Blink Pattern Deepfake Detector  
**Version**: V1  
**Primary Detection Target**: Unnatural eye blink patterns in videos

---

## Overview

### What Is This Model?

EyeBlink-CNN-LSTM is a video deepfake detector that analyzes eye blink patterns to identify synthetic or manipulated facial videos. The model combines facial landmark detection, eye region extraction, and temporal sequence analysis using a CNN-LSTM architecture.

### The Core Concept

Natural human eye blinks follow specific temporal patterns and physical characteristics. The model detects anomalies in these blink patterns that may indicate deepfake manipulation, as many generative models struggle to accurately reproduce natural blinking behavior.

### Architecture Overview

The model uses a two-stage approach:

1. **Blink Detection**: Locates and extracts eye blink sequences from video frames
2. **Classification**: Analyzes extracted blink sequences using Xception CNN + LSTM

---

## How It Works

### Step-by-Step Process

#### Phase 1: Face and Eye Landmark Detection

```text
Input Video → Frame Extraction → Face Detection → Eye Landmark Detection
```

**Components**:

- **Face Detector**: dlib frontal face detector
- **Landmark Predictor**: dlib 68-point facial landmark predictor
- **Eye Regions**: Extracts left and right eye coordinates (points 36-41 for left, 42-47 for right)

**Purpose**: Locate eyes in each frame to calculate Eye Aspect Ratio (EAR)

#### Phase 2: Eye Aspect Ratio (EAR) Calculation

```text
Eye Landmarks → EAR Calculation → Blink Detection
```

**EAR Formula**:

```text
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

Where:
- p1, p2, p3, p4, p5, p6 are the 6 landmarks for each eye
- || || represents Euclidean distance
```

**Characteristics**:

- **Open Eye**: EAR ≈ 0.2-0.4
- **Closed Eye (Blink)**: EAR < threshold (default: 0.2)
- Calculated for both eyes and averaged

#### Phase 3: Blink Sequence Extraction

```text
EAR Values → Blink Detection → Eye Region Cropping → Sequence Formation
```

**Blink Detection Logic**:

1. Monitor EAR values across consecutive frames
2. Detect when EAR drops below threshold (eye closing)
3. Track consecutive frames with low EAR
4. Extract eye region crops from detected blink frames
5. Form sequences of configurable length

**Parameters**:

- **Blink Threshold**: EAR < 0.2 (configurable)
- **Consecutive Frames**: Minimum frames required to confirm blink
- **Sequence Length**: Number of frames per analysis sequence
- **Frame Skip**: Analyzes every 10th frame for efficiency

#### Phase 4: CNN Feature Extraction

```text
Blink Frame Sequences → Xception CNN → Feature Vectors
```

**Xception Architecture**:

- Pre-trained on ImageNet
- Frozen weights (not fine-tuned)
- Extracts 2048-dimensional features per frame
- Global Average Pooling applied to spatial dimensions

**Processing**:

```text
Input: (batch_size, sequence_length, 3, height, width)
CNN Processing: Each frame → 2048-dim feature vector
Output: (batch_size, sequence_length, 2048)
```

#### Phase 5: Temporal LSTM Analysis

```text
Feature Sequences → LSTM → Classification
```

**LSTM Configuration**:

- **Type**: Unidirectional LSTM
- **Hidden Size**: 128 units
- **Layers**: 1 layer
- **Direction**: Forward only
- **Output**: Uses last hidden state

**Classification Head**:

```text
LSTM Output (128-dim) → Dropout(0.3) → Linear(128 → 1) → Sigmoid
```

**Output**: Probability score (0-1)

- Values closer to 1: More likely REAL
- Values closer to 0: More likely FAKE

**Note**: The output is inverted during post-processing to align with standard convention (1 = FAKE)

---

## Architecture Details

### Model Structure

```text
┌─────────────────────────────────────────────────────────────┐
│                      Input Video                             │
│                   (any duration, any FPS)                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Face & Eye Landmark Detection (dlib)                 │
│   • Frontal face detector                                    │
│   • 68-point shape predictor                                 │
│   • Extract eye landmarks (points 36-47)                     │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Eye Aspect Ratio (EAR) Calculation              │
│   • Calculate EAR for left eye                               │
│   • Calculate EAR for right eye                              │
│   • Average both eyes                                        │
│   • Detect blinks (EAR < threshold)                          │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Blink Sequence Extraction                       │
│   • Identify consecutive low-EAR frames                      │
│   • Crop eye regions from blink frames                       │
│   • Form sequences of specified length                       │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│          Xception CNN (Frozen, Pre-trained)                  │
│   • Process each frame independently                         │
│   • Extract 2048-dim features per frame                      │
│   • Global Average Pooling                                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Temporal Feature Sequence                            │
│   Shape: (batch_size, sequence_length, 2048)                │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Unidirectional LSTM                             │
│   • Hidden size: 128 units                                   │
│   • Num layers: 1                                            │
│   • Direction: Forward                                       │
│   • Output: Last hidden state (128-dim)                      │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  Classification Head                         │
│   Dropout(0.3) → Linear(128 → 1) → Sigmoid                  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Blink Sequence Score [0.0 - 1.0]                    │
│   (Inverted to: 0=Real, 1=Fake)                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **dlib Face Detection**
   - Pre-trained frontal face detector
   - 68-point shape predictor for facial landmarks
   - Eye landmarks: points 36-41 (left), 42-47 (right)

2. **Xception CNN (Frozen)**
   - Pre-trained on ImageNet
   - Parameters: ~22M
   - NOT fine-tuned (feature extractor only)
   - Output features: 2048 dimensions

3. **LSTM (Trainable)**
   - Parameters: ~1.1M
   - Analyzes temporal patterns in blink sequences
   - Unidirectional (forward-only processing)

4. **Classifier Head (Trainable)**
   - Parameters: ~129
   - Simple linear projection with dropout

**Total Parameters**: ~23M (mostly in frozen Xception)  
**Trainable Parameters**: ~1.1M (LSTM + classifier)

---

## Input Requirements

### Video Specifications

| Parameter | Requirement | Notes |
|-----------|-------------|-------|
| **Format** | MP4, AVI, MOV, MKV, WebM | Common video formats supported |
| **Resolution** | Any (recommended: 480p+) | Higher resolution helps face detection |
| **Duration** | Minimum 2-3 seconds | Need sufficient frames for blink detection |
| **FPS** | Any (recommended: 24-30) | Standard video frame rates |
| **Face Visibility** | **Required** | Clear frontal face with visible eyes |
| **Lighting** | Good lighting required | Poor lighting affects landmark detection |

### Critical Requirements

⚠️ **Face Detection Dependency**: This model requires:

- Clear frontal or near-frontal face view
- Both eyes visible and unobstructed
- Sufficient lighting for landmark detection
- Minimal face occlusion

Videos without detectable faces or eyes will return low-confidence fallback results.

### Preprocessing Pipeline

1. **Frame Sampling**: Analyzes every 10th frame for efficiency
2. **Face Detection**: dlib frontal face detector
3. **Landmark Detection**: 68-point facial shape prediction
4. **EAR Calculation**: Compute Eye Aspect Ratio per frame
5. **Blink Detection**: Identify frames where EAR < threshold
6. **Eye Cropping**: Extract eye region from blink frames
7. **Sequence Formation**: Group consecutive blink frames into sequences
8. **Image Transformation**:
   - Convert to tensor
   - Resize to model input size
   - Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

---

## Output Format

### JSON Response Structure

```json
{
  "prediction": "FAKE",
  "confidence": 0.73,
  "processing_time": 12.4,
  "note": null,
  "frame_count": 450,
  "frames_analyzed": 23,
  "frame_predictions": [
    {
      "index": 0,
      "score": 0.68,
      "prediction": "FAKE"
    },
    {
      "index": 1,
      "score": 0.82,
      "prediction": "FAKE"
    }
  ],
  "metrics": {
    "blink_sequences_found": 23,
    "suspicious_blink_sequences": 15,
    "average_blink_suspicion": 0.73
  },
  "visualization_path": "/path/to/visualization.mp4"
}
```

### Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | string | "REAL" or "FAKE" classification |
| `confidence` | float | Confidence score (0-1) |
| `processing_time` | float | Total analysis time in seconds |
| `note` | string \| null | Warning messages (e.g., insufficient blinks) |
| `frame_count` | int | Total frames in input video |
| `frames_analyzed` | int | Number of blink sequences analyzed |
| `frame_predictions` | array | Per-sequence predictions |
| `metrics` | object | Analysis metrics |
| `visualization_path` | string \| null | Path to visualization video (if generated) |

### Metrics Object

| Metric | Description |
|--------|-------------|
| `blink_sequences_found` | Total number of blink sequences detected |
| `suspicious_blink_sequences` | Count of sequences scored as suspicious (≥ 0.5) |
| `average_blink_suspicion` | Mean score across all sequences |

### Fallback Behavior

If insufficient blinks are detected (less than sequence length):

```json
{
  "prediction": "REAL",
  "confidence": 0.51,
  "note": "Insufficient blinks detected (X frames found). Result is a low-confidence fallback.",
  "frames_analyzed": 0
}
```

---

## Architecture Strengths & Limitations

### Strengths

1. **Targeted Detection Approach**:
   - Focuses on specific physiological artifact (eye blinks)
   - Exploits known weakness in deepfake generation
   - Interpretable detection mechanism

2. **Pre-trained Visual Features**:
   - Leverages Xception's ImageNet knowledge
   - No need to train feature extractor from scratch
   - Efficient use of parameters

3. **Temporal Analysis**:
   - LSTM captures blink sequence patterns
   - Analyzes transitions and flow
   - Detects unnatural blink timing

4. **Efficient Processing**:
   - Analyzes only blink frames (not entire video)
   - Frame skipping reduces computation
   - Focused analysis on relevant regions

### Limitations

1. **Face Detection Dependency**:
   - **Critical**: Requires clear frontal face with visible eyes
   - Fails on profile views or occluded faces
   - Poor lighting degrades performance
   - Cannot analyze videos without detectable faces

2. **Blink Requirement**:
   - Needs sufficient blinks in video
   - Very short videos may lack blinks
   - Some videos naturally have few blinks
   - Cannot analyze if no blinks detected

3. **Single Subject Analysis**:
   - Analyzes only the primary (largest) detected face
   - Cannot process multiple subjects simultaneously
   - Background faces are ignored

4. **Fixed Sequence Processing**:
   - Requires sequences of specific length
   - Cannot adapt to variable blink patterns
   - May miss isolated anomalous blinks

5. **Landmark Detection Sensitivity**:
   - Accuracy depends on landmark precision
   - Errors in landmark detection propagate to EAR calculation
   - May struggle with unusual face angles or expressions

6. **Limited to Facial Videos**:
   - Cannot analyze non-facial content
   - Not applicable to non-human subjects
   - Requires human eye characteristics

---

## Technical Deep Dive

### Eye Aspect Ratio (EAR) Calculation

**Mathematical Definition**:

The EAR is calculated using the Euclidean distances between specific eye landmarks:

```python
def calculate_ear(eye_landmarks):
    """
    Calculate Eye Aspect Ratio for a single eye
    
    Args:
        eye_landmarks: Array of 6 (x,y) coordinates
                      [p1, p2, p3, p4, p5, p6]
    
    Returns:
        float: Eye Aspect Ratio value
    """
    # Vertical eye landmark distances
    vertical_1 = euclidean_distance(eye_landmarks[1], eye_landmarks[5])
    vertical_2 = euclidean_distance(eye_landmarks[2], eye_landmarks[4])
    
    # Horizontal eye landmark distance
    horizontal = euclidean_distance(eye_landmarks[0], eye_landmarks[3])
    
    # EAR formula
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    
    return ear
```

**Interpretation**:

- **High EAR (0.25-0.4)**: Eye is open
- **Low EAR (<0.2)**: Eye is closed (blink detected)
- **Zero division protection**: Returns 0.0 if horizontal distance is 0

**Blink Detection**:

```python
# Averaged across both eyes
left_ear = calculate_ear(left_eye_landmarks)
right_ear = calculate_ear(right_eye_landmarks)
average_ear = (left_ear + right_ear) / 2.0

# Blink detected if below threshold
is_blinking = average_ear < blink_threshold  # default: 0.2
```

### Xception CNN Architecture

**Model**: `timm.create_model('xception', pretrained=True, features_only=True)`

**Architecture Characteristics**:

- Depthwise separable convolutions
- Entry flow → Middle flow → Exit flow
- Residual connections
- Output features: 2048 dimensions

**Feature Extraction**:

```python
# Input shape: (batch * seq_len, 3, H, W)
features = xception_model(images)[-1]  # Get last feature map

# Global Average Pooling
# Shape: (batch * seq_len, 2048, H', W') → (batch * seq_len, 2048)
pooled_features = features.mean(dim=[-1, -2])
```

### LSTM Temporal Processing

**Configuration**:

```python
nn.LSTM(
    input_size=2048,     # Xception feature dimension
    hidden_size=128,     # LSTM hidden state size
    num_layers=1,        # Single LSTM layer
    bidirectional=False, # Unidirectional (forward only)
    batch_first=True     # Batch dimension first
)
```

**Processing Flow**:

```python
# Input: (batch_size, sequence_length, 2048)
lstm_out, (h_n, c_n) = lstm(sequence_features)

# Use last hidden state for classification
# h_n shape: (num_layers * num_directions, batch_size, hidden_size)
last_hidden = h_n[-1]  # Shape: (batch_size, 128)
```

### Classification Head

**Architecture**:

```python
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 1)
    
    def forward(self, lstm_hidden):
        # lstm_hidden shape: (batch_size, 128)
        x = self.dropout(lstm_hidden)
        logits = self.fc(x)
        
        # Note: Sigmoid applied separately
        return logits
```

**Training vs Inference**:

- **Training**: Uses BCEWithLogitsLoss (sigmoid included in loss)
- **Inference**: Applies sigmoid to get probabilities

### Blink Sequence Formation

**Algorithm**:

```python
def extract_blink_sequences(video_path, threshold=0.2, min_consecutive=3):
    """
    Extract blink sequences from video
    
    Args:
        video_path: Path to input video
        threshold: EAR threshold for blink detection
        min_consecutive: Minimum consecutive blink frames
    
    Returns:
        List of cropped eye region images
    """
    all_blink_frames = []
    blink_frame_buffer = []
    consecutive_blink_count = 0
    
    for frame in video_frames:
        # Detect face and calculate EAR
        ear = calculate_average_ear(frame)
        
        if ear < threshold:
            # Eye is closed
            consecutive_blink_count += 1
            eye_crop = extract_eye_region(frame)
            blink_frame_buffer.append(eye_crop)
        else:
            # Eye is open - check if we had a complete blink
            if consecutive_blink_count >= min_consecutive:
                # Valid blink sequence detected
                all_blink_frames.extend(blink_frame_buffer)
            
            # Reset buffers
            consecutive_blink_count = 0
            blink_frame_buffer = []
    
    # Handle final blink if video ends during blink
    if consecutive_blink_count >= min_consecutive:
        all_blink_frames.extend(blink_frame_buffer)
    
    return all_blink_frames
```

---

## Integration Example

```python
import torch
from pathlib import Path

# Load the model
model = load_model('EyeBlink-CNN-LSTM-V1')
model.eval()

# Analyze a video file
video_path = Path("video.mp4")

# Run analysis (with visualization generation)
result = model.analyze(
    media_path=str(video_path),
    generate_visualizations=True
)

# Access results
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Blink Sequences Found: {result['metrics']['blink_sequences_found']}")
print(f"Suspicious Sequences: {result['metrics']['suspicious_blink_sequences']}")

# Check for warnings
if result['note']:
    print(f"Note: {result['note']}")

# Access visualization if generated
if result['visualization_path']:
    print(f"Visualization saved to: {result['visualization_path']}")
```

### Handling Insufficient Blinks

```python
result = model.analyze(media_path=video_path)

if result['frames_analyzed'] == 0:
    print("Warning: No blink sequences detected")
    print(f"Reason: {result['note']}")
    # Consider fallback analysis or different model
else:
    # Normal processing
    confidence = result['confidence']
    prediction = result['prediction']
```
