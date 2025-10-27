# SIGLIP-LSTM (V1/V3/V4)

**Model Category**: Video Analysis  
**Model Type**: Temporal Deepfake Detector  
**Versions**: V1, V3, V4  
**Primary Detection Target**: Frame-to-frame temporal inconsistencies in videos

---

## Overview

### What Is This Model?

SIGLIP-LSTM is a video deepfake detector that combines **vision transformers** with **recurrent neural networks** to analyze temporal consistency across video frames. It's designed to detect frame-to-frame inconsistencies that may indicate synthetic or manipulated video content.

### The Core Concept

The model analyzes temporal flow across video frames—how objects move, how lighting changes, and how facial expressions transition. It uses pre-trained vision features combined with sequence analysis to identify temporal inconsistencies.

### Three Versions

The model has evolved through three versions:

- **V1**: Foundational architecture with basic classifier
- **V3**: Enhanced with rolling window analysis and advanced metrics
- **V4**: Deeper classifier with dropout regularization

All versions share the same core architecture but differ in post-processing and classification layers.

---

## How It Works

### Step-by-Step Process

#### Phase 1: Frame Extraction

```text
Input Video → Extract N frames (120 by default) → Uniform temporal sampling
```

- Videos are uniformly sampled to extract exactly 120 frames
- Frames are distributed across the entire video duration
- This ensures temporal coverage without processing every frame

#### Phase 2: Feature Extraction (SigLIP Vision Transformer)

```text
Each Frame → SigLIP Processor → Vision Transformer → 768-dim Feature Vector
```

**What's Happening**:

- Each frame is processed through Google's **SigLIP** (Sigmoid Loss for Language-Image Pre-Training) model
- SigLIP is a vision transformer pre-trained on image datasets
- For each frame, a **768-dimensional feature vector** is extracted that encodes:
  - Semantic content (objects present)
  - Facial characteristics
  - Scene composition
  - Visual patterns and textures

#### Phase 3: Temporal Analysis (Bi-LSTM)

```text
Sequence of 120 Feature Vectors → Bi-LSTM (512 hidden units, 2 layers) → Temporal Features
```

**What's Happening**:

- The 120 feature vectors are treated as a **sequence**
- A **Bidirectional LSTM** (Long Short-Term Memory) processes this sequence
- **Bidirectional** means it processes the sequence both forward and backward
- The LSTM analyzes:
  - Transitions between frames
  - Temporal patterns and inconsistencies
  - Sequence-level features

**Why Bi-LSTM?**

- LSTMs are designed for sequence analysis
- Bidirectional processing captures context from both past and future frames
- Can detect long-range dependencies across frames

#### Phase 4: Classification

```text
LSTM Output → Classifier Head → Sigmoid → Confidence Score (0 to 1)
```

**V1 & V3**: Simple linear classifier (1024-dim → 1-dim → Sigmoid)  
**V4**: Deeper classifier with dropout:

```text
1024-dim → 256-dim (ReLU) → Dropout (0.5) → 1-dim → Sigmoid
```

**Output**: A single confidence score:

- **0.0 - 0.5**: Classified as REAL
- **0.5 - 1.0**: Classified as FAKE
- **Threshold**: 0.5 (default)

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
│              Frame Extraction & Sampling                     │
│         • Extract 120 frames uniformly                       │
│         • Resize to 224x224 (SigLIP input size)             │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│          SigLIP Vision Transformer (Per Frame)               │
│   google/siglip-base-patch16-224                            │
│   • Patch Embedding (16x16 patches)                         │
│   • Transformer Encoder (12 layers)                         │
│   • Output: 768-dim feature vector per frame                │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Temporal Feature Sequence (Shape: 120 x 768)        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Bidirectional LSTM                              │
│   • Hidden size: 512 units                                   │
│   • Num layers: 2                                            │
│   • Direction: Bidirectional (forward + backward)           │
│   • Output: 1024-dim vector (512 x 2 directions)            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  Classifier Head                             │
│   V1/V3: Linear(1024 → 1) → Sigmoid                         │
│   V4:    Linear(1024 → 256) → ReLU → Dropout(0.5)          │
│          → Linear(256 → 1) → Sigmoid                         │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Final Output: Confidence Score [0.0 - 1.0]          │
│             + Per-Frame Predictions                          │
│             + Temporal Metrics (V3/V4)                       │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **SigLIP Base (Frozen)**
   - Pre-trained weights from Google
   - Parameters: ~86M
   - NOT fine-tuned (used as feature extractor only)
   - Provides pre-trained visual understanding

2. **Bi-LSTM (Trainable)**
   - Parameters: ~4M
   - Trained component for temporal analysis
   - Learns sequence patterns

3. **Classifier Head (Trainable)**
   - V1/V3: ~1K parameters (single linear layer)
   - V4: ~260K parameters (two layers + dropout)

**Total Parameters**: ~90M (mostly in SigLIP, which is frozen)  
**Trainable Parameters**: ~4-5M (LSTM + classifier)

---

## Version Differences

### V1: The Foundation

**Architecture**:

- SIGLIP → LSTM → Linear classifier
- Extracts 120 frames uniformly
- Simple averaging for final prediction

**Classifier Head**:

```python
Linear(1024 → 1) → Sigmoid
```

**Characteristics**:

- Minimal hyperparameters
- Straightforward architecture
- Single-layer classifier

---

### V3: Enhanced Analytics

**What's Different**:

1. **Rolling Window Smoothing**
   - Window size: 10 frames
   - Smooths per-frame predictions
   - Formula: `smoothed[i] = mean(predictions[i-5 : i+5])`

2. **Advanced Metrics Collection**
   - Score variance
   - Temporal consistency metrics
   - Per-frame confidence tracking

3. **Enhanced Output**
   - More detailed metadata
   - Frame-level analysis
   - Temporal pattern indicators

**Classifier Head**: Same as V1

```python
Linear(1024 → 1) → Sigmoid
```

---

### V4: Regularization & Depth

**What's Different**:

1. **Deeper Classifier Head**
   - Added hidden layer
   - Non-linearity (ReLU activation)
   - Increased model capacity

2. **Dropout Regularization**
   - Dropout rate: 0.5
   - Applied during training
   - Helps prevent overfitting

**Classifier Head**:

```python
Linear(1024 → 256) → ReLU → Dropout(0.5) → Linear(256 → 1) → Sigmoid
```

**Inherits from V3**:

- Rolling window smoothing
- Advanced metrics
- Enhanced output format

---

### Version Comparison Table

| Feature | V1 | V3 | V4 |
|---------|----|----|-----|
| **Classifier Depth** | 1 layer | 1 layer | 2 layers + Dropout |
| **Classifier Parameters** | ~1K | ~1K | ~260K |
| **Rolling Window** | ❌ | ✅ (10 frames) | ✅ (10 frames) |
| **Advanced Metrics** | ❌ | ✅ | ✅ |
| **Dropout Regularization** | ❌ | ❌ | ✅ |

---

## Input Requirements

### Accepted Video Formats

- **Container**: MP4, AVI, MOV, MKV, WebM
- **Codecs**: H.264, H.265, VP9, AV1
- **Resolution**: Any (automatically resized to 224x224 for processing)
- **Duration**: Any (minimum ~2 seconds recommended)
- **FPS**: Any (frames extracted uniformly regardless of FPS)

### Input Specifications

| Parameter | Requirement | Notes |
|-----------|-------------|-------|
| **Duration** | Minimum 2 seconds | Very short videos may lack temporal context |
| **Resolution** | Any | Resized to 224x224 internally |
| **FPS** | Any | 120 frames extracted uniformly |
| **Audio** | Optional | Not used by this model |

### Preprocessing Pipeline

1. **Frame Extraction**: 120 frames uniformly sampled across video duration
2. **Resizing**: Each frame resized to 224×224 pixels (SigLIP requirement)
3. **Normalization**: Pixel values normalized to [0, 1] range
4. **Tensor Conversion**: Converted to PyTorch tensors

---

## Output Format

### JSON Response Structure

```json
{
  "prediction": "FAKE",
  "confidence": 0.9823,
  "metadata": {
    "model_version": "V4",
    "frames_analyzed": 120,
    "video_duration": 15.5,
    "per_frame_scores": [0.92, 0.94, 0.98, ...],
    "temporal_metrics": {
      "score_variance": 0.023,
      "consistency_score": 0.87
    }
  }
}
```

### Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | string | "REAL" or "FAKE" classification |
| `confidence` | float | Confidence score (0-1) |
| `model_version` | string | Version used (V1/V3/V4) |
| `frames_analyzed` | int | Number of frames processed |
| `video_duration` | float | Video length in seconds |
| `per_frame_scores` | array | Individual frame predictions (V3/V4) |
| `temporal_metrics` | object | Temporal analysis metrics (V3/V4) |

### Temporal Metrics (V3/V4 only)

| Metric | Description |
|--------|-------------|
| `score_variance` | Variance of per-frame predictions (lower = more consistent) |
| `consistency_score` | Measure of temporal smoothness (0-1, higher = more consistent) |

---

## Architecture Strengths & Limitations

### Strengths

1. **Pre-trained Visual Features**:
   - Leverages SigLIP's pre-trained knowledge
   - Strong visual understanding without domain-specific training
   - Efficient feature extraction

2. **Temporal Analysis**:
   - Bidirectional LSTM captures sequence patterns
   - Analyzes frame-to-frame transitions
   - Detects long-range temporal dependencies

3. **Version Flexibility**:
   - V1: Simple, fast baseline
   - V3: Enhanced analytics and smoothing
   - V4: Improved generalization with deeper classifier

4. **Efficient Architecture**:
   - Frozen SigLIP reduces training requirements
   - Only LSTM and classifier are trainable
   - Relatively small trainable parameter count (~4-5M)

### Limitations

1. **Fixed Frame Count**:
   - Always processes exactly 120 frames
   - Cannot adapt to variable-length requirements
   - Very short videos may have limited temporal context

2. **SigLIP Dependency**:
   - Requires SigLIP model to be available
   - Feature quality depends on SigLIP's capabilities
   - Frozen weights cannot adapt to specific use cases

3. **Sequence Processing**:
   - LSTM processes entire sequence at once
   - Memory requirements scale with sequence length
   - Cannot process streaming video in real-time

4. **Version-Specific Tradeoffs**:
   - V1: Simpler classifier may have limited capacity
   - V3/V4: Rolling window adds processing overhead
   - V4: Deeper classifier adds parameters and computation

---

## Technical Deep Dive

### SigLIP Feature Extraction

**Architecture**: google/siglip-base-patch16-224

**Processing Pipeline**:

1. **Patch Embedding**:
   - Input: 224×224 image
   - Divided into 16×16 patches
   - Creates 196 patches (14×14 grid)
   - Each patch embedded to 768 dimensions

2. **Transformer Encoder**:
   - 12 transformer layers
   - Multi-head self-attention (12 heads)
   - Feed-forward network per layer
   - Layer normalization

3. **Feature Output**:
   - Global pooled representation: 768 dimensions
   - Used as input to LSTM

### LSTM Temporal Processing

**Architecture Configuration**:

```python
nn.LSTM(
    input_size=768,      # SigLIP feature dimension
    hidden_size=512,     # LSTM hidden state size
    num_layers=2,        # Two stacked LSTM layers
    bidirectional=True,  # Process forward and backward
    batch_first=True     # Batch dimension first
)
```

**Processing Flow**:

1. **Forward Pass**:
   - Processes frames from t=0 to t=119
   - Hidden state: 512 dimensions
   - Captures forward temporal dependencies

2. **Backward Pass**:
   - Processes frames from t=119 to t=0
   - Hidden state: 512 dimensions
   - Captures backward temporal dependencies

3. **Concatenation**:
   - Forward + Backward = 1024 dimensions
   - Provides full temporal context
   - Final hidden state used for classification

### Version-Specific Implementations

#### V1 Classifier

```python
class ClassifierV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, lstm_out):
        # lstm_out shape: (batch, 1024)
        logits = self.fc(lstm_out)
        return self.sigmoid(logits)
```

#### V4 Classifier (with Dropout)

```python
class ClassifierV4(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, lstm_out):
        # lstm_out shape: (batch, 1024)
        x = self.fc1(lstm_out)
        x = self.relu(x)
        x = self.dropout(x)  # Only active during training
        logits = self.fc2(x)
        return self.sigmoid(logits)
```

### Rolling Window Smoothing (V3/V4)

**Purpose**: Reduce noise in per-frame predictions

**Implementation**:

```python
def rolling_window_smooth(predictions, window_size=10):
    """
    Smooth predictions using a rolling average window
    
    Args:
        predictions: Array of per-frame predictions (120,)
        window_size: Size of smoothing window (default: 10)
    
    Returns:
        Smoothed predictions array
    """
    smoothed = []
    half_window = window_size // 2
    
    for i in range(len(predictions)):
        start = max(0, i - half_window)
        end = min(len(predictions), i + half_window + 1)
        window = predictions[start:end]
        smoothed.append(np.mean(window))
    
    return np.array(smoothed)
```

---

## Integration Example

```python
import torch
from pathlib import Path

# Load the model (specify version)
model = load_model('SIGLIP-LSTM', version='V4')
model.eval()

# Analyze a video file
video_path = Path("video.mp4")

# Run inference
result = model.predict(video_path)

# Access results
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Frames Analyzed: {result['metadata']['frames_analyzed']}")

# Access version-specific features (V3/V4)
if 'temporal_metrics' in result['metadata']:
    metrics = result['metadata']['temporal_metrics']
    print(f"Score Variance: {metrics['score_variance']:.4f}")
    print(f"Consistency: {metrics['consistency_score']:.2%}")
```
