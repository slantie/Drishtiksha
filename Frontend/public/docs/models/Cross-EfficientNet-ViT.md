# Cross-EfficientNet-ViT Model

**Model Category**: Video & Image Analysis  
**Model Type**: Multimodal Deepfake Detector  
**Version**: V1  
**Primary Detection Target**: Face manipulation in images and videos

---

## Overview

### What Is This Model?

Cross-EfficientNet-ViT is a hybrid deepfake detector that combines EfficientNet convolutional neural networks with Vision Transformers (ViT) in a novel multi-scale cross-attention architecture. The model processes images at two different scales simultaneously and uses cross-attention mechanisms to enable information flow between the scales, creating a richer representation for deepfake detection.

### The Core Concept

The model operates on two parallel processing streams:

- **Small-scale (SM) branch**: Processes fine-grained details using smaller patches
- **Large-scale (LG) branch**: Captures broader context using larger patches

These branches communicate through **Cross-Transformer** layers that allow each scale to attend to information from the other scale, combining local details with global context for more accurate detection.

### Key Innovation

**Multi-Scale Cross-Attention Architecture**:

- Two separate EfficientNet feature extractors at different network depths
- Dual Vision Transformer branches processing different patch sizes
- Cross-attention mechanism enabling bidirectional information flow
- Combined classification from both scales

---

## How It Works

### Step-by-Step Process

#### Phase 1: Face Detection & Extraction

```text
Input Image/Video → MTCNN Face Detector → Face Crops
```

**Face Detection** (using MTCNN):

```python
face_detector = MTCNN(
    keep_all=True,           # Detect all faces in image
    device=device,
    select_largest=False,
    min_face_size=20,        # Minimum face size in pixels
    thresholds=[0.6, 0.7, 0.7]  # P-Net, R-Net, O-Net thresholds
)
```

**Process**:

- Detects all faces in frame/image
- Extracts face crops with bounding boxes
- Converts to RGB format
- Returns list of face crops

**Handling No Faces**:

- If no faces detected → classify as REAL (confidence 0.99)
- Continues to next frame/image

#### Phase 2: Preprocessing & Normalization

```text
Face Crops → Resize → Normalize → Tensor
```

**Transformation Pipeline**:

```python
transforms.Compose([
    Resize(height=380, width=380),  # Standard size
    Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet normalization
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])
```

**Normalization**:

- Uses ImageNet statistics
- Standardizes input distribution
- Helps with transfer learning

#### Phase 3: Dual EfficientNet Feature Extraction

```text
Input Image → Small-Scale EfficientNet + Large-Scale EfficientNet → Multi-Level Features
```

**Small-Scale Branch** (fine details):

```python
efficient_net_sm = EfficientNet.from_name('efficientnet-b0')
efficient_net_sm.delete_blocks(16)  # Extract at block 16 (deep)
features_sm = efficient_net_sm.extract_features_at_block(img, block=16)
```

**Large-Scale Branch** (broad context):

```python
efficient_net_lg = EfficientNet.from_name('efficientnet-b0')
efficient_net_lg.delete_blocks(1)   # Extract at block 1 (shallow)
features_lg = efficient_net_lg.extract_features_at_block(img, block=1)
```

**Why Different Blocks?**

| Branch | Block | Features | Resolution | Purpose |
|--------|-------|----------|------------|---------|
| **Small** | 16 (deep) | High-level semantic | Lower spatial | Fine-grained patterns |
| **Large** | 1 (shallow) | Low-level visual | Higher spatial | Broad context |

#### Phase 4: Patch Embedding

```text
CNN Features → Rearrange into Patches → Linear Projection → Add Position Embeddings
```

**Small-Scale Patch Embedding**:

```python
# Rearrange feature map into patches
patches_sm = rearrange(
    features_sm,
    'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
    p1=sm_patch_size,  # e.g., 4
    p2=sm_patch_size
)

# Linear projection to embedding dimension
embeddings_sm = Linear(patch_dim, sm_dim)(patches_sm)  # e.g., 64-dim

# Add [CLS] token and position embeddings
cls_token_sm = Parameter(torch.randn(1, 1, sm_dim))
embeddings_sm = concat([cls_token_sm, embeddings_sm], dim=1)
embeddings_sm += pos_embedding_sm
```

**Large-Scale Patch Embedding** (similar process with different dimensions):

```python
embeddings_lg = Linear(patch_dim, lg_dim)(patches_lg)  # e.g., 128-dim
embeddings_lg = concat([cls_token_lg, embeddings_lg], dim=1)
embeddings_lg += pos_embedding_lg
```

**Position Embeddings**:

- Learnable parameters
- Encode spatial position of patches
- Enable ViT to understand spatial relationships

#### Phase 5: Multi-Scale Transformer Encoding

```text
Dual Embeddings → Multi-Scale Encoder (Transformers + Cross-Attention) → Enhanced Tokens
```

**For Each Layer** (repeated `depth` times):

**Step 1 - Self-Attention Within Scales**:

```python
# Small-scale self-attention
sm_tokens = Transformer(sm_tokens)

# Large-scale self-attention  
lg_tokens = Transformer(lg_tokens)
```

**Step 2 - Cross-Attention Between Scales**:

```python
sm_tokens, lg_tokens = CrossTransformer(sm_tokens, lg_tokens)
```

**Self-Attention Mechanism**:

For each token in a scale:

```python
Q = Linear(tokens)  # Query
K = Linear(tokens)  # Key
V = Linear(tokens)  # Value

attention_weights = softmax(Q @ K.T / sqrt(dim_head))
output = attention_weights @ V
```

**Cross-Attention Mechanism**:

Small branch attends to large:

```python
Q_sm = Linear(sm_cls_token)           # Query from small
K_lg = Linear(lg_patch_tokens)        # Keys from large patches
V_lg = Linear(lg_patch_tokens)        # Values from large patches

attention_sm_to_lg = softmax(Q_sm @ K_lg.T / sqrt(dim_head))
sm_cls_enhanced = attention_sm_to_lg @ V_lg
```

Large branch attends to small (symmetric):

```python
Q_lg = Linear(lg_cls_token)
K_sm = Linear(sm_patch_tokens)
V_sm = Linear(sm_patch_tokens)

attention_lg_to_sm = softmax(Q_lg @ K_sm.T / sqrt(dim_head))
lg_cls_enhanced = attention_lg_to_sm @ V_sm
```

**Bidirectional Information Flow**:

- Small-scale learns from large-scale context
- Large-scale learns from small-scale details
- Both scales become richer representations

#### Phase 6: Classification Heads

```text
Enhanced Tokens → Extract [CLS] Tokens → Dual MLPs → Combine Logits
```

**Small-Scale Classification**:

```python
sm_cls = sm_tokens[:, 0]  # Extract [CLS] token
sm_logits = LayerNorm(sm_cls)
sm_logits = Linear(sm_dim → 1)(sm_logits)
```

**Large-Scale Classification**:

```python
lg_cls = lg_tokens[:, 0]  # Extract [CLS] token
lg_logits = LayerNorm(lg_cls)
lg_logits = Linear(lg_dim → 1)(lg_logits)
```

**Combined Output**:

```python
final_logits = sm_logits + lg_logits
probability = sigmoid(final_logits)
```

#### Phase 7: Aggregation (Video/Multi-Face)

```text
Per-Face Predictions → Aggregation Strategy → Final Classification
```

**Aggregation Logic**:

```python
def aggregate_predictions(face_scores):
    # If any face is highly suspicious, classify as FAKE
    for score in face_scores:
        if score > 0.65:
            return score
    
    # Otherwise, average the scores
    return mean(face_scores)
```

**Per-Frame** (video):

```python
frame_score = max(face_scores_in_frame)
```

**Final Video Score**:

```python
all_frame_face_scores = [all face scores from all frames]
final_score = aggregate_predictions(all_frame_face_scores)
```

---

## Architecture Details

### Model Structure

```text
┌─────────────────────────────────────────────────────────────┐
│                  Input Image/Video Frame                     │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│           Face Detection (MTCNN)                             │
│   • Detect all faces in frame                                │
│   • Extract face crops                                       │
│   • Handle no-face cases (→ REAL)                            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│           Preprocessing & Normalization                      │
│   • Resize to 380×380                                        │
│   • Normalize with ImageNet stats                            │
│   • Convert to tensor                                        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ├──────────────────┬──────────────────┐
                        ▼                  ▼                  │
┌────────────────────────────┐  ┌────────────────────────────┐│
│   Small-Scale Branch       │  │   Large-Scale Branch       ││
│   (Fine Details)           │  │   (Broad Context)          ││
│                            │  │                            ││
│ EfficientNet-B0 (Block 16) │  │ EfficientNet-B0 (Block 1)  ││
│ → Deep features            │  │ → Shallow features         ││
│ → Smaller patches (4×4)    │  │ → Larger patches (16×16)   ││
│ → sm_dim (64-dim)          │  │ → lg_dim (128-dim)         ││
└────────────┬───────────────┘  └───────────┬────────────────┘│
             │                              │                 │
             ▼                              ▼                 │
┌────────────────────────────┐  ┌────────────────────────────┐│
│   Patch Embedding (SM)     │  │   Patch Embedding (LG)     ││
│   • Rearrange to patches   │  │   • Rearrange to patches   ││
│   • Linear projection      │  │   • Linear projection      ││
│   • Add [CLS] token        │  │   • Add [CLS] token        ││
│   • Add position embedding │  │   • Add position embedding ││
└────────────┬───────────────┘  └───────────┬────────────────┘│
             │                              │                 │
             └──────────┬───────────────────┘                 │
                        ▼                                     │
┌─────────────────────────────────────────────────────────────┐
│         Multi-Scale Encoder (depth=N layers)                │
│                                                              │
│  For each layer:                                             │
│  ┌────────────────────────────────────────────────────┐    │
│  │ 1. Small-Scale Transformer (Self-Attention)        │    │
│  │    • Multi-head self-attention                     │    │
│  │    • Feed-forward network                          │    │
│  │                                                     │    │
│  │ 2. Large-Scale Transformer (Self-Attention)        │    │
│  │    • Multi-head self-attention                     │    │
│  │    • Feed-forward network                          │    │
│  │                                                     │    │
│  │ 3. Cross-Transformer (Cross-Attention)             │    │
│  │    • SM [CLS] attends to LG patches                │    │
│  │    • LG [CLS] attends to SM patches                │    │
│  │    • Bidirectional information exchange            │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  → Enhanced SM tokens, Enhanced LG tokens                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Classification Heads                                 │
│                                                              │
│  Small-Scale MLP:                                            │
│  SM [CLS] → LayerNorm → Linear(sm_dim → 1) → sm_logits     │
│                                                              │
│  Large-Scale MLP:                                            │
│  LG [CLS] → LayerNorm → Linear(lg_dim → 1) → lg_logits     │
│                                                              │
│  Combined: final_logits = sm_logits + lg_logits             │
│  Probability: sigmoid(final_logits)                          │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Aggregation (if multiple faces/frames)               │
│   • Per-face predictions                                     │
│   • High-score detection (>0.65 → suspicious)                │
│   • Average aggregation as fallback                          │
│   → Final prediction: REAL or FAKE                           │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **EfficientNet Backbones** (2 instances)
   - EfficientNet-B0 architecture
   - Trainable parameters
   - Extract features at different depths
   - Small branch: ~5.3M params, Large branch: ~0.1M params

2. **Patch Embedders** (2 instances)
   - Rearrange CNN features into patches
   - Linear projection to embedding space
   - Learnable [CLS] tokens
   - Learnable position embeddings

3. **Multi-Scale Transformer Encoder**
   - Small-scale Transformer: depth layers
   - Large-scale Transformer: depth layers
   - Cross-Transformer: bidirectional attention
   - Each layer has self-attention + cross-attention

4. **MLP Classification Heads** (2 instances)
   - LayerNorm + Linear layers
   - Separate outputs from each scale
   - Combined via addition

**Total Parameters**: ~10-15M (varies with config)  
**Trainable Parameters**: All parameters trainable

### Cross-Attention Mechanism

**Cross-Transformer Layer**:

```python
# Extract class tokens and patch tokens
sm_cls, sm_patches = sm_tokens[:, :1], sm_tokens[:, 1:]
lg_cls, lg_patches = lg_tokens[:, :1], lg_tokens[:, 1:]

# Small attends to large
sm_cls_enhanced = Attention(
    query=sm_cls,
    key=lg_patches,
    value=lg_patches,
    include_self=True  # Also attend to own patches
)

# Large attends to small
lg_cls_enhanced = Attention(
    query=lg_cls,
    key=sm_patches,
    value=sm_patches,
    include_self=True
)

# Update class tokens
sm_cls = sm_cls_enhanced + sm_cls  # Residual connection
lg_cls = lg_cls_enhanced + lg_cls

# Reassemble tokens
sm_tokens = concat([sm_cls, sm_patches], dim=1)
lg_tokens = concat([lg_cls, lg_patches], dim=1)
```

**Why Cross-Attention?**

- Enables information sharing between scales
- Small-scale can query large-scale for context
- Large-scale can query small-scale for details
- More powerful than independent processing
- Creates multi-scale representations

---

## Input Requirements

### Image/Video Specifications

| Parameter | Requirement | Notes |
|-----------|-------------|-------|
| **Format** | Images: JPG, PNG, etc. | Common image formats |
|  | Videos: MP4, AVI, MKV, etc. | Common video formats |
| **Resolution** | Any | Internally resized to 380×380 |
| **Aspect Ratio** | Any | Resized without preserving ratio |
| **Faces** | Optional | No-face detection → REAL classification |
| **Face Size** | Minimum 20 pixels | MTCNN minimum face size |

### Configuration Parameters

**Model Architecture**:

- **image_size**: Input image size (e.g., 380)
- **sm_dim**: Small-scale embedding dimension (e.g., 64)
- **lg_dim**: Large-scale embedding dimension (e.g., 128)
- **sm_patch_size**: Small-scale patch size (e.g., 4)
- **lg_patch_size**: Large-scale patch size (e.g., 16)
- **depth**: Number of multi-scale encoder layers (e.g., 4)
- **sm_enc_depth**: Small-scale transformer depth (e.g., 2)
- **lg_enc_depth**: Large-scale transformer depth (e.g., 2)
- **cross_attn_depth**: Cross-attention depth (e.g., 2)
- **dropout**: Dropout rate (e.g., 0.1)

**Face Detection**:

- **min_face_size**: Minimum face size in pixels (20)
- **thresholds**: MTCNN thresholds [0.6, 0.7, 0.7]

**Video Processing**:

- **num_frames**: Number of frames to sample (e.g., 30)

---

## Output Format

### Image Analysis Response

```json
{
  "prediction": "FAKE",
  "confidence": 0.84,
  "processing_time": 1.2,
  "dimensions": {
    "width": 1920,
    "height": 1080
  },
  "note": null
}
```

### Video Analysis Response

```json
{
  "prediction": "FAKE",
  "confidence": 0.76,
  "processing_time": 18.5,
  "frames_analyzed": 30,
  "frame_predictions": [
    {"index": 0, "score": 0.72, "prediction": "FAKE"},
    {"index": 1, "score": 0.68, "prediction": "FAKE"},
    {"index": 2, "score": 0.81, "prediction": "FAKE"},
    ...
  ],
  "metrics": {
    "final_average_score": 0.76
  },
  "note": null
}
```

### Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | string | "REAL" or "FAKE" classification |
| `confidence` | float | Confidence score (0-1) |
| `processing_time` | float | Total analysis time in seconds |
| `dimensions` | object | Image/video dimensions (image only) |
| `frames_analyzed` | int | Number of frames processed (video only) |
| `frame_predictions` | array | Per-frame predictions (video only) |
| `metrics` | object | Additional metrics |
| `note` | string | Optional note (e.g., "No face detected") |

---

## Architecture Strengths & Limitations

### Strengths

1. **Multi-Scale Architecture**:
   - Processes information at two different scales simultaneously
   - Captures both fine details and broad context
   - More comprehensive than single-scale approaches
   - Hierarchical representation learning

2. **Cross-Attention Mechanism**:
   - Bidirectional information flow between scales
   - Scales enhance each other's representations
   - More powerful than independent processing
   - Novel architecture for deepfake detection

3. **EfficientNet + ViT Hybrid**:
   - Combines CNN inductive biases with Transformer expressiveness
   - EfficientNet provides strong feature extraction
   - ViT captures long-range dependencies
   - Best of both worlds

4. **Face-Focused Analysis**:
   - MTCNN provides accurate face detection
   - Focuses computation on relevant regions
   - Can handle multiple faces
   - Reduces background noise

5. **Flexible Aggregation**:
   - Sophisticated aggregation strategy
   - High-score detection for suspicious faces
   - Robust to partial manipulations
   - Works with variable number of faces

6. **Multi-Modal Support**:
   - Works with both images and videos
   - Unified architecture for both modalities
   - Frame sampling for efficient video processing

### Limitations

1. **Face Detection Dependency**:
   - Requires face detection to work
   - No-face images always classified as REAL
   - May miss non-face manipulations
   - Sensitive to face detector quality

2. **Computational Complexity**:
   - Dual EfficientNet backbones
   - Multiple Transformer layers
   - Cross-attention adds overhead
   - Slower than single-scale models

3. **Fixed Patch Sizes**:
   - Patch sizes are hyperparameters
   - May not be optimal for all image sizes
   - No adaptive patching strategy
   - Requires careful tuning

4. **Memory Requirements**:
   - Two parallel processing streams
   - Attention mechanisms scale quadratically
   - Large intermediate representations
   - GPU memory intensive

5. **Aggregation Heuristic**:
   - Simple threshold-based logic (>0.65)
   - May not generalize to all scenarios
   - Fixed threshold not adaptive
   - Could benefit from learned aggregation

6. **No Temporal Modeling** (video):
   - Frames processed independently
   - No sequential dependencies
   - Missing temporal patterns
   - Cannot detect temporal inconsistencies

---

## Technical Deep Dive

### EfficientNet Architecture

**What Is EfficientNet?**

EfficientNet is a family of CNNs that scale network depth, width, and resolution with a compound coefficient:

```text
depth: d = α^φ
width: w = β^φ
resolution: r = γ^φ

where α·β²·γ² ≈ 2 and φ is the compound coefficient
```

**EfficientNet-B0** (baseline):

- 5.3M parameters
- Input: 224×224 (we use 380×380)
- 16 MBConv blocks
- Efficient mobile inverted bottleneck convolutions

**Why Different Blocks?**

- **Block 1** (shallow): Low-level features (edges, textures)
- **Block 16** (deep): High-level semantic features (faces, objects)

By extracting at different depths, we get multi-level representations.

### Vision Transformer (ViT) Components

**Patch Embedding**:

Converts 2D image to 1D sequence of tokens:

```python
# From (B, C, H, W) to (B, N, D)
patches = rearrange(
    image,
    'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
    p1=patch_size,
    p2=patch_size
)
embeddings = Linear(patch_size² * channels, embedding_dim)(patches)
```

**[CLS] Token**:

- Special learnable token prepended to sequence
- Aggregates information from all patches
- Used for final classification

**Position Embeddings**:

- Learnable embeddings added to patch embeddings
- Encode spatial position
- Enable attention to understand location

**Self-Attention**:

For each token, compute:

```python
Q = W_q @ token  # Query: "What am I looking for?"
K = W_k @ token  # Key: "What do I contain?"
V = W_v @ token  # Value: "What information do I have?"

attention = softmax(Q @ K.T / sqrt(d_k))
output = attention @ V
```

### Cross-Attention Mathematics

**Asymmetric Attention**:

Small-scale class token attends to large-scale patches:

```python
Q_sm = W_q^sm @ cls_sm        # (1, dim_sm)
K_lg = W_k^lg @ patches_lg    # (N_lg, dim_lg)
V_lg = W_v^lg @ patches_lg    # (N_lg, dim_lg)

# Project to common dimension
Q_sm_proj = W_proj_sm(Q_sm)   # (1, dim_common)
K_lg_proj = W_proj_lg(K_lg)   # (N_lg, dim_common)
V_lg_proj = W_proj_lg(V_lg)   # (N_lg, dim_common)

attention_weights = softmax(Q_sm_proj @ K_lg_proj.T / sqrt(d))
cls_sm_enhanced = attention_weights @ V_lg_proj
```

**Bidirectional Flow**:

- SM → LG: Small learns context from large
- LG → SM: Large learns details from small
- Both scales become richer

### Aggregation Strategy Analysis

**Why Threshold-Based?**

```python
if any(score > 0.65):
    return max_score  # One suspicious face → FAKE
else:
    return mean(scores)  # All faces normal → average
```

**Rationale**:

- Deepfakes often leave some frames/faces intact
- One highly suspicious face is strong evidence
- Averaging could dilute strong signals
- Conservative approach: better false positives than false negatives

**Threshold Choice** (0.65):

- Above majority threshold (0.5)
- High confidence required for immediate classification
- Balances sensitivity and specificity

---

## Integration Example

```python
import torch
from pathlib import Path

# Load the model
model = load_model('CROSS-EFFICIENT-VIT-V1')

# Analyze an image
image_path = Path("suspicious_image.jpg")
result = model.analyze(media_path=str(image_path))

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Processing Time: {result['processing_time']:.2f}s")

if result.get('note'):
    print(f"Note: {result['note']}")

# Analyze a video
video_path = Path("suspicious_video.mp4")
result = model.analyze(media_path=str(video_path))

print(f"\nVideo Analysis:")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Frames Analyzed: {result['frames_analyzed']}")

# Analyze per-frame predictions
suspicious_frames = [
    f for f in result['frame_predictions']
    if f['score'] > 0.7
]
print(f"Suspicious frames: {len(suspicious_frames)}")
```

### Batch Processing

```python
from pathlib import Path

# Process multiple images
image_dir = Path("images/")
results = []

for image_path in image_dir.glob("*.jpg"):
    result = model.analyze(media_path=str(image_path))
    results.append({
        'filename': image_path.name,
        'prediction': result['prediction'],
        'confidence': result['confidence']
    })

# Find most suspicious images
suspicious = sorted(
    results,
    key=lambda x: x['confidence'] if x['prediction'] == 'FAKE' else 0,
    reverse=True
)

print("Top 5 suspicious images:")
for item in suspicious[:5]:
    print(f"  {item['filename']}: {item['confidence']:.2%}")
```
