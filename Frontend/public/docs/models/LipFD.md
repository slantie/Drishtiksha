# LipFD Model (Lips Are Lying)

**Model Category**: Audio-Visual Analysis  
**Model Type**: Multimodal Deepfake Detector  
**Version**: V1  
**Primary Detection Target**: Lip-sync manipulation in videos (audio-visual inconsistencies)

---

## Overview

### What Is This Model?

LipFD (Lips Are Lying) is an audio-visual deepfake detector that analyzes the synchronization between lip movements and audio content in videos. The model combines visual analysis of facial regions with audio spectral features to identify inconsistencies indicative of deepfake manipulation, particularly lip-sync attacks and audio substitution.

### The Core Concept

The model processes video as **audio-visual windows**, where each window contains:

- **Visual**: A sequence of video frames showing facial regions at multiple scales
- **Audio**: Mel-spectrogram representation of the corresponding audio segment

By analyzing these paired modalities together, the model can detect misalignments between what is being said (audio) and how the lips are moving (visual).

### Key Innovation

LipFD uses a **Region-Aware Attention Mechanism** that:

- Analyzes facial regions at three different scales (full frame, face crop, lip crop)
- Extracts global features from combined audio-visual input using CLIP
- Applies attention weights to focus on the most discriminative regions
- Combines multi-scale regional features for final classification

---

## How It Works

### Step-by-Step Process

#### Phase 1: Audio Extraction

```text
Input Video → Extract Audio Track → Save as WAV
```

**Audio Extraction**:

- Uses MoviePy to extract audio from video file
- Saves as 16-bit PCM WAV format
- Sampling rate: 16kHz
- **Requirement**: Video must have an audio track

**Error Handling**:

```python
if clip.audio is None:
    raise MediaProcessingError("Video requires audio track for LipFD analysis")
```

#### Phase 2: Mel-Spectrogram Generation

```text
Audio WAV → Librosa Loading → Mel-Spectrogram → Power to dB → PNG Image
```

**Spectrogram Processing**:

1. **Load Audio**:

   ```python
   data, sr = librosa.load(audio_path, sr=16000)
   ```

2. **Compute Mel-Spectrogram**:

   ```python
   mel_spec = librosa.feature.melspectrogram(y=data, sr=sr)
   mel_db = librosa.power_to_db(mel_spec, ref=np.min)
   ```

3. **Save as Image**:

   ```python
   plt.imsave(spec_path, mel_db, cmap='viridis')
   mel_img = (plt.imread(spec_path) * 255).astype(np.uint8)
   ```

**Output**: 2D image representing audio frequency content over time

#### Phase 3: Video Window Extraction

```text
Video Frames → Sample Window Start Positions → Extract Frame Sequences
```

**Window Sampling Strategy**:

- **window_len**: Number of frames per window (e.g., 5 frames)
- **n_extract**: Number of windows to sample from video (e.g., 10 windows)
- **Sampling**: Linearly spaced positions across video duration

**Example** (100 frame video, 5-frame windows, 10 samples):

```python
indices = np.linspace(0, 100-5-1, 10, dtype=int)
# Samples at: [0, 10, 21, 31, 42, 52, 63, 73, 84, 94]
```

**Per Window**:

- Extract consecutive frames starting from sampled position
- Resize each frame to 500×500 pixels
- Convert BGR to RGB color space

#### Phase 4: Audio-Visual Alignment

```text
Video Window Frames + Corresponding Mel-Spectrogram Segment → Combined Image
```

**Temporal Alignment**:

```python
mapping = mel_img.shape[1] / frame_count  # Mel width to video frame ratio
begin = int(start_idx * mapping)
end = int((start_idx + window_len) * mapping)
sub_mel = mel_img[:, begin:end]  # Extract corresponding audio segment
```

**Combined Image Construction**:

1. **Concatenate Video Frames Horizontally**:

   ```python
   frames_concat = np.concatenate(frames_resized, axis=1)
   # Shape: (500, 500*window_len, 3)
   ```

2. **Resize Mel-Spectrogram Segment**:

   ```python
   sub_mel_resized = cv2.resize(sub_mel, (500*window_len, 500))
   # Shape: (500, 500*window_len, 3)
   ```

3. **Stack Vertically**:

   ```python
   combined_img = np.concatenate((sub_mel_rgb, frames_concat), axis=0)
   # Shape: (1000, 500*window_len, 3)
   # Top half: Mel-spectrogram
   # Bottom half: Video frames
   ```

**Example** (5-frame window):

```text
┌──────────────────────────────────────────────────────┐
│        Mel-Spectrogram (500×2500, viridis)          │ ← Audio
├──────────────────────────────────────────────────────┤
│ Frame1 │ Frame2 │ Frame3 │ Frame4 │ Frame5          │ ← Visual
│ (500×  │ (500×  │ (500×  │ (500×  │ (500×           │
│  500)  │  500)  │  500)  │  500)  │  500)           │
└──────────────────────────────────────────────────────┘
Total: 1000×2500 pixels
```

#### Phase 5: Multi-Scale Crop Extraction

```text
Combined Image → Full Frame Crops + Face Crops + Lip Crops
```

**Three Scale Levels**:

**1. Full Frame (1.0× scale)**:

- Extract bottom half (frames): 5 crops of 500×500
- Each crop contains one full frame
- Resize to 224×224

**2. Face Crop (0.65× scale)**:

- Crop face region from each full frame
- Crop indices: [28:196, 28:196] (168×168 from 224×224)
- Captures facial features
- Resize to 224×224

**3. Lip Crop (0.45× scale)**:

- Crop lip region from each face crop
- Crop indices: [61:163, 61:163] (102×102 from 224×224)
- Focuses on mouth/lip area
- Resize to 224×224

**Result**: 3 sets of 5 crops each (15 crops total per window)

#### Phase 6: Global Feature Extraction (CLIP)

```text
Combined Image → Conv 5×5 (1120→224 resize) → CLIP Encoder → Global Features
```

**CLIP Processing**:

1. **Resize Full Combined Image**:

   ```python
   img_resized = transforms.Resize((1120, 1120))(combined_img_tensor)
   ```

2. **Strided Convolution**:

   ```python
   conv1 = nn.Conv2d(3, 3, kernel_size=5, stride=5)
   # (1120, 1120) → (224, 224)
   ```

3. **CLIP Image Encoder**:

   ```python
   features = clip_encoder.encode_image(img_resized)
   # Output: 768-dimensional feature vector
   ```

**CLIP Role**:

- Pre-trained vision-language model
- Extracts semantic features from audio-visual combination
- Provides global context for region-aware attention

#### Phase 7: Region-Aware Attention Network

```text
Multi-Scale Crops + Global Features → ResNet-50 Backbone → Attention Weighting → Classification
```

**Per-Frame Processing** (for each of 5 frames in window):

**For Each Scale** (full, face, lip):

1. **Feature Extraction** (ResNet-50):

   ```python
   conv1 → bn1 → relu → maxpool →
   layer1 → layer2 → layer3 → layer4 →
   avgpool → flatten
   # Output: 2048-dimensional features
   ```

2. **Concatenate with Global Features**:

   ```python
   regional_feature = torch.cat([resnet_features, clip_features], dim=1)
   # Shape: (2048 + 768) = 2816 dimensions
   ```

3. **Compute Attention Weight**:

   ```python
   weight = sigmoid(Linear(2816 → 1)(regional_feature))
   # Single scalar weight for this scale
   ```

**Attention-Weighted Aggregation**:

```python
# For each frame, across 3 scales
features = [full_feature, face_feature, lip_feature]
weights = [weight_full, weight_face, weight_lip]

# Softmax normalize weights
weights_normalized = softmax(weights)

# Weighted sum
frame_feature = sum(features[i] * weights_normalized[i] for i in range(3))
```

**Across Frames**:

```python
# Average features across all 5 frames
window_feature = mean([frame_1_feature, ..., frame_5_feature])
```

**Final Classification**:

```python
output = Linear(2816 → 1)(window_feature)
score = sigmoid(output)  # Probability of FAKE
```

#### Phase 8: Window Score Aggregation

```text
Per-Window Scores → Average → Final Prediction
```

**Aggregation**:

```python
window_scores = [score_1, score_2, ..., score_N]
final_score = mean(window_scores)

if final_score >= 0.5:
    prediction = "FAKE"
    confidence = final_score
else:
    prediction = "REAL"
    confidence = 1 - final_score
```

---

## Architecture Details

### Model Structure

```text
┌─────────────────────────────────────────────────────────────┐
│                      Input Video File                        │
│                (must contain audio track)                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Audio Extraction (MoviePy)                      │
│   • Extract audio track                                      │
│   • Save as 16kHz WAV (pcm_s16le)                           │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Mel-Spectrogram Generation                      │
│   • Load audio with librosa (sr=16kHz)                       │
│   • Compute mel-spectrogram                                  │
│   • Convert to dB scale                                      │
│   • Save as PNG image (viridis colormap)                     │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Video Window Sampling                           │
│   • Sample n_extract window start positions                  │
│   • Extract window_len consecutive frames per window         │
│   • Resize frames to 500×500 RGB                             │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Audio-Visual Alignment                          │
│   • Map video window to mel-spectrogram segment              │
│   • Resize mel segment to 500×(500*window_len)              │
│   • Concatenate frames horizontally                          │
│   • Stack mel (top) + frames (bottom) vertically             │
│   → Combined image: (1000, 500*window_len, 3)               │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Multi-Scale Crop Extraction                     │
│   For each frame in window:                                  │
│   • Full frame crop (1.0× scale): 500×500 → 224×224         │
│   • Face crop (0.65× scale): 168×168 → 224×224              │
│   • Lip crop (0.45× scale): 102×102 → 224×224               │
│   → 3 scales × window_len frames = 3 × 5 = 15 crops        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Global Feature Extraction (CLIP)                     │
│   • Resize combined image to 1120×1120                       │
│   • Conv2D(3→3, k=5, s=5): 1120→224 downsampling           │
│   • CLIP image encoder (ViT or ResNet)                       │
│   • Output: 768-dimensional global feature                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Region-Aware Attention Network                       │
│   For each frame, for each scale:                            │
│   • ResNet-50 feature extraction (2048-dim)                  │
│   • Concat with global features (2048+768=2816)              │
│   • Compute attention weight: sigmoid(Linear(2816→1))       │
│                                                              │
│   Per frame aggregation:                                     │
│   • Softmax-normalize attention weights across scales        │
│   • Weighted sum of regional features                        │
│                                                              │
│   Across frames:                                             │
│   • Average features across all frames in window             │
│                                                              │
│   Final classification:                                      │
│   • Linear(2816 → 1) → Sigmoid                               │
│   • Output: Window FAKE probability                          │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Window Score Aggregation                        │
│   • Collect scores from all windows                          │
│   • Average to get final score                               │
│   • Apply threshold (0.5) for prediction                     │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Final Prediction: REAL or FAKE                       │
│         + Confidence + Per-Window Scores                     │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **CLIP Encoder** (Pre-trained, frozen)
   - Vision transformer or ResNet-based
   - Extracts 768-dim global features
   - Captures semantic audio-visual content

2. **ResNet-50 Backbone** (Trainable)
   - 50-layer residual network
   - Bottleneck architecture [3, 4, 6, 3]
   - ~25M parameters
   - Extracts 2048-dim regional features

3. **Attention Module** (Trainable)
   - Linear(2816 → 1) + Sigmoid
   - ~2.8K parameters per scale
   - Learns importance of each region

4. **Classifier** (Trainable)
   - Linear(2816 → 1)
   - ~2.8K parameters
   - Binary classification head

**Total Parameters**: ~25M (mostly ResNet-50)  
**Trainable Parameters**: ~25M (ResNet + attention + classifier)  
**Non-trainable**: CLIP encoder (frozen)

### Region-Aware Attention Mechanism

**Attention Weight Computation**:

For each scale s and frame f:

```python
regional_features[s][f] = ResNet50(crop[s][f])  # 2048-dim
combined[s][f] = concat(regional_features[s][f], global_features)  # 2816-dim
weight[s][f] = sigmoid(Linear(combined[s][f]))  # scalar
```

**Softmax Normalization** (across scales for frame f):

```python
weights_normalized[f] = softmax([weight[0][f], weight[1][f], weight[2][f]])
```

**Weighted Aggregation**:

```python
frame_output[f] = sum(combined[s][f] * weights_normalized[f][s] for s in scales)
window_output = mean(frame_output[f] for f in frames)
```

**Why Region-Aware Attention?**

- Different regions (full, face, lip) contain different information
- Attention learns to focus on most discriminative regions
- Adaptive weighting based on content
- More robust than fixed-weight averaging

---

## Input Requirements

### Video Specifications

| Parameter | Requirement | Notes |
|-----------|-------------|-------|
| **Format** | MP4, AVI, MKV, MOV, etc. | Common video formats supported |
| **Audio Track** | **Required** | Video must contain audio (no silent videos) |
| **Duration** | Minimum = window_len frames | Must be longer than single window |
| **Frame Rate** | Any | Frames sampled at regular intervals |
| **Resolution** | Any | Frames resized to 500×500 internally |
| **Aspect Ratio** | Any | Resized without preserving aspect ratio |

### Configuration Parameters

**Window Sampling**:

- **window_len**: Number of frames per window (e.g., 5)
- **n_extract**: Number of windows to sample from video (e.g., 10)

**Model Architecture**:

- **arch**: CLIP model architecture (e.g., "ViT-B/32", "RN50")

**Audio Processing**:

- **sampling_rate**: 16000 Hz (fixed)
- **codec**: pcm_s16le (16-bit PCM WAV)

### Preprocessing Requirements

**Video Must**:

- Contain visible faces (for face/lip crops to be meaningful)
- Have synchronized audio track
- Be longer than window_len frames

**Optimal Conditions**:

- Clear facial visibility
- Minimal occlusion of face/mouth
- Good audio quality
- Sufficient duration for multiple window samples

---

## Output Format

### JSON Response Structure

```json
{
  "prediction": "FAKE",
  "confidence": 0.76,
  "processing_time": 18.3,
  "frames_analyzed": 10,
  "frame_predictions": [
    {"index": 0, "score": 0.72, "prediction": "FAKE"},
    {"index": 1, "score": 0.68, "prediction": "FAKE"},
    {"index": 2, "score": 0.81, "prediction": "FAKE"},
    {"index": 3, "score": 0.79, "prediction": "FAKE"},
    {"index": 4, "score": 0.74, "prediction": "FAKE"},
    {"index": 5, "score": 0.77, "prediction": "FAKE"},
    {"index": 6, "score": 0.82, "prediction": "FAKE"},
    {"index": 7, "score": 0.71, "prediction": "FAKE"},
    {"index": 8, "score": 0.75, "prediction": "FAKE"},
    {"index": 9, "score": 0.80, "prediction": "FAKE"}
  ],
  "metrics": {
    "final_average_score": 0.76,
    "window_scores": [0.72, 0.68, 0.81, 0.79, 0.74, 0.77, 0.82, 0.71, 0.75, 0.80]
  }
}
```

### Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | string | "REAL" or "FAKE" classification |
| `confidence` | float | Confidence score (0-1) |
| `processing_time` | float | Total analysis time in seconds |
| `frames_analyzed` | int | Number of windows analyzed |
| `frame_predictions` | array | Per-window predictions and scores |
| `metrics` | object | Additional metrics and scores |

### Frame Predictions Array

Each element contains:

- **index**: Window index (0 to n_extract-1)
- **score**: Window FAKE probability (0-1)
- **prediction**: "REAL" or "FAKE" based on score threshold

### Metrics Object

- **final_average_score**: Mean of all window scores
- **window_scores**: Raw scores for each window

---

## Architecture Strengths & Limitations

### Strengths

1. **Multimodal Analysis**:
   - Combines visual and audio information
   - Detects audio-visual inconsistencies
   - More robust than single-modality approaches
   - Specifically designed for lip-sync detection

2. **Multi-Scale Regional Analysis**:
   - Analyzes facial features at three scales
   - Full frame provides context
   - Face crop focuses on facial features
   - Lip crop targets mouth movements
   - Hierarchical information capture

3. **Region-Aware Attention**:
   - Learns to weight regions adaptively
   - Focuses on most discriminative scales
   - More flexible than fixed weighting
   - Improves interpretability

4. **CLIP Global Features**:
   - Leverages pre-trained vision-language model
   - Semantic understanding of audio-visual content
   - Strong feature extraction without training
   - Captures high-level patterns

5. **Window-Based Sampling**:
   - Can analyze videos of any length
   - Multiple samples provide robustness
   - Temporal coverage across video
   - Averages out per-window noise

6. **ResNet-50 Backbone**:
   - Well-established architecture
   - Strong feature extraction
   - Pre-trained on ImageNet (if used)
   - Proven performance on vision tasks

### Limitations

1. **Audio Track Requirement**:
   - Cannot analyze videos without audio
   - Fails on silent videos
   - Not applicable to image-only deepfakes
   - Requires audio-visual synchronization

2. **Complex Preprocessing**:
   - Multiple processing steps (audio extraction, spectrogram, cropping)
   - Requires temporary file storage
   - More points of failure
   - Higher computational overhead

3. **Fixed Crop Positions**:
   - Face and lip crops use fixed pixel coordinates
   - Assumes faces are centered and similar size
   - May miss faces in different positions
   - No face detection/alignment used

4. **Large Combined Images**:
   - 1000×2500 pixel images (for 5-frame windows)
   - High memory consumption
   - Slower processing
   - Requires significant GPU memory

5. **Window Sampling Limitations**:
   - Fixed number of windows (n_extract)
   - May miss short manipulation segments
   - Linearly spaced (may not capture all patterns)
   - Averaging can dilute localized signals

6. **No Temporal Modeling**:
   - Each window analyzed independently
   - No sequential dependencies between windows
   - Missing temporal dynamics
   - Cannot capture long-range patterns

7. **Heavy Computational Cost**:
   - 15 crops per window (3 scales × 5 frames)
   - ResNet-50 forward pass per crop
   - Multiple windows per video
   - Slow inference for long videos

---

## Technical Deep Dive

### Why Combine Audio and Visual?

**Audio-Visual Deepfakes**:

Modern deepfakes often manipulate both modalities:

- Voice cloning + face swapping
- Lip-sync attacks (audio doesn't match lips)
- Audio substitution with face re-enactment

**Single-Modality Limitations**:

- Visual-only: May miss audio artifacts
- Audio-only: Cannot detect visual inconsistencies

**Multimodal Advantage**:

- Detects cross-modal inconsistencies
- More robust to single-modality attacks
- Harder to fool both modalities simultaneously

### CLIP Integration

**What Is CLIP?**

Contrastive Language-Image Pre-training:

- Trained on 400M image-text pairs
- Learns aligned vision-language representations
- Strong zero-shot transfer capabilities

**Why Use CLIP?**

- Pre-trained on massive dataset
- Understands semantic content
- Provides rich global features
- No need to train from scratch

**Architecture Options**:

- **ViT-B/32**: Vision Transformer (768-dim output)
- **RN50**: ResNet-50 variant (768-dim output)

### ResNet-50 Bottleneck Architecture

**Bottleneck Block**:

```python
nn.Conv2d(in → width, k=1) → BN → ReLU →
nn.Conv2d(width → width, k=3, padding=1) → BN → ReLU →
nn.Conv2d(width → 4*width, k=1) → BN →
+ residual connection → ReLU
```

**Layer Structure**:

- Layer 1: 3 blocks, 64 channels
- Layer 2: 4 blocks, 128 channels
- Layer 3: 6 blocks, 256 channels
- Layer 4: 3 blocks, 512 channels

**Output**: 2048-dimensional features (512 × 4 expansion)

### Attention Mechanism Details

**Softmax Normalization**:

```python
weights = [w1, w2, w3]  # Raw attention scores
exp_weights = [exp(w1), exp(w2), exp(w3)]
sum_exp = exp(w1) + exp(w2) + exp(w3)
normalized = [exp(w1)/sum_exp, exp(w2)/sum_exp, exp(w3)/sum_exp]
```

**Properties**:

- Sum to 1.0
- All positive values
- Differentiable (for training)
- Emphasizes higher weights

**Weighted Sum**:

```python
output = sum(feature[i] * weight[i] for i in scales)
```

### Mel-Spectrogram Details

**Why Mel Scale?**

Perceptually-motivated frequency scale:

```python
mel = 2595 * log10(1 + freq / 700)
```

**Benefits for Lip-Sync Detection**:

- Captures speech phonetic content
- Represents audio in visual form
- Can be aligned with video frames
- Enables CNN processing of audio

### Multi-Scale Cropping Strategy

**Scale Hierarchy**:

1. **Full Frame (1.0×)**: Context
   - Body language
   - Scene information
   - Overall appearance

2. **Face (0.65×)**: Facial Features
   - Expression
   - Head pose
   - Facial structure

3. **Lip (0.45×)**: Fine Details
   - Mouth shape
   - Lip movements
   - Articulation patterns

**Why Multiple Scales?**

- Different scales capture different information
- Redundancy improves robustness
- Attention learns optimal combination
- Similar to multi-scale CNN approaches

### Temporal Alignment Math

**Mapping Video to Audio**:

```python
video_frames = 100
mel_width = 645  # Spectrogram width in pixels

mapping = mel_width / video_frames = 6.45 pixels/frame

# For window starting at frame 20 with 5 frames
start_frame = 20
window_len = 5

mel_start = int(20 * 6.45) = 129
mel_end = int((20 + 5) * 6.45) = 161

mel_segment = mel_img[:, 129:161]
```

**Ensures**:

- Temporal correspondence between audio and video
- Same time span for both modalities
- Proper synchronization for analysis

---

## Integration Example

```python
import torch
from pathlib import Path

# Load the model
model = load_model('LIPFD-V1')

# Analyze a video file (must have audio)
video_path = Path("suspicious_video.mp4")
result = model.analyze(media_path=str(video_path))

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Processing Time: {result['processing_time']:.2f}s")
print(f"Windows Analyzed: {result['frames_analyzed']}")

# Access per-window predictions
print("\nPer-Window Scores:")
for pred in result['frame_predictions']:
    print(f"  Window {pred['index']}: {pred['score']:.3f} ({pred['prediction']})")

# Access metrics
print(f"\nFinal Average Score: {result['metrics']['final_average_score']:.3f}")
print(f"Window Scores: {result['metrics']['window_scores']}")
```

### Handling Videos Without Audio

```python
from src.ml.exceptions import MediaProcessingError

try:
    result = model.analyze(media_path=video_path)
except MediaProcessingError as e:
    if "no audio track" in str(e).lower():
        print("Error: This model requires videos with audio tracks")
        print("For image/silent video analysis, use a visual-only model")
```

### Interpreting Per-Window Scores

```python
# Get variance of window scores
window_scores = result['metrics']['window_scores']
score_variance = np.var(window_scores)

if score_variance > 0.05:
    print("High variance - manipulation may be localized to certain segments")
else:
    print("Low variance - consistent prediction across video")

# Identify suspicious windows
threshold = 0.7
suspicious_windows = [
    i for i, score in enumerate(window_scores) if score > threshold
]
print(f"Suspicious windows: {suspicious_windows}")
```
