# STFT-Spectrogram-CNN Model

**Model Category**: Audio Analysis  
**Model Type**: Audio Deepfake Detector  
**Version**: V1  
**Primary Detection Target**: Synthetic or manipulated audio content

---

## Overview

### What Is This Model?

STFT-Spectrogram-CNN is an audio deepfake detector that analyzes dual-band STFT (Short-Time Fourier Transform) spectrograms using a Convolutional Neural Network. The model processes audio by creating both wideband and narrowband spectrogram representations, stacking them as a composite image, and analyzing this multi-resolution view for patterns indicative of synthetic audio.

### The Core Concept

The model converts audio into two complementary spectrogram views with different frequency resolutions, combines them into a single RGB image, and applies deep learning to detect manipulation artifacts. By analyzing both high-resolution (narrowband) and overview (wideband) frequency information simultaneously, the model can capture both fine-grained and broad spectral patterns.

### Why Dual-Band Spectrograms?

- **Wideband STFT**: Captures broad frequency overview with good time resolution
- **Narrowband STFT**: Captures detailed frequency information with better frequency resolution
- **Stacked Representation**: Provides multi-scale view for comprehensive analysis

---

## How It Works

### Step-by-Step Process

#### Phase 1: Audio Preprocessing

```text
Input Audio → Standardization → Mono Conversion → Resampling
```

**Audio Standardization**:

- Convert to mono (single channel)
- Resample to target sampling rate (configurable, typically 16kHz)
- Load into memory as waveform array

#### Phase 2: Audio Chunking

```text
Audio Waveform → Split into Fixed-Length Chunks → Pad Final Chunk if Needed
```

**Chunking Parameters**:

- **Chunk Duration**: Fixed length segments (e.g., 3 seconds)
- **Padding**: Zero-padding applied to last chunk if audio length not divisible by chunk size
- **No Overlap**: Chunks are non-overlapping sequential segments

**Example** (15.5s audio, 3s chunks):

```text
Chunk 1: [0.0s - 3.0s]
Chunk 2: [3.0s - 6.0s]
Chunk 3: [6.0s - 9.0s]
Chunk 4: [9.0s - 12.0s]
Chunk 5: [12.0s - 15.0s]
Chunk 6: [15.0s - 15.5s] + 2.5s zero-padding
```

#### Phase 3: Dual-Band STFT Computation

**Per-Chunk Processing**:

```text
Audio Chunk → Wideband STFT + Narrowband STFT → Power to dB → Dual Spectrograms
```

**Wideband STFT**:

```python
stft_wide = librosa.stft(
    audio_chunk,
    n_fft=n_fft_wide,     # Larger FFT size (e.g., 2048)
    hop_length=hop_length, # Step between frames (e.g., 512)
    window="hann"
)
spec_wide_db = librosa.amplitude_to_db(np.abs(stft_wide), ref=np.max)
```

**Narrowband STFT**:

```python
stft_narrow = librosa.stft(
    audio_chunk,
    n_fft=n_fft_narrow,   # Smaller FFT size (e.g., 512)
    hop_length=hop_length, # Same hop length
    window="hann"
)
spec_narrow_db = librosa.amplitude_to_db(np.abs(stft_narrow), ref=np.max)
```

**Key Differences**:

| Parameter | Wideband | Narrowband |
|-----------|----------|------------|
| **FFT Size** | Larger (e.g., 2048) | Smaller (e.g., 512) |
| **Frequency Resolution** | Lower (broader bins) | Higher (finer bins) |
| **Time Resolution** | Higher (shorter windows) | Lower (longer windows) |
| **Use Case** | Temporal dynamics | Frequency detail |

#### Phase 4: Spectrogram Image Generation

```text
Dual Spectrograms → Stack Vertically → Render as RGB Image → Resize
```

**Visualization Process**:

1. **Create Figure**: 2 subplots (wideband top, narrowband bottom)
2. **Plot Spectrograms**: Use librosa.display.specshow()
3. **Remove Axes**: Turn off all axis labels/ticks
4. **Render to Buffer**: Save as PNG in memory
5. **Convert to PIL Image**: Load as RGB image
6. **Apply Transforms**: Resize to target dimensions

**Transformation Pipeline**:

```python
transforms.Compose([
    transforms.ToTensor(),              # Convert to tensor
    transforms.Resize((height, width)), # Resize to fixed size
    transforms.Normalize(               # Normalize RGB channels
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
```

#### Phase 5: CNN Classification

```text
Spectrogram Image Tensor → Conv Layers → Pooling → FC Layers → Binary Output
```

**Convolutional Backbone**:

```text
Input (3×H×W) →
Conv2D(3→16) → ReLU → BatchNorm → MaxPool2D(2×2) →
Conv2D(16→32) → ReLU → BatchNorm → MaxPool2D(2×2) →
Conv2D(32→64) → ReLU → BatchNorm → MaxPool2D(2×2) →
Conv2D(64→128) → ReLU → BatchNorm → MaxPool2D(2×2) →
Feature Maps (128×H'×W')
```

**Fully Connected Classifier**:

```text
Flatten →
Linear(conv_output_size → 256) → ReLU → Dropout(0.5) →
Linear(256 → 2) →
Softmax → [prob_real, prob_fake]
```

#### Phase 6: Aggregation & Decision

```text
Per-Chunk Predictions → Average Probabilities → Final Classification
```

**Averaging Process**:

```python
all_chunk_predictions = [chunk_1_probs, chunk_2_probs, ..., chunk_N_probs]
avg_probs = mean(all_chunk_predictions)  # [prob_real, prob_fake]

if prob_real >= prob_fake:
    prediction = "REAL"
    confidence = prob_real
else:
    prediction = "FAKE"
    confidence = prob_fake
```

---

## Architecture Details

### Model Structure

```text
┌─────────────────────────────────────────────────────────────┐
│                      Input Audio File                        │
│                   (any format, any duration)                 │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Audio Preprocessing                             │
│   • Convert to mono (1 channel)                              │
│   • Resample to target SR (e.g., 16kHz)                      │
│   • Load as waveform array                                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Audio Chunking                                  │
│   • Non-overlapping sequential chunks                        │
│   • Fixed duration (e.g., 3 seconds)                         │
│   • Zero-padding for final chunk if needed                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Per-Chunk: Dual-Band STFT Computation                │
│   Wideband STFT:  n_fft=large (e.g., 2048)                  │
│   Narrowband STFT: n_fft=small (e.g., 512)                  │
│   Both → Power to dB conversion                              │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Per-Chunk: Spectrogram Stacking                      │
│   • Vertically stack wideband (top) + narrowband (bottom)   │
│   • Render as RGB PNG image                                  │
│   • Resize to fixed dimensions (e.g., 256×256)               │
│   • Normalize RGB channels                                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              CNN Feature Extraction                          │
│   Conv Block 1: 3→16 channels                               │
│   Conv Block 2: 16→32 channels                              │
│   Conv Block 3: 32→64 channels                              │
│   Conv Block 4: 64→128 channels                             │
│   Each block: Conv3×3 → ReLU → BatchNorm → MaxPool2×2      │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Fully Connected Classifier                      │
│   Flatten → FC(→256) → ReLU → Dropout(0.5) → FC(→2)       │
│   Output: [prob_real, prob_fake]                             │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Aggregation Across Chunks                       │
│   • Average probabilities from all chunks                    │
│   • Select class with higher probability                     │
│   • Extract confidence score                                 │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Final Prediction: REAL or FAKE                       │
│         + Confidence Score + Audio Metrics                   │
└─────────────────────────────────────────────────────────────┘
```

### CNN Architecture Breakdown

**Convolutional Layers**:

| Layer | Input Channels | Output Channels | Kernel Size | Padding | Output Size After Pool |
|-------|----------------|-----------------|-------------|---------|------------------------|
| Conv1 | 3 | 16 | 3×3 | 1 | H/2 × W/2 |
| Conv2 | 16 | 32 | 3×3 | 1 | H/4 × W/4 |
| Conv3 | 32 | 64 | 3×3 | 1 | H/8 × W/8 |
| Conv4 | 64 | 128 | 3×3 | 1 | H/16 × W/16 |

**Each Conv Block**:

```text
Conv2D → ReLU → BatchNorm2D → MaxPool2D(2×2)
```

**Fully Connected Layers**:

```text
Flatten → Linear(conv_output_size → 256) → ReLU → Dropout(0.5) → Linear(256 → 2)
```

**Output**:

- 2 logits: [real_logit, fake_logit]
- Softmax applied: [prob_real, prob_fake]

### Parameter Count

**Convolutional Layers**:

- Conv1: 3×16×3×3 + 16 = 448
- Conv2: 16×32×3×3 + 32 = 4,640
- Conv3: 32×64×3×3 + 64 = 18,496
- Conv4: 64×128×3×3 + 128 = 73,856
- **Total Conv**: ~97K parameters

**Batch Normalization**: ~400 parameters (scale + shift per channel)

**Fully Connected** (depends on conv_output_size):

- Assuming 256×256 input → 16×16 feature map after 4 poolings
- conv_output_size = 128 × 16 × 16 = 32,768
- FC1: 32,768×256 + 256 = 8,389,120
- FC2: 256×2 + 2 = 514
- **Total FC**: ~8.4M parameters

**Total Model Parameters**: ~8.5M (mostly in first FC layer)

---

## Input Requirements

### Audio Specifications

| Parameter | Requirement | Notes |
|-----------|-------------|-------|
| **Format** | MP3, WAV, OGG, FLAC, M4A, etc. | Common audio formats supported |
| **Sample Rate** | Any | Resampled internally (typically to 16kHz) |
| **Duration** | Minimum = chunk duration | Must be at least as long as one chunk |
| **Channels** | Any | Converted to mono internally |
| **Bitrate** | Any | No specific requirement |

### Configuration Parameters

**STFT Parameters**:

- **sampling_rate**: Target sample rate (e.g., 16000 Hz)
- **n_fft_wide**: FFT size for wideband STFT (e.g., 2048)
- **n_fft_narrow**: FFT size for narrowband STFT (e.g., 512)
- **hop_length**: STFT hop length (e.g., 512)

**Chunking Parameters**:

- **chunk_duration_s**: Length of each audio chunk (e.g., 3 seconds)

**Image Parameters**:

- **img_height**: Target image height (e.g., 256)
- **img_width**: Target image width (e.g., 256)
- **dpi**: Resolution for spectrogram rendering (e.g., 100)

### Preprocessing Pipeline

1. **Audio Loading**: Load file using pydub
2. **Standardization**:
   - Convert to mono
   - Resample to target SR
3. **Chunking**: Split into fixed-length segments with zero-padding
4. **Per-Chunk Processing**:
   - Dual-band STFT computation
   - Spectrogram stacking and visualization
   - Image conversion and normalization

---

## Output Format

### JSON Response Structure

```json
{
  "prediction": "FAKE",
  "confidence": 0.82,
  "processing_time": 6.7,
  "properties": {
    "duration_seconds": 18.0,
    "sample_rate": 16000,
    "channels": 1
  },
  "pitch": {
    "mean_pitch_hz": 165.3,
    "pitch_stability_score": 0.78
  },
  "energy": {
    "rms_energy": 0.038,
    "silence_ratio": 0.15
  },
  "spectral": {
    "spectral_centroid": 1580.2,
    "spectral_contrast": 22.4
  },
  "visualization": {
    "spectrogram_url": "/path/to/spectrogram.png",
    "spectrogram_data": [[...], [...]]
  }
}
```

### Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | string | "REAL" or "FAKE" classification |
| `confidence` | float | Confidence score (0-1) |
| `processing_time` | float | Total analysis time in seconds |
| `properties` | object | Audio file properties |
| `pitch` | object | Pitch analysis metrics |
| `energy` | object | Energy and silence analysis |
| `spectral` | object | Spectral characteristics |
| `visualization` | object | Visualization data and path |

### Audio Analysis Metrics

**Properties**:

- `duration_seconds`: Total audio duration
- `sample_rate`: Sample rate used
- `channels`: Number of channels (always 1 after preprocessing)

**Pitch Analysis**:

- `mean_pitch_hz`: Average fundamental frequency
- `pitch_stability_score`: Consistency of pitch (0-1, higher = more stable)

**Energy Analysis**:

- `rms_energy`: Root mean square energy level
- `silence_ratio`: Proportion of audio detected as silence

**Spectral Analysis**:

- `spectral_centroid`: Center of mass of spectrum (indicates brightness)
- `spectral_contrast`: Difference between peaks and valleys in spectrum

---

## Architecture Strengths & Limitations

### Strengths

1. **Dual-Band Analysis**:
   - Captures both time and frequency domain information at multiple resolutions
   - Wideband provides temporal dynamics
   - Narrowband provides frequency detail
   - Complementary views enhance detection capability

2. **End-to-End CNN Architecture**:
   - All parameters trainable (no fixed feature extractors)
   - Can learn optimal features for audio deepfake detection
   - Hierarchical feature learning through conv layers

3. **Chunk-Based Processing**:
   - Can handle audio of any length
   - Parallel processing potential for chunks
   - Memory-efficient for long audio files

4. **Standard CNN Components**:
   - Well-established architecture pattern
   - BatchNorm for training stability
   - MaxPool for translation invariance
   - Dropout for regularization

5. **Comprehensive Audio Metrics**:
   - Provides interpretable audio characteristics
   - Pitch, energy, and spectral analysis
   - Helps understand audio properties beyond binary prediction

### Limitations

1. **Minimum Duration Requirement**:
   - Audio must be at least as long as chunk duration
   - Very short audio clips cannot be analyzed
   - Zero-padding may affect final chunk quality

2. **Fixed Chunking Strategy**:
   - All chunks are same length
   - May not align with natural audio boundaries
   - Potentially splits speech/audio events

3. **Image Conversion Overhead**:
   - Requires rendering spectrograms as images
   - Adds computational cost (matplotlib rendering)
   - May lose some numerical precision in image conversion

4. **Simple Aggregation**:
   - Averages chunk probabilities equally
   - Doesn't weight chunks by reliability
   - No temporal modeling across chunks

5. **Large Fully Connected Layer**:
   - First FC layer contains most parameters (~8M)
   - Potential overfitting risk
   - Large memory footprint

6. **No Temporal Context Between Chunks**:
   - Each chunk analyzed independently
   - Doesn't capture patterns spanning multiple chunks
   - Missing long-range temporal dependencies

---

## Technical Deep Dive

### STFT (Short-Time Fourier Transform)

**What Is STFT?**

STFT analyzes audio by computing FFT over short, overlapping windows:

```python
stft = librosa.stft(
    y=audio,
    n_fft=2048,      # Window size
    hop_length=512,  # Step between windows
    window="hann"    # Window function
)
```

**Time-Frequency Trade-off**:

| Parameter | Effect on Time Resolution | Effect on Frequency Resolution |
|-----------|---------------------------|--------------------------------|
| **Large n_fft** | Lower (wider window) | Higher (finer bins) |
| **Small n_fft** | Higher (narrower window) | Lower (broader bins) |

**Wideband vs Narrowband**:

- **Wideband (large n_fft)**: Good for frequency detail, shows harmonics clearly
- **Narrowband (small n_fft)**: Good for temporal changes, shows timing better

### Why Stack Two Spectrograms?

**Complementary Information**:

1. **Wideband** captures:
   - Fine frequency structure
   - Harmonic patterns
   - Spectral details

2. **Narrowband** captures:
   - Temporal dynamics
   - Rapid changes
   - Time-domain artifacts

**Combined Analysis**:

By stacking both views, the CNN receives a richer representation that includes both time and frequency perspectives simultaneously.

### CNN Feature Learning

**Hierarchical Pattern Detection**:

**Layer 1** (3→16 channels):

- Learns basic patterns: edges, gradients, simple textures
- Small receptive field

**Layer 2** (16→32 channels):

- Combines basic patterns into mid-level features
- Detects simple structures

**Layer 3** (32→64 channels):

- Higher-level patterns
- More complex combinations

**Layer 4** (64→128 channels):

- Abstract features specific to audio deepfakes
- Large receptive field
- Holistic pattern recognition

**MaxPooling**:

- Reduces spatial dimensions by 2× each layer
- Provides translation invariance
- Reduces computation in deeper layers

### Normalization Strategy

**RGB Channel Normalization**:

```python
transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)
```

**Effect**:

- Scales pixel values from [0, 1] to [-1, 1]
- Centers data around zero
- Helps with gradient flow during training
- Standard practice for image-based models

### Chunk Padding Strategy

**Zero-Padding**:

```python
if len(audio) % chunk_len != 0:
    pad_len = chunk_len - (len(audio) % chunk_len)
    audio = np.pad(audio, (0, pad_len), mode="constant")
```

**Why Zero-Padding?**

- Ensures all chunks are same length
- Allows batch processing
- Simpler implementation

**Potential Issues**:

- Last chunk may have significant silence
- Could affect prediction for that chunk
- Mitigated by averaging across all chunks

---

## Integration Example

```python
import torch
from pathlib import Path

# Load the model
model = load_model('STFT-SPECTROGRAM-CNN-V1')

# Analyze an audio file
audio_path = Path("suspicious_audio.mp3")
result = model.analyze(media_path=str(audio_path))

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Processing Time: {result['processing_time']:.2f}s")
print(f"Duration: {result['properties']['duration_seconds']:.1f}s")

# Access pitch analysis
if result['pitch']['mean_pitch_hz']:
    print(f"Mean Pitch: {result['pitch']['mean_pitch_hz']:.1f} Hz")
    print(f"Pitch Stability: {result['pitch']['pitch_stability_score']:.2f}")

# Access spectral characteristics
print(f"Spectral Centroid: {result['spectral']['spectral_centroid']:.1f} Hz")
print(f"Spectral Contrast: {result['spectral']['spectral_contrast']:.1f} dB")

# Access visualization
if result['visualization']['spectrogram_url']:
    print(f"Spectrogram saved: {result['visualization']['spectrogram_url']}")
```

### Handling Short Audio

```python
from src.ml.exceptions import MediaProcessingError

try:
    result = model.analyze(media_path=audio_path)
    print(f"Prediction: {result['prediction']}")
except MediaProcessingError as e:
    print(f"Error: Audio too short for analysis")
    print(f"Minimum duration required: {model.config.chunk_duration_s}s")
```
