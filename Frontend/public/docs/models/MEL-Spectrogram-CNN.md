# MEL-Spectrogram-CNN Model

**Model Category**: Audio Analysis  
**Model Type**: Audio Deepfake Detector  
**Versions**: V2, V3  
**Primary Detection Target**: Synthetic or manipulated audio content

---

## Overview

### What Is This Model?

MEL-Spectrogram-CNN is an audio deepfake detector that analyzes mel-spectrogram representations of audio using a Convolutional Neural Network with Wavelet Scattering Transform features. The model processes audio in chunks and identifies patterns indicative of synthetic or manipulated audio.

### The Core Concept

The model converts audio into mel-spectrograms (visual representations of audio frequency content over time) and analyzes these spectrograms using deep learning. It leverages the Kymatio Wavelet Scattering Transform as a feature extractor to capture multi-scale patterns in the spectrogram data.

### Two Versions

- **V2**: Standard non-overlapping chunk analysis
- **V3**: Overlapping chunk analysis with temporal visualization

Both versions share the same core architecture but differ in how they segment and analyze audio.

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

**V2 (Non-overlapping)**:

```text
Audio Waveform → Split into Non-overlapping Chunks → Process Each Chunk
```

**V3 (Overlapping)**:

```text
Audio Waveform → Split into Overlapping Chunks → Process Each Chunk
```

**Chunking Parameters**:

- **Chunk Duration**: Fixed length segments (e.g., 3 seconds)
- **Chunk Overlap** (V3 only): Overlap between consecutive chunks
- **Step Size**: Distance between chunk start positions

#### Phase 3: Mel-Spectrogram Generation

```text
Audio Chunk → Pre-emphasis → STFT → Mel Filterbank → Power to dB → Spectrogram Image
```

**Processing Steps**:

1. **Pre-emphasis Filtering**:

   ```python
   y_preemphasized = librosa.effects.preemphasis(audio_chunk)
   ```

   - Boosts high frequencies
   - Compensates for natural spectral slope

2. **Mel-Spectrogram Computation**:

   ```python
   mel_spec = librosa.feature.melspectrogram(
       y=y_preemphasized,
       sr=sampling_rate,
       n_fft=n_fft,           # FFT window size
       hop_length=hop_length, # Step size between frames
       n_mels=n_mels          # Number of mel bands
   )
   ```

3. **Convert to Decibels**:

   ```python
   mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
   ```

4. **Render as Image**:
   - Plot spectrogram using matplotlib
   - Save as PNG image in memory
   - Resize to 256×256 pixels

#### Phase 4: Feature Extraction (Wavelet Scattering Transform)

```text
Spectrogram Image → Wavelet Scattering (Kymatio) → Multi-scale Features
```

**Scattering2D Configuration**:

```python
Scattering2D(J=4, L=8, shape=(256, 256))
```

**Parameters**:

- **J=4**: Number of scales (4 octaves)
- **L=8**: Number of angles (8 orientations)
- **shape=(256, 256)**: Input image dimensions

**What It Does**:

- Applies cascaded wavelet transforms
- Captures multi-scale, multi-orientation patterns
- Produces translation-invariant features
- Fixed (non-trainable) feature extractor

#### Phase 5: Classification Network

```text
Scattering Features → Flatten → Deep Classifier → Binary Output
```

**Classifier Architecture**:

```text
Flatten →
Linear(scattering_dim → 1024) → ReLU → BatchNorm → Dropout(0.4) →
Linear(1024 → 512) → ReLU → BatchNorm → Dropout(0.4) →
Linear(512 → 128) → ReLU → BatchNorm → Dropout(0.3) →
Linear(128 → 1)
```

**Output**: Single logit value

- Sigmoid applied during inference to get probability
- **V2**: Probability of REAL (inverted to get FAKE score)
- **V3**: Directly outputs FAKE score

#### Phase 6: Aggregation & Decision

**V2 (Non-overlapping)**:

```text
Chunk Predictions → Average → Final Classification
```

**V3 (Overlapping)**:

```text
Overlapping Chunk Predictions → Average → Temporal Visualization → Final Classification
```

**Decision Threshold**: 0.5

- Score ≥ 0.5: FAKE
- Score < 0.5: REAL

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
│   V2: Non-overlapping chunks                                 │
│   V3: Overlapping chunks (with configurable overlap)         │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Per-Chunk: Mel-Spectrogram Generation                │
│   • Pre-emphasis filtering                                   │
│   • Mel-spectrogram computation (librosa)                    │
│   • Power to dB conversion                                   │
│   • Render as 256×256 PNG image                              │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Wavelet Scattering Transform (Kymatio)               │
│   Fixed Feature Extractor (Non-trainable)                    │
│   • J=4 scales                                               │
│   • L=8 orientations                                         │
│   • Translation-invariant features                           │
│   • Output: Multi-dimensional feature vector                 │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Deep Classifier Network                         │
│   Flatten → FC(→1024) → ReLU → BN → Dropout(0.4)           │
│          → FC(→512)  → ReLU → BN → Dropout(0.4)            │
│          → FC(→128)  → ReLU → BN → Dropout(0.3)            │
│          → FC(→1)                                            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Per-Chunk Prediction (Sigmoid Applied)               │
│   Score: [0.0 - 1.0]                                         │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Aggregation Across Chunks                       │
│   • Average all chunk scores                                 │
│   • Apply threshold (0.5)                                    │
│   • Generate final prediction                                │
│   • V3: Generate temporal visualization                      │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Final Prediction: REAL or FAKE                       │
│         + Confidence Score + Metrics                         │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Kymatio Scattering2D (Fixed)**
   - Non-trainable wavelet transform
   - Multi-scale feature extraction
   - Translation-invariant representations

2. **Deep Classifier (Trainable)**
   - 4-layer fully connected network
   - Batch normalization for stability
   - Dropout for regularization
   - Parameters: Varies based on scattering output dimension

**Total Parameters**: ~1-2M (mostly in classifier)  
**Trainable Parameters**: ~1-2M (classifier only)

### Version-Specific Differences

| Aspect | V2 | V3 |
|--------|----|----|
| **Chunking** | Non-overlapping | Overlapping |
| **Chunk Overlap** | 0 seconds | Configurable |
| **Step Size** | = Chunk duration | = Chunk duration - Overlap |
| **Output Format** | Average probability | Average + temporal plot |
| **Visualization** | Last mel-spectrogram | Temporal score plot |
| **Metrics** | Basic audio properties | + chunk-level predictions |

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

**Common (V2 & V3)**:

- **sampling_rate**: Target sample rate (e.g., 16000 Hz)
- **chunk_duration_s**: Length of each audio chunk (e.g., 3 seconds)
- **n_fft**: FFT window size (e.g., 2048)
- **hop_length**: STFT hop length (e.g., 512)
- **n_mels**: Number of mel frequency bands (e.g., 128)
- **dpi**: Resolution for spectrogram rendering (e.g., 100)

**V3 Specific**:

- **chunk_overlap_s**: Overlap between chunks (e.g., 1.5 seconds)

### Preprocessing Pipeline

1. **Audio Loading**: Load file using pydub/librosa
2. **Standardization**:
   - Convert to mono
   - Resample to target SR
3. **Chunking**: Split into fixed-length segments
4. **Per-Chunk Processing**:
   - Pre-emphasis filtering
   - Mel-spectrogram generation
   - Image conversion
   - Tensor transformation

---

## Output Format

### JSON Response Structure (V2)

```json
{
  "prediction": "FAKE",
  "confidence": 0.73,
  "processing_time": 8.4,
  "properties": {
    "duration_seconds": 15.5,
    "sample_rate": 16000,
    "channels": 1
  },
  "pitch": {
    "mean_pitch_hz": 180.5,
    "pitch_stability_score": 0.85
  },
  "energy": {
    "rms_energy": 0.042,
    "silence_ratio": 0.12
  },
  "spectral": {
    "spectral_centroid": 1420.3,
    "spectral_contrast": 18.7
  },
  "visualization": {
    "spectrogram_url": "/path/to/spectrogram.png",
    "spectrogram_data": [[...], [...]]
  }
}
```

### JSON Response Structure (V3)

```json
{
  "prediction": "FAKE",
  "confidence": 0.68,
  "processing_time": 12.1,
  "properties": { "..." },
  "pitch": { "..." },
  "energy": { "..." },
  "spectral": { "..." },
  "visualization": {
    "spectrogram_url": "/path/to/temporal_plot.png",
    "spectrogram_data": [0.65, 0.72, 0.68, 0.71, ...]
  },
  "metrics": {
    "average_chunk_fake_score": 0.68,
    "chunk_predictions": [0.65, 0.72, 0.68, 0.71, ...],
    "chunk_duration_s": 3.0,
    "chunk_overlap_s": 1.5,
    "num_chunks": 12
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
| `metrics` | object | Model-specific metrics (V3 only) |

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

1. **Wavelet Scattering Transform**:
   - Multi-scale feature extraction
   - Translation-invariant representations
   - Captures complex patterns at multiple resolutions
   - Fixed features (no need to train feature extractor)

2. **Chunk-Based Analysis**:
   - Can process audio of any length
   - Localized analysis for temporal patterns
   - V3 overlapping provides smoother temporal coverage

3. **Deep Classifier**:
   - Multiple fully connected layers for pattern learning
   - Batch normalization for training stability
   - Dropout regularization prevents overfitting

4. **Comprehensive Audio Metrics**:
   - Provides interpretable audio characteristics
   - Pitch, energy, and spectral analysis
   - Helps understand audio properties beyond binary prediction

### Limitations

1. **Minimum Duration Requirement**:
   - Audio must be at least as long as chunk duration
   - Very short audio clips cannot be analyzed
   - Cannot adapt to arbitrary lengths below chunk size

2. **Fixed Chunking Strategy**:
   - All chunks are same length
   - May not align with natural audio boundaries
   - Potentially splits speech/audio events

3. **Spectrogram Image Conversion**:
   - Requires rendering spectrogram as image
   - Adds computational overhead
   - May lose some information in image conversion

4. **Aggregation Simplicity**:
   - Simple averaging of chunk scores
   - Doesn't weight chunks by reliability
   - All chunks contribute equally to final prediction

5. **Feature Extractor Constraints**:
   - Scattering transform is fixed (cannot be trained)
   - May not adapt optimally to specific audio types
   - Limited to patterns the scattering can capture

6. **V3 Computational Cost**:
   - Overlapping chunks increase processing time
   - More chunks to analyze for same duration audio
   - Trade-off between temporal resolution and speed

---

## Technical Deep Dive

### Wavelet Scattering Transform

**Kymatio Implementation**:

The Scattering2D transform applies cascaded wavelet convolutions:

```python
from kymatio.torch import Scattering2D

scattering = Scattering2D(J=4, L=8, shape=(256, 256))
```

**Parameters Explained**:

- **J=4**: Number of scales
  - 4 octaves of frequency decomposition
  - Captures patterns at scales: 2^0, 2^1, 2^2, 2^3
  
- **L=8**: Number of orientations
  - 8 directional wavelets
  - Covers 180° (π radians) uniformly
  
- **shape=(256, 256)**: Input image dimensions
  - Fixed size for consistency
  - All spectrograms resized to this

**Output Characteristics**:

- Multi-dimensional feature vector
- Translation-invariant (robust to small shifts)
- Captures local and global patterns
- Dimension varies based on J and L parameters

### Mel-Spectrogram Generation

**Step-by-Step**:

1. **Pre-emphasis**:

   ```python
   y_preemphasized = librosa.effects.preemphasis(audio_chunk)
   ```

   - High-pass filter
   - Compensates for spectral tilt
   - Emphasizes high frequencies

2. **Mel-Spectrogram**:

   ```python
   mel_spec = librosa.feature.melspectrogram(
       y=y_preemphasized,
       sr=16000,
       n_fft=2048,      # Window size: 128ms at 16kHz
       hop_length=512,  # Step size: 32ms
       n_mels=128       # Frequency bands
   )
   ```

3. **Decibel Conversion**:

   ```python
   mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
   ```

   - Converts power to logarithmic scale
   - More perceptually relevant
   - Better dynamic range

### Classifier Network Details

**Architecture**:

```python
nn.Sequential(
    nn.Flatten(),
    nn.Linear(scattering_output_dim, 1024),
    nn.ReLU(),
    nn.BatchNorm1d(1024),
    nn.Dropout(0.4),
    
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.4),
    
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.BatchNorm1d(128),
    nn.Dropout(0.3),
    
    nn.Linear(128, 1)
)
```

**Layer Purpose**:

- **Flatten**: Converts multi-dimensional scattering output to 1D
- **Linear**: Fully connected layers for pattern learning
- **ReLU**: Non-linear activation
- **BatchNorm1d**: Normalizes activations for stability
- **Dropout**: Randomly drops neurons during training (regularization)

**Dropout Rates**:

- Higher dropout (0.4) in earlier layers
- Lower dropout (0.3) in later layers
- No dropout in final layer

### Chunking Strategies

**V2 (Non-overlapping)**:

```python
chunk_len = int(chunk_duration_s * sample_rate)
step_size = chunk_len  # No overlap

chunks = []
start = 0
while start + chunk_len <= len(audio):
    chunks.append(audio[start : start + chunk_len])
    start += step_size
```

**V3 (Overlapping)**:

```python
chunk_len = int(chunk_duration_s * sample_rate)
overlap_len = int(chunk_overlap_s * sample_rate)
step_size = chunk_len - overlap_len  # Overlap

chunks = []
start = 0
while start + chunk_len <= len(audio):
    chunks.append(audio[start : start + chunk_len])
    start += step_size
```

**Example** (3s chunks, 1.5s overlap):

```text
Chunk 1: [0.0s - 3.0s]
Chunk 2: [1.5s - 4.5s]  ← 1.5s overlap with Chunk 1
Chunk 3: [3.0s - 6.0s]  ← 1.5s overlap with Chunk 2
...
```

---

## Integration Example

```python
import torch
from pathlib import Path

# Load the model (specify version)
model_v2 = load_model('MEL-Spectrogram-CNN-V2')
model_v3 = load_model('MEL-Spectrogram-CNN-V3')

# Analyze an audio file with V2
audio_path = Path("audio.mp3")
result_v2 = model_v2.analyze(media_path=str(audio_path))

print(f"Prediction: {result_v2['prediction']}")
print(f"Confidence: {result_v2['confidence']:.2%}")
print(f"Duration: {result_v2['properties']['duration_seconds']:.1f}s")
print(f"Mean Pitch: {result_v2['pitch']['mean_pitch_hz']:.1f} Hz")

# Analyze with V3 (overlapping chunks)
result_v3 = model_v3.analyze(media_path=str(audio_path))

print(f"\nV3 Analysis:")
print(f"Prediction: {result_v3['prediction']}")
print(f"Chunks Analyzed: {result_v3['metrics']['num_chunks']}")
print(f"Chunk Overlap: {result_v3['metrics']['chunk_overlap_s']}s")

# Access temporal visualization (V3)
if result_v3['visualization']['spectrogram_url']:
    print(f"Temporal plot saved: {result_v3['visualization']['spectrogram_url']}")
```

### Handling Short Audio

```python
result = model.analyze(media_path=audio_path)

# Check if audio was long enough
try:
    prediction = result['prediction']
except MediaProcessingError as e:
    print(f"Error: {e}")
    # Audio too short for analysis
```
