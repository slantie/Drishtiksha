# Scattering-Wave Model

**Model Category**: Audio Analysis  
**Model Type**: Audio Deepfake Detector  
**Version**: V1  
**Primary Detection Target**: Synthetic or manipulated audio content

---

## Overview

### What Is This Model?

Scattering-Wave is an audio deepfake detector that analyzes mel-spectrogram representations using a Wavelet Scattering Transform followed by a deep classifier network. The model extracts multi-scale, translation-invariant features from audio spectrograms and uses these features to identify patterns indicative of synthetic or manipulated audio.

### The Core Concept

The model converts audio into a mel-spectrogram (visual representation of audio frequency content), applies a 2D Wavelet Scattering Transform to extract robust features, and passes these features through a deep neural network classifier. The scattering transform acts as a powerful fixed feature extractor that captures complex patterns at multiple scales and orientations.

### Key Innovation

Unlike standard CNN approaches, this model uses the **Wavelet Scattering Transform** (via Kymatio library) as a mathematically-grounded feature extractor that:

- Provides translation-invariant representations
- Captures multi-scale patterns automatically
- Requires no training for the feature extraction stage
- Produces stable, interpretable features

---

## How It Works

### Step-by-Step Process

#### Phase 1: Audio Extraction & Preprocessing

```text
Input Media → Audio Extraction → Mono Conversion → Resampling → Pre-emphasis
```

**Audio Standardization**:

- Extract audio track from media file (audio/video)
- Convert to mono (single channel)
- Resample to target sampling rate (configurable, typically 16kHz)
- Apply pre-emphasis filtering to boost high frequencies

**Pre-emphasis Filter**:

```python
y_preemphasized = librosa.effects.preemphasis(y)
```

- High-pass filter that amplifies higher frequencies
- Compensates for natural spectral tilt in audio
- Emphasizes important high-frequency details

#### Phase 2: Audio Duration Standardization

```text
Audio Waveform → Pad or Truncate → Fixed Duration
```

**Duration Normalization**:

- **If too short**: Zero-pad to target duration
- **If too long**: Truncate to target duration
- **Target**: Fixed duration (e.g., 10 seconds)

**Example** (target = 10 seconds):

```python
target_length = sampling_rate * duration_seconds  # e.g., 16000 * 10 = 160000 samples

if len(audio) < target_length:
    audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
else:
    audio = audio[:target_length]
```

#### Phase 3: Mel-Spectrogram Generation

```text
Pre-emphasized Audio → STFT → Mel Filterbank → Power to dB → Mel-Spectrogram
```

**Processing Steps**:

1. **Mel-Spectrogram Computation**:

   ```python
   mel_spec = librosa.feature.melspectrogram(
       y=y_preemphasized,
       sr=sampling_rate,
       n_fft=2048,        # FFT window size
       hop_length=512,    # Step between frames
       n_mels=256         # Number of mel bands
   )
   ```

2. **Convert to Decibels**:

   ```python
   mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
   ```

   - Converts power to logarithmic scale
   - More perceptually relevant
   - Better dynamic range for visualization

#### Phase 4: Spectrogram Image Conversion

```text
Mel-Spectrogram → Matplotlib Rendering → PNG Image → Grayscale → Resize → Tensor
```

**Visualization Process**:

1. **Create Figure**: 4×4 inches at 100 DPI
2. **Plot Spectrogram**: Use librosa.display.specshow()
3. **Remove Axes**: Turn off all borders and labels
4. **Render to Buffer**: Save as PNG in memory
5. **Convert to PIL Image**: Load as grayscale ('L' mode)

**Transformation Pipeline**:

```python
transforms.Compose([
    transforms.Resize((256, 256), antialias=True),  # Resize to fixed size
    transforms.Grayscale(num_output_channels=1),    # Ensure grayscale
    transforms.ToTensor()                            # Convert to tensor
])
```

**Output**: Single-channel tensor of shape (1, 256, 256)

#### Phase 5: Wavelet Scattering Transform

```text
Spectrogram Image Tensor → Scattering2D → Multi-scale Features
```

**Scattering2D Configuration**:

```python
Scattering2D(J=4, L=8, shape=(256, 256))
```

**Parameters**:

- **J=4**: Number of scales (4 octaves)
- **L=8**: Number of orientations (8 angles)
- **shape=(256, 256)**: Input image dimensions

**What It Does**:

- Applies cascaded wavelet transforms
- Computes wavelet coefficients at multiple scales
- Takes modulus and averages to create scattering coefficients
- Produces translation-invariant, multi-scale features
- Fixed (non-trainable) feature extractor

**Output**: High-dimensional feature vector (dimension varies based on J and L)

#### Phase 6: Deep Classifier Network

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
- Output is probability of REAL
- FAKE probability = 1 - REAL probability

#### Phase 7: Decision & Analysis

```text
Model Output → Sigmoid → Probability → Threshold → Final Classification
```

**Decision Logic**:

```python
prob_real = sigmoid(output)
prob_fake = 1.0 - prob_real

if prob_real >= 0.5:
    prediction = "REAL"
    confidence = prob_real
else:
    prediction = "FAKE"
    confidence = prob_fake
```

**Additional Analysis**:

- Pitch analysis (mean pitch, stability)
- Energy analysis (RMS energy, silence ratio)
- Spectral analysis (centroid, contrast)
- Spectrogram visualization saved to disk

---

## Architecture Details

### Model Structure

```text
┌─────────────────────────────────────────────────────────────┐
│                      Input Media File                        │
│                   (audio or video format)                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Audio Extraction & Preprocessing                │
│   • Extract audio track (pydub)                              │
│   • Convert to mono (1 channel)                              │
│   • Resample to target SR (e.g., 16kHz)                      │
│   • Pre-emphasis filtering                                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Duration Standardization                        │
│   • Pad to target duration if too short                      │
│   • Truncate to target duration if too long                  │
│   • Fixed duration output (e.g., 10 seconds)                 │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Mel-Spectrogram Generation                      │
│   • STFT with n_fft=2048, hop_length=512                    │
│   • Mel filterbank with n_mels=256                           │
│   • Power to dB conversion                                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Spectrogram Image Conversion                    │
│   • Render spectrogram using matplotlib                      │
│   • Save as PNG in memory buffer                             │
│   • Convert to PIL grayscale image                           │
│   • Resize to 256×256 pixels                                 │
│   • Convert to tensor (1×256×256)                            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Wavelet Scattering Transform (Kymatio)               │
│   Fixed Feature Extractor (Non-trainable)                    │
│   • J=4 scales (4 octaves)                                   │
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
│         Sigmoid Activation → Probability                     │
│   prob_real = sigmoid(output)                                │
│   prob_fake = 1 - prob_real                                  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Audio Analysis & Metrics Computation                 │
│   • Pitch analysis (pYIN algorithm)                          │
│   • Energy analysis (RMS, silence detection)                 │
│   • Spectral analysis (centroid, contrast)                   │
│   • Save spectrogram visualization                           │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Final Prediction: REAL or FAKE                       │
│         + Confidence Score + Comprehensive Metrics           │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Kymatio Scattering2D (Fixed)**
   - Non-trainable wavelet transform
   - Multi-scale, multi-orientation feature extraction
   - Translation-invariant representations
   - Mathematically grounded feature extractor

2. **Deep Classifier (Trainable)**
   - 4-layer fully connected network
   - Batch normalization for training stability
   - Dropout for regularization
   - Parameters: ~1-2M (depends on scattering output dimension)

**Total Parameters**: ~1-2M (all in classifier)  
**Trainable Parameters**: ~1-2M (classifier only)  
**Non-trainable**: Scattering2D transform

### Scattering Transform Details

**Wavelet Scattering Theory**:

The scattering transform computes:

```text
S₀x = x * φ                    (Low-pass filtering)
S₁x = |x * ψⱼ,ₗ| * φ          (First-order scattering)
S₂x = ||x * ψⱼ₁,ₗ₁| * ψⱼ₂,ₗ₂| * φ  (Second-order scattering)
```

Where:

- **ψⱼ,ₗ**: Wavelet at scale j and orientation l
- **φ**: Averaging filter
- **|·|**: Modulus (complex magnitude)
- **\***: Convolution operator

**Output Features**:

- Zeroth-order: Low-pass coefficients
- First-order: Single wavelet responses
- Second-order: Cascaded wavelet responses
- All concatenated into feature vector

---

## Input Requirements

### Media Specifications

| Parameter | Requirement | Notes |
|-----------|-------------|-------|
| **Format** | Audio: MP3, WAV, OGG, FLAC, M4A, etc. | Common formats supported via pydub |
|  | Video: MP4, AVI, MKV, etc. | Audio extracted from video |
| **Sample Rate** | Any | Resampled internally (typically to 16kHz) |
| **Duration** | Any | Padded or truncated to fixed duration |
| **Channels** | Any | Converted to mono internally |
| **Bitrate** | Any | No specific requirement |

### Configuration Parameters

**Audio Processing**:

- **sampling_rate**: Target sample rate (e.g., 16000 Hz)
- **duration_seconds**: Fixed duration for analysis (e.g., 10 seconds)
- **n_fft**: FFT window size (e.g., 2048)
- **hop_length**: STFT hop length (e.g., 512)
- **n_mels**: Number of mel frequency bands (e.g., 256)

**Image Processing**:

- **image_size**: Target image dimensions (e.g., [256, 256])

**Model Processing**:

- **J**: Scattering scale parameter (e.g., 4)
- **L**: Scattering orientation parameter (e.g., 8)

### Preprocessing Pipeline

1. **Media Loading**: Load file using pydub
2. **Audio Extraction**: Extract audio track (if video)
3. **Standardization**:
   - Convert to mono
   - Resample to target SR
   - Apply pre-emphasis filter
4. **Duration Normalization**: Pad or truncate to fixed length
5. **Spectrogram Generation**: Compute mel-spectrogram
6. **Image Conversion**: Render and resize to tensor

---

## Output Format

### JSON Response Structure

```json
{
  "prediction": "FAKE",
  "confidence": 0.87,
  "processing_time": 4.2,
  "properties": {
    "duration_seconds": 10.0,
    "sample_rate": 16000,
    "channels": 1
  },
  "pitch": {
    "mean_pitch_hz": 172.8,
    "pitch_stability_score": 0.82
  },
  "energy": {
    "rms_energy": 0.045,
    "silence_ratio": 0.88
  },
  "spectral": {
    "spectral_centroid": 1620.5,
    "spectral_contrast": 21.3
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

- `duration_seconds`: Fixed duration used for analysis
- `sample_rate`: Sample rate used
- `channels`: Number of channels (always 1 after preprocessing)

**Pitch Analysis** (using pYIN algorithm):

- `mean_pitch_hz`: Average fundamental frequency
- `pitch_stability_score`: Consistency of pitch (0-1, higher = more stable)
  - Calculated as: max(0, 1 - std_dev/100)
  - Penalizes high pitch variation

**Energy Analysis**:

- `rms_energy`: Root mean square energy level
- `silence_ratio`: Proportion of non-silent audio (1 - silent_duration/total_duration)

**Spectral Analysis**:

- `spectral_centroid`: Center of mass of spectrum (indicates brightness)
- `spectral_contrast`: Difference between peaks and valleys in spectrum

---

## Architecture Strengths & Limitations

### Strengths

1. **Wavelet Scattering Transform**:
   - Mathematically grounded feature extractor
   - Translation-invariant representations (robust to small shifts)
   - Multi-scale analysis (captures patterns at different resolutions)
   - Multi-orientation analysis (detects directional patterns)
   - No training required for feature extraction
   - Stable features with proven mathematical properties

2. **Fixed Duration Analysis**:
   - Consistent input size for all audio
   - Simplified architecture (no variable-length handling)
   - Predictable processing time
   - Fair comparison across different audio samples

3. **Pre-emphasis Filtering**:
   - Boosts high frequencies important for speech
   - Compensates for natural spectral slope
   - Improves feature extraction quality

4. **Comprehensive Audio Metrics**:
   - Provides interpretable audio characteristics
   - Pitch, energy, and spectral analysis
   - Helps understand audio properties beyond binary prediction

5. **Deep Classifier**:
   - Multiple fully connected layers for pattern learning
   - Batch normalization for training stability
   - Dropout regularization prevents overfitting

### Limitations

1. **Fixed Duration Constraint**:
   - All audio processed as fixed-length segments (e.g., 10 seconds)
   - Truncates longer audio (loses information)
   - Pads shorter audio with zeros (may affect quality)
   - Doesn't leverage full duration of longer samples

2. **Single-Point Analysis**:
   - Analyzes entire duration as one sample
   - No temporal segmentation
   - Cannot detect localized manipulation within longer audio
   - Missing temporal dynamics across audio

3. **Image Conversion Overhead**:
   - Requires rendering spectrogram as image
   - Adds computational cost (matplotlib rendering)
   - May lose some numerical precision in conversion
   - Indirect representation (audio → spectrogram → image → tensor)

4. **No Chunk-Based Analysis**:
   - Unlike MEL-Spectrogram-CNN or STFT-Spectrogram-CNN
   - Cannot analyze very long audio files
   - No temporal resolution within analysis window

5. **Large Fully Connected Layers**:
   - First FC layer contains most parameters
   - Potential overfitting risk
   - Large memory footprint
   - Could benefit from dimensionality reduction before classifier

6. **Scattering Output Dimension**:
   - Scattering transform produces high-dimensional features
   - Increases first FC layer size
   - More parameters to train
   - Longer training times

---

## Technical Deep Dive

### Wavelet Scattering Transform

**Mathematical Foundation**:

The scattering transform is a hierarchical representation that iteratively applies:

1. **Wavelet Convolution**: Convolve with wavelets at different scales and orientations
2. **Modulus**: Take complex magnitude (non-linearity)
3. **Averaging**: Low-pass filter to create translation invariance

**Why Scattering?**

Traditional features (MFCCs, spectrograms):

- May not capture complex patterns
- Can be sensitive to small translations
- Require careful design

Scattering features:

- Capture multi-scale patterns automatically
- Translation-invariant by design
- Mathematically proven stability
- No feature engineering needed

**Kymatio Implementation**:

```python
Scattering2D(J=4, L=8, shape=(256, 256))
```

**Parameters Explained**:

- **J=4**: Number of scales
  - 4 octaves of decomposition
  - Captures patterns at scales: 2⁰, 2¹, 2², 2³
  - Covers both fine details and coarse structures

- **L=8**: Number of orientations
  - 8 directional wavelets
  - Covers 180° (π radians) uniformly
  - Detects patterns at different angles

- **shape=(256, 256)**: Input dimensions
  - Must match spectrogram image size
  - Determines wavelet filter sizes

**Output Dimension Calculation**:

The scattering transform produces features with dimension:

```text
dim = (1 + J*L + J*(J-1)/2*L²) * M
```

Where M is the spatial averaging factor. For J=4, L=8, this produces a high-dimensional feature vector (typically several thousand dimensions).

### Mel-Spectrogram vs Regular Spectrogram

**Why Mel Scale?**

The mel scale is a perceptual scale that mimics human hearing:

```python
mel = 2595 * log10(1 + freq / 700)
```

**Benefits**:

- More resolution at lower frequencies (where humans are more sensitive)
- Less resolution at higher frequencies (where humans are less sensitive)
- More biologically relevant representation
- Better for speech/audio analysis

**Mel Filterbank**:

- Collection of triangular filters
- Spaced linearly in mel scale
- Overlap to create smooth representation
- Reduces dimensionality while preserving perceptual information

### Pre-emphasis Filter

**Implementation**:

```python
y_preemphasized = librosa.effects.preemphasis(y)
```

**Filter Equation**:

```text
y'[n] = y[n] - α*y[n-1]
```

Where α ≈ 0.97 (typical value)

**Frequency Response**:

- High-pass filter
- Boosts high frequencies proportionally
- Flattens spectral envelope

**Why Use It?**

Natural speech/audio has spectral tilt:

- More energy at low frequencies
- Less energy at high frequencies
- Pre-emphasis compensates for this
- Improves feature extraction quality

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

**Design Choices**:

- **Progressive dimensionality reduction**: scattering_dim → 1024 → 512 → 128 → 1
- **Higher dropout early**: 0.4 in first two layers (more regularization)
- **Lower dropout late**: 0.3 in third layer (less aggressive)
- **No dropout in output**: Preserves final prediction quality
- **Batch normalization**: Stabilizes training, allows higher learning rates

### Pitch Analysis (pYIN Algorithm)

**What Is pYIN?**

Probabilistic YIN algorithm for fundamental frequency estimation:

```python
pitch_values, _, _ = librosa.pyin(
    y,
    fmin=librosa.note_to_hz('C2'),  # ~65 Hz
    fmax=librosa.note_to_hz('C7')   # ~2093 Hz
)
```

**How It Works**:

1. Autocorrelation analysis to find periodicity
2. Probabilistic estimation of pitch candidates
3. Viterbi decoding for smooth pitch track

**Output**:

- Array of pitch values (one per frame)
- NaN values where pitch couldn't be detected (silence, noise, etc.)

**Pitch Stability Score**:

```python
pitch_stability = max(0.0, 1.0 - (std_dev / 100.0))
```

- Lower standard deviation → higher stability
- Normalized by dividing by 100 (typical pitch variation range)
- Clamped to [0, 1] range

---

## Integration Example

```python
import torch
from pathlib import Path

# Load the model
model = load_model('SCATTERING-WAVE-V1')

# Analyze an audio file
audio_path = Path("suspicious_audio.mp3")
result = model.analyze(media_path=str(audio_path))

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Processing Time: {result['processing_time']:.2f}s")

# Access pitch analysis
if result['pitch']['mean_pitch_hz']:
    print(f"Mean Pitch: {result['pitch']['mean_pitch_hz']:.1f} Hz")
    print(f"Pitch Stability: {result['pitch']['pitch_stability_score']:.2f}")

# Access energy analysis
print(f"RMS Energy: {result['energy']['rms_energy']:.4f}")
print(f"Silence Ratio: {result['energy']['silence_ratio']:.2f}")

# Access spectral characteristics
print(f"Spectral Centroid: {result['spectral']['spectral_centroid']:.1f} Hz")
print(f"Spectral Contrast: {result['spectral']['spectral_contrast']:.1f} dB")

# Access visualization
if result['visualization']['spectrogram_url']:
    print(f"Spectrogram saved: {result['visualization']['spectrogram_url']}")
```

### Analyzing Video Files

```python
# The model can extract audio from video files
video_path = Path("suspicious_video.mp4")
result = model.analyze(media_path=str(video_path))

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### With Progress Events

```python
# For asynchronous processing with progress updates
result = model.analyze(
    media_path=str(audio_path),
    video_id="unique_media_id",
    user_id="user_id"
)

# Progress events will be published:
# - AUDIO_EXTRACTION_START
# - AUDIO_EXTRACTION_COMPLETE
# - SPECTROGRAM_GENERATION_START
# - SPECTROGRAM_GENERATION_COMPLETE
# - FRAME_ANALYSIS_PROGRESS
# - ANALYSIS_COMPLETE (or ANALYSIS_FAILED)
```
