# Audio-MFT-CRNN Model

| Attribute                    | Details                                                 |
| :--------------------------- | :------------------------------------------------------ |
| **Model Category**           | Audio Analysis                                          |
| **Model Type**               | Audio Deepfake Detector                                 |
| **Version**                  | V1                                                      |
| **Primary Detection Target** | Synthetic or manipulated audio, including voice clones. |

---

## 1. Overview

### What Is This Model?

The `AUDIO-MFT-CRNN-V1` is a sophisticated audio deepfake detector that uses a **Multi-Feature Temporal (MFT)** approach. Instead of relying on a single representation of the audio, it extracts a rich vector of seven distinct acoustic features for each moment in time. This feature vector is then analyzed by a powerful hybrid **Convolutional Recurrent Neural Network (CRNN)** architecture.

### The Core Concept

The model's philosophy is that different types of audio manipulation leave behind different kinds of artifacts. By analyzing the audio from multiple perspectives simultaneously—its pitch, energy, timbre, and phonetic structure—the model has a much higher chance of detecting an anomaly. The CRNN architecture first finds local, cross-feature patterns and then analyzes how these patterns evolve over time to spot unnatural sequences.

### Why a Multi-Feature Vector Approach?

- **Rich Context:** Provides a far more detailed "fingerprint" of the audio than a single spectrogram.
- **Robustness:** A deepfake might successfully mimic one feature (like pitch) but fail to replicate another (like spectral texture). Analyzing multiple features makes the model harder to fool.
- **Specialization:** Each feature is chosen to target a specific characteristic of human speech that is difficult for AI to generate perfectly.

---

## 2. How It Works

### Step-by-Step Process

#### Phase 1: Audio Preprocessing

```text
Input Audio → Standardization → Mono Conversion → Resampling (16kHz)
```

This initial phase is identical to other audio models, ensuring the raw audio is in a consistent format before feature extraction.

#### Phase 2: Audio Chunking (Overlapping)

```text
Audio Waveform → Split into Overlapping Chunks → Process Each Chunk
```

The audio is segmented into 1-second chunks with a 0.5-second overlap. This overlapping strategy ensures that patterns at the boundary of a chunk are fully captured in the next one, providing better temporal continuity for the model.

**Example** (3s audio, 1s chunks, 0.5s overlap):

```text
Chunk 1: [0.0s - 1.0s]
Chunk 2: [0.5s - 1.5s]
Chunk 3: [1.0s - 2.0s]
Chunk 4: [1.5s - 2.5s]
Chunk 5: [2.0s - 3.0s]
```

#### Phase 3: Per-Chunk Multi-Feature Extraction

For each 1-second chunk, a comprehensive set of features is extracted to form a feature vector.

- **Mel Spectrogram:** Captures the unique "texture" and timbre of a voice.
- **STFT Spectrograms (Wide & Narrow-band):** Provide two complementary views of the frequency content, one focused on timing and the other on frequency detail.
- **MFCCs (Mel-Frequency Cepstral Coefficients):** Represents the core phonetic sounds of speech. Fakes often struggle with natural transitions between these sounds.
- **Chroma Features:** Tracks the musical intonation or "melody" of speech. Fake voices can sound robotic or have unnatural pitch changes.
- **Spectral Contrast:** Measures the clarity of the audio. Synthetic speech might be overly smooth or have strange noisy peaks.
- **Zero-Crossing Rate (ZCR):** Measures "noisiness" or breathiness, particularly for fricative sounds like 's'.

#### Phase 4: Vector Stacking

```text
[Mel] + [STFT-W] + [STFT-N] + [MFCC] + [Chroma] + ... → Stacked Feature Tensor
```

All the 2D feature matrices from Phase 3 are stacked vertically into a single, large 2D tensor. This tensor represents a rich, multi-faceted snapshot of the 1-second audio chunk.

#### Phase 5: CRNN Classification

The stacked feature tensor is fed into the **Convolutional Recurrent Neural Network (CRNN)**, which works in two stages:

1.  **CNN Feature Extraction:** The convolutional layers scan the stacked tensor to find small, significant local patterns. For example, it might learn that a specific shape in the Mel spectrogram that co-occurs with a particular MFCC value is a strong indicator of a real voice.
2.  **LSTM Sequence Analysis:** The sequence of patterns identified by the CNN is then fed into a **bidirectional LSTM**. The LSTM's job is to analyze the temporal flow of these patterns over the 1-second duration. It learns the expected sequence of sounds in natural speech and can detect when this sequence is illogical, jittery, or "feels" synthetic.

#### Phase 6: Aggregation & Final Verdict

```text
Per-Chunk Predictions → Majority Vote → Final Classification
```

The model makes a `REAL` or `FAKE` prediction for each overlapping chunk. The final verdict for the entire audio file is determined by a **majority vote** of all chunk predictions.

**Example:**

```text
- 5 Chunks Analyzed
- Predictions: [FAKE, FAKE, REAL, FAKE, REAL]
- Vote Count: FAKE: 3, REAL: 2
- Final Verdict: FAKE (Confidence: 3/5 = 60%)
```

---

## 3. Architecture Details

### Model Structure

```text
┌─────────────────────────────────────────────────────────────┐
│                      Input Audio File                        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Audio Preprocessing & Chunking                  │
│   • Mono, 16kHz, Overlapping 1-second chunks                 │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Per-Chunk: Multi-Feature Extraction                │
│   • Mel Spectrogram, STFTs, MFCCs, Chroma, etc.              │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Per-Chunk: Vector Stacking                           │
│   • All feature matrices are vertically stacked (np.vstack)  │
│   • Result: A single, tall tensor (e.g., [1, 441, 32])       │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              CNN Feature Extraction                          │
│   • 4 Conv Blocks (Conv2D -> ReLU -> BatchNorm -> MaxPool)   │
│   • Learns local, cross-feature patterns                     │
│   • Output: Sequence of feature vectors over time            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              LSTM Temporal Analysis                          │
│   • 2-layer, Bidirectional LSTM with Dropout                 │
│   • Analyzes the sequence of features from the CNN           │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Fully Connected Classifier                      │
│   • Concatenates final forward/backward LSTM states          │
│   • FC(256 -> 64) -> ReLU -> Dropout -> FC(64 -> 1)          │
│   • Output: A single logit for the chunk                     │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Aggregation Across Chunks                       │
│   • Apply Sigmoid to get probability for each chunk          │
│   • Majority vote determines final prediction                │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Final Prediction: REAL or FAKE                       │
│         + Confidence Score + Audio Metrics                   │
└─────────────────────────────────────────────────────────────┘
```

### CRNN Architecture Breakdown

1.  **CNN Extractor**: A standard 4-layer CNN that reduces the feature dimension while preserving the time dimension.
2.  **LSTM Layer**: A 2-layer, bidirectional LSTM with a hidden size of 128. The bidirectional nature allows it to learn patterns from both past and future context within each 1-second chunk.
3.  **Classifier**: A simple 2-layer feed-forward network with dropout for regularization.

**Dynamic Input Size:** The model's input layer is **dynamically sized** based on the features enabled in `config.yaml`. The `_get_lstm_input_size` method performs a "dummy" forward pass on a sample tensor to calculate the exact feature dimension after the CNN layers, making the architecture flexible to configuration changes.

---

## 4. Input Requirements

### Audio Specifications

| Parameter       | Requirement          | Notes                                                        |
| :-------------- | :------------------- | :----------------------------------------------------------- |
| **Format**      | MP3, WAV, FLAC, etc. | Common audio formats are supported via `pydub`.              |
| **Sample Rate** | Any                  | Automatically resampled to **16kHz** internally.             |
| **Duration**    | > 1 second           | Must be long enough to generate at least one 1-second chunk. |
| **Channels**    | Any                  | Automatically converted to mono (1 channel).                 |

### Configuration Parameters (`config.yaml`)

This model is highly configurable. The key parameters are:

- **`preprocessing_params`**:
  - `sampling_rate`: Target sample rate (16000).
  - `chunk_duration_s`: Length of each audio chunk (1.0).
  - `chunk_overlap_s`: Overlap between chunks (0.5).
- **`feature_extraction`**: Boolean flags and parameters (`n_mfcc`, `n_chroma`, etc.) to control which features are generated and included in the feature vector.
- **`combined_vector_model`**: Technical parameters for feature generation, such as FFT sizes (`mel_n_fft`, `stft_n_fft_wide`) and the `shared_hop_length`.

---

## 5. Output Format

The model returns the standard `AudioAnalysisResult` schema, ensuring consistency with other audio detectors.

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
    "pitch_stability_score": null
  },
  "energy": {
    "rms_energy": 0.038,
    "silence_ratio": 0.0
  },
  "spectral": {
    "spectral_centroid": 1580.2,
    "spectral_contrast": 0.0
  },
  "visualization": null
}
```

---

## 6. Architecture Strengths & Limitations

### Strengths

1.  **Rich Feature Set:** By combining multiple acoustic features, the model gets a comprehensive view of the audio, making it robust against different types of manipulation.
2.  **Temporal Modeling:** The use of an LSTM allows the model to learn the expected sequence of audio patterns, a critical element for detecting unnatural speech.
3.  **Hybrid Architecture:** The CRNN architecture effectively combines the spatial feature extraction power of CNNs with the sequential modeling power of RNNs.
4.  **Overlapping Chunks:** The overlapping analysis strategy ensures that no temporal information is lost at the boundaries of chunks.

### Limitations

1.  **Simple Aggregation:** The final verdict is based on a simple majority vote, which gives equal weight to every chunk. It does not account for the confidence of each chunk's prediction.
2.  **No Long-Range Context:** The LSTM analyzes patterns _within_ each 1-second chunk but has no memory of the previous chunk. It cannot detect inconsistencies that span several seconds.
3.  **Complexity:** Extracting a large vector of features for every chunk is computationally more intensive than generating a single spectrogram.

---

## 7. Integration Example

### CLI Usage

The model can be run directly from the command line for fast analysis.

```bash
# Ensure the model is listed in the .env ACTIVE_MODELS
# Then run the analyze command:

drishtiksha analyze /path/to/your_audio.wav --model AUDIO-MFT-CRNN-V1
```

This will process the audio file and print a summary of the results to the console, including the per-chunk predictions and the final verdict.
