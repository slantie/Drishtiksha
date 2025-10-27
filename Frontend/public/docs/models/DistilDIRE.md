# DistilDIRE Model

**Model Category**: Image Analysis  
**Model Type**: Image Deepfake Detector  
**Version**: V1  
**Primary Detection Target**: AI-generated and manipulated images (Diffusion model artifacts)

---

## Overview

### What Is This Model?

DistilDIRE (Distilled Diffusion Reconstruction Error) is an image deepfake detector that analyzes the reconstruction error patterns produced by diffusion models. The model uses a pre-trained diffusion model (ADM - Ablated Diffusion Model) to generate a "noise map" representing how well the image can be reconstructed, then feeds both the original image and this noise map into a ResNet-50 detector to classify the image as real or fake.

### The Core Concept

**DIRE (Diffusion Reconstruction Error)** is based on a key insight:

- **Real images** have characteristic noise patterns when processed through diffusion models
- **AI-generated images** (especially from diffusion models) have different noise signatures
- **Manipulated images** show inconsistent noise patterns

By analyzing the diffusion model's reconstruction error (the "epsilon" or noise map), the detector can identify artifacts invisible to the human eye.

### Key Innovation

**Two-Stage Architecture**:

1. **ADM (Teacher)**: Pre-trained diffusion model generates DIRE noise map
2. **DistilDIRE (Student)**: Lightweight ResNet-50 analyzes 6-channel input (RGB + noise map)

This knowledge distillation approach provides:

- Rich diffusion-based features from ADM
- Efficient detection from compact ResNet-50
- Best of both worlds: accuracy + speed

---

## How It Works

### Step-by-Step Process

#### Phase 1: Image Preprocessing

```text
Input Image → Load → Convert RGB → Resize & Center Crop → Normalize to [-1, 1]
```

**Preprocessing Steps**:

1. **Load Image**:

   ```python
   image = Image.open(media_path).convert("RGB")
   ```

2. **Transform**:

   ```python
   transforms.Compose([
       Resize(256, antialias=True),      # Resize to 256×256
       CenterCrop(256)                    # Center crop
   ])
   ```

3. **Normalize to [-1, 1]**:

   ```python
   img_tensor = TF.to_tensor(image) * 2 - 1
   # From [0, 1] to [-1, 1] range
   ```

**Output**: (1, 3, 256, 256) tensor in [-1, 1] range

#### Phase 2: DIRE Noise Map Generation (ADM)

```text
Normalized Image → ADM Diffusion Model → DDIM Reverse Sampling → Epsilon (Noise Map)
```

**What Is ADM?**

Ablated Diffusion Model:

- Pre-trained unconditional diffusion model
- Trained on large-scale image datasets
- Can reconstruct/denoise images
- 256×256 resolution

**DDIM Reverse Sampling**:

```python
t = torch.zeros(batch_size).long()  # Start at t=0 (clean image)

eps = diffusion.ddim_reverse_sample_only_eps(
    model=adm_model,
    x=img_tensor,        # Input image
    t=t,                 # Time step = 0
    clip_denoised=True,  # Clip predictions to [-1, 1]
    eta=0.0              # Deterministic (no randomness)
)
```

**What Is Epsilon (ε)?**

The epsilon tensor represents:

- The "noise" that would be added to the image at diffusion time t
- Reconstruction error of the diffusion model
- Artifacts and inconsistencies in the image

**Why Time Step t=0?**

- t=0 is the "clean image" time step
- First-step noise reveals immediate reconstruction patterns
- Captures subtle artifacts without heavy diffusion

**Output**: (1, 3, 256, 256) epsilon tensor (noise map)

#### Phase 3: Visualization Generation (Optional)

```text
Epsilon Tensor → Normalize to [0, 255] → Transpose to HWC → Save as PNG
```

**Visualization Process**:

```python
# Normalize epsilon to [0, 1]
normalized = (eps - eps.min()) / (eps.max() - eps.min() + 1e-6)

# Convert to uint8 image
image_to_save = (normalized * 255).astype(np.uint8)

# Save as PNG
Image.fromarray(image_to_save).save(path)
```

**When Generated**:

- Only if `generate_visualizations=True` parameter is set
- Saved to temporary file
- Included in response

**Purpose**:

- Visual inspection of DIRE noise patterns
- Debugging and analysis
- Understanding model decisions

#### Phase 4: Channel Concatenation

```text
RGB Image Tensor (3 channels) + Epsilon Tensor (3 channels) → Combined (6 channels)
```

**Concatenation**:

```python
combined_input = torch.cat([img_tensor, eps_tensor], dim=1)
# Shape: (1, 6, 256, 256)
# Channels: [R, G, B, ε_R, ε_G, ε_B]
```

**Why 6 Channels?**

- First 3 channels: Original RGB image (appearance)
- Last 3 channels: DIRE noise map (reconstruction error)
- Model learns to correlate appearance with noise patterns

#### Phase 5: ResNet-50 Detection

```text
6-Channel Input → Modified ResNet-50 → Feature Extraction → Classification
```

**Modified ResNet-50 Architecture**:

**Input Layer Modification**:

```python
# Standard ResNet-50: Conv2D(3, 64, kernel=7, stride=2)
# Modified for 6 channels:
conv1 = Conv2D(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
```

**Backbone** (standard ResNet-50 layers):

```python
conv1 → bn1 → relu → maxpool →
layer1 (3 bottleneck blocks) →
layer2 (4 bottleneck blocks) →
layer3 (6 bottleneck blocks) →
layer4 (3 bottleneck blocks)
```

**Feature Extraction**:

```python
feature = backbone(combined_input)  # Shape: (1, 2048, H', W')
```

**Classification Head**:

```python
pooled = AdaptiveAvgPool2d(1)(feature)  # (1, 2048, 1, 1)
flattened = Flatten()(pooled)            # (1, 2048)
logit = Linear(2048, 1)(flattened)       # (1, 1)
```

**Output**:

- **logit**: Raw classification score
- **feature**: Feature map for heatmap generation

#### Phase 6: Probability & Decision

```text
Logit → Sigmoid → Probability → Threshold (0.5) → Prediction
```

**Probability Calculation**:

```python
prob_fake = sigmoid(logit)

if prob_fake >= 0.5:
    prediction = "FAKE"
    confidence = prob_fake
else:
    prediction = "REAL"
    confidence = 1 - prob_fake
```

#### Phase 7: Heatmap Generation

```text
Feature Map → Channel-wise Average → Resize to Original → Normalize → Heatmap
```

**Heatmap Process**:

```python
# Average across channels
heatmap = mean(feature_map, dim=1)  # (H', W')

# Resize to original image size
heatmap_resized = cv2.resize(heatmap, (width, height))

# Normalize to [0, 1]
heatmap_normalized = (heatmap - min) / (max - min + 1e-6)
```

**Purpose**:

- Spatial attention map
- Shows which regions contributed to decision
- Helps identify manipulated areas

---

## Architecture Details

### Model Structure

```text
┌─────────────────────────────────────────────────────────────┐
│                      Input Image                             │
│                   (any size, RGB)                            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Preprocessing                                   │
│   • Resize to 256×256                                        │
│   • Center crop                                              │
│   • Normalize to [-1, 1]                                     │
│   → Tensor: (1, 3, 256, 256)                                │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         ADM (Ablated Diffusion Model) - TEACHER              │
│   Pre-trained diffusion model (256×256)                      │
│                                                              │
│   DDIM Reverse Sampling (t=0):                               │
│   • Input: Normalized image                                  │
│   • Output: Epsilon (noise) tensor                           │
│   • Represents reconstruction error                          │
│                                                              │
│   → Epsilon Tensor: (1, 3, 256, 256)                        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Optional: Visualization Generation                   │
│   If generate_visualizations=True:                           │
│   • Normalize epsilon to [0, 255]                            │
│   • Save as PNG image                                        │
│   • Include path in response                                 │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Channel Concatenation                                │
│   RGB Image ⊕ Epsilon Map → 6-channel input                 │
│   Channels: [R, G, B, ε_R, ε_G, ε_B]                        │
│   → Combined: (1, 6, 256, 256)                              │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         DistilDIRE Detector - STUDENT                        │
│         Modified ResNet-50                                   │
│                                                              │
│   Input Layer (modified):                                    │
│   Conv2D(6→64, k=7, s=2, p=3) → BN → ReLU → MaxPool        │
│                                                              │
│   Backbone (standard ResNet-50):                             │
│   • Layer 1: 3 bottleneck blocks (64→256 channels)         │
│   • Layer 2: 4 bottleneck blocks (128→512 channels)        │
│   • Layer 3: 6 bottleneck blocks (256→1024 channels)       │
│   • Layer 4: 3 bottleneck blocks (512→2048 channels)       │
│                                                              │
│   → Feature Map: (1, 2048, H', W')                          │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Classification Head                                  │
│   AdaptiveAvgPool2D(1) → Flatten → Linear(2048→1)          │
│   → Logit: (1, 1)                                            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Sigmoid & Threshold                                  │
│   Probability = sigmoid(logit)                               │
│   Prediction = "FAKE" if prob ≥ 0.5 else "REAL"            │
│   Confidence = prob if FAKE else (1-prob)                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Heatmap Generation                                   │
│   • Average feature map across channels                      │
│   • Resize to original image dimensions                      │
│   • Normalize to [0, 1]                                      │
│   → Spatial attention map                                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Final Result                                         │
│   • Prediction: REAL or FAKE                                 │
│   • Confidence score                                         │
│   • Heatmap scores (spatial)                                 │
│   • Optional: DIRE visualization                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **ADM (Teacher Model)** - Pre-trained, frozen
   - Unconditional diffusion model
   - 256×256 resolution
   - Generates DIRE noise maps
   - ~280M parameters (not trainable)

2. **DistilDIRE (Student Model)** - Trainable
   - Modified ResNet-50 backbone
   - 6-channel input layer
   - Classification head
   - ~25M parameters (trainable)

**Total Model Size**: ~305M parameters  
**Trainable Parameters**: ~25M (DistilDIRE only)  
**Frozen Parameters**: ~280M (ADM only)

### ResNet-50 Bottleneck Architecture

**Bottleneck Block**:

```python
# For each block:
Conv2D(in → in/4, k=1) → BN → ReLU →
Conv2D(in/4 → in/4, k=3, p=1) → BN → ReLU →
Conv2D(in/4 → in, k=1) → BN →
+ residual_connection → ReLU
```

**Layer Configuration**:

| Layer | Blocks | Output Channels | Output Size |
|-------|--------|-----------------|-------------|
| conv1 | - | 64 | 128×128 |
| layer1 | 3 | 256 | 64×64 |
| layer2 | 4 | 512 | 32×32 |
| layer3 | 6 | 1024 | 16×16 |
| layer4 | 3 | 2048 | 8×8 |

---

## Input Requirements

### Image Specifications

| Parameter | Requirement | Notes |
|-----------|-------------|-------|
| **Format** | JPG, PNG, BMP, etc. | Common image formats |
| **Resolution** | Any | Resized to 256×256 internally |
| **Aspect Ratio** | Any | Center cropped after resize |
| **Color Mode** | RGB | Converted to RGB automatically |
| **File Size** | Any | Limited by system memory |

### Configuration Parameters

**Model Paths**:

- **model_path**: Path to DistilDIRE weights (.pt file)
- **adm_model_path**: Path to ADM weights (.pt file)

**Preprocessing**:

- **image_size**: Target size (default: 256)

**ADM Configuration**:

- **clip_denoised**: Clip predictions to [-1, 1] (default: True)
- **use_fp16**: Use half-precision for ADM (default: False)

**Visualization**:

- **generate_visualizations**: Generate DIRE map PNG (default: False)

---

## Output Format

### JSON Response Structure

```json
{
  "prediction": "FAKE",
  "confidence": 0.87,
  "processing_time": 3.2,
  "dimensions": {
    "width": 1920,
    "height": 1080
  },
  "heatmap_scores": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...],
  "visualization_path": "/tmp/dire_map_xyz.png"
}
```

### Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | string | "REAL" or "FAKE" classification |
| `confidence` | float | Confidence score (0-1) |
| `processing_time` | float | Total analysis time in seconds |
| `dimensions` | object | Original image dimensions |
| `heatmap_scores` | array | 2D spatial attention map (normalized) |
| `visualization_path` | string | Path to DIRE visualization (if generated) |

### Heatmap Scores

- 2D array matching original image dimensions
- Values in [0, 1] range
- Higher values = more important regions for decision
- Can be overlaid on original image for visualization

### Visualization Path

- Only present if `generate_visualizations=True`
- Path to PNG file showing DIRE noise map
- Temporary file (cleanup required)
- RGB visualization of epsilon tensor

---

## Architecture Strengths & Limitations

### Strengths

1. **Diffusion-Based Detection**:
   - Leverages powerful diffusion model representations
   - Captures subtle artifacts invisible to humans
   - Effective against diffusion-generated images
   - Novel approach to deepfake detection

2. **Knowledge Distillation**:
   - Combines heavy teacher (ADM) with lightweight student (ResNet-50)
   - Inherits ADM's detection capability
   - Efficient inference (only student needed at runtime)
   - Best of both worlds: accuracy + speed

3. **DIRE Noise Analysis**:
   - Analyzes reconstruction error patterns
   - Reveals inconsistencies in generation process
   - Works even when visual quality is high
   - Robust to post-processing

4. **Spatial Heatmaps**:
   - Provides interpretability
   - Shows manipulated regions
   - Helps understand model decisions
   - Useful for forensic analysis

5. **6-Channel Architecture**:
   - Jointly analyzes appearance and noise
   - Learns correlations between RGB and epsilon
   - More informative than RGB-only
   - Captures multi-modal features

6. **Optional Visualization**:
   - Can generate DIRE maps on demand
   - Useful for debugging and analysis
   - No overhead when not needed
   - Helps understand detection rationale

### Limitations

1. **Computational Cost**:
   - Requires running ADM diffusion model
   - DDIM reverse sampling is expensive
   - Two models in pipeline (ADM + DistilDIRE)
   - Slower than single-model detectors

2. **Fixed Resolution**:
   - ADM operates at 256×256
   - All images resized and cropped
   - Loss of detail for high-resolution images
   - Cannot analyze at multiple scales

3. **ADM Dependency**:
   - Requires large pre-trained diffusion model
   - ADM must be loaded into memory
   - ~280M additional parameters
   - High memory footprint

4. **Single Image Only**:
   - No video support
   - Cannot analyze temporal consistency
   - Missing sequential information
   - Limited to static images

5. **Diffusion Model Bias**:
   - Detection depends on ADM's training data
   - May not generalize to all image types
   - Biased toward diffusion-generated fakes
   - Less effective on non-diffusion fakes

6. **No Face Detection**:
   - Analyzes entire image uniformly
   - Doesn't focus on faces specifically
   - Background noise may affect results
   - No region-specific analysis

---

## Technical Deep Dive

### Diffusion Models Background

**What Are Diffusion Models?**

Generative models that learn to denoise images through a gradual process:

1. **Forward Process** (add noise):

   ```text
   x₀ → x₁ → x₂ → ... → xₜ → ... → x_T
   (clean)              (noise step)   (pure noise)
   ```

2. **Reverse Process** (denoise):

   ```text
   x_T → x_{T-1} → ... → x₁ → x₀
   (pure noise)            (clean image)
   ```

**Why Use for Detection?**

- Diffusion models learn rich representations of natural images
- Can reconstruct real images with low error
- Fake images have different noise characteristics
- Reconstruction error reveals manipulation

### DDIM Reverse Sampling

**DDIM (Denoising Diffusion Implicit Models)**:

Deterministic sampling algorithm for diffusion models:

```python
# Standard reverse step
x_{t-1} = √(α_{t-1}) * predicted_x₀ + √(1-α_{t-1}) * ε

# At t=0 (our case)
ε = (x₀ - √(α₀) * x₀) / √(1-α₀)
```

**Why t=0?**

- First reverse step captures immediate artifacts
- No cumulative denoising errors
- Faster computation (single step)
- Sufficient for detection

**What Is Epsilon?**

The "noise" that would be added to create x₁ from x₀:

```python
x₁ = √(α₁) * x₀ + √(1-α₁) * ε
```

For real images: ε follows learned distribution  
For fake images: ε shows anomalies

### ResNet-50 Input Modification

**Standard ResNet-50**:

```python
Conv2D(3, 64, kernel_size=7, stride=2, padding=3)
```

**Modified for 6 Channels**:

```python
Conv2D(6, 64, kernel_size=7, stride=2, padding=3)
```

**Weight Initialization**:

When loading pre-trained weights:

- First 3 channels: Use ImageNet weights
- Last 3 channels: Initialize randomly or duplicate RGB weights

**Why This Works?**

- ResNet-50 can still extract RGB features
- Additional channels learn epsilon patterns
- Early fusion of appearance + noise
- End-to-end trainable

### Heatmap Generation Mathematics

**Feature Map Aggregation**:

```python
# Feature map: (B, C, H, W) = (1, 2048, 8, 8)
# Average across channels
heatmap = mean(feature_map, dim=1)  # (1, 8, 8)

# Resize to original dimensions
heatmap_resized = interpolate(heatmap, size=(H_orig, W_orig))

# Normalize
min_val = heatmap.min()
max_val = heatmap.max()
heatmap_normalized = (heatmap - min_val) / (max_val - min_val + 1e-6)
```

**Interpretation**:

- High values: Important for classification
- Low values: Less important regions
- Can identify manipulated areas
- Not perfect localization (due to pooling)

### Knowledge Distillation Concept

**Teacher-Student Framework**:

**Teacher (ADM)**:

- Large, powerful model
- Expensive to run
- Provides rich features (epsilon)

**Student (DistilDIRE)**:

- Smaller, efficient model
- Fast inference
- Learns to use teacher's features

**Distillation Process** (during training):

```python
# Teacher generates epsilon
eps = ADM(image)

# Student learns from RGB + epsilon
combined = concat([image, eps], dim=1)
logit = DistilDIRE(combined)

# Loss combines:
# 1. Classification loss (real/fake labels)
# 2. Feature matching loss (match teacher features)
```

**Inference** (our case):

```python
# Teacher still needed to generate epsilon
eps = ADM(image)

# Student does classification
combined = concat([image, eps], dim=1)
prediction = DistilDIRE(combined)
```

---

## Integration Example

```python
import torch
from pathlib import Path

# Load the model
model = load_model('DISTILDIRE-V1')

# Analyze an image (no visualization)
image_path = Path("suspicious_image.jpg")
result = model.analyze(media_path=str(image_path))

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Processing Time: {result['processing_time']:.2f}s")
print(f"Image Size: {result['dimensions']['width']}×{result['dimensions']['height']}")

# Analyze with DIRE visualization
result_with_viz = model.analyze(
    media_path=str(image_path),
    generate_visualizations=True
)

if result_with_viz.get('visualization_path'):
    print(f"\nDIRE Map saved to: {result_with_viz['visualization_path']}")
    
    # Display the DIRE map
    import matplotlib.pyplot as plt
    from PIL import Image
    
    dire_map = Image.open(result_with_viz['visualization_path'])
    plt.imshow(dire_map)
    plt.title("DIRE Noise Map")
    plt.axis('off')
    plt.show()

# Access heatmap for visualization
if result['heatmap_scores']:
    import numpy as np
    
    heatmap = np.array(result['heatmap_scores'])
    
    # Overlay on original image
    original = Image.open(image_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title("Attention Heatmap")
    axes[2].imshow(original)
    axes[2].imshow(heatmap, cmap='jet', alpha=0.5)
    axes[2].set_title("Overlay")
    plt.show()
```

### Batch Processing

```python
from pathlib import Path

# Process directory of images
image_dir = Path("images/")
results = []

for image_path in image_dir.glob("*.jpg"):
    result = model.analyze(
        media_path=str(image_path),
        generate_visualizations=False  # Faster without viz
    )
    
    results.append({
        'filename': image_path.name,
        'prediction': result['prediction'],
        'confidence': result['confidence']
    })

# Find most suspicious images
fake_images = [r for r in results if r['prediction'] == 'FAKE']
fake_images_sorted = sorted(
    fake_images,
    key=lambda x: x['confidence'],
    reverse=True
)

print(f"Found {len(fake_images)} suspicious images:")
for item in fake_images_sorted[:10]:
    print(f"  {item['filename']}: {item['confidence']:.2%}")
```
