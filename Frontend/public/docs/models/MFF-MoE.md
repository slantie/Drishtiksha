# MFF-MoE (V1)

## Overview

**MFF-MoE** (Multi-scale Feature Fusion Mixture-of-Experts) is an ensemble-based deepfake detection model that combines predictions from **7 expert networks** using a Mixture-of-Experts (MoE) architecture. The model leverages diverse backbone architectures (ConvNeXt and EfficientNet variants) to capture multi-scale features.

**Model Type:** Image & Video Analysis  
**Primary Use:** Ensemble-based deepfake detection  
**Architecture:** 7-Expert MoE (ConvNeXt-V2 + EfficientNet B4/B5/B6)  

### Key Characteristics

- **Multi-Expert Ensemble**: Combines 7 independently trained expert networks
- **Diverse Backbones**: Uses ConvNeXt-V2 and EfficientNet (B4, B5, B6) variants
- **Exponential Moving Average (EMA)**: Each expert maintains EMA weights for inference
- **Multi-Modal Support**: Handles both single images and video frame sequences
- **Averaging Fusion**: Simple mean aggregation across expert predictions

---

## How It Works

### Pipeline Overview

```text
Input Image/Video Frame
    ↓
Preprocessing (Resize 512×512, Normalize)
    ↓
Parallel Processing by 7 Experts:
    ├─ Expert 0: ConvNeXt-V2-Tiny
    ├─ Expert 1: ConvNeXt-V2-Tiny (duplicate)
    ├─ Expert 2: EfficientNet-B4
    ├─ Expert 3: EfficientNet-B4 (duplicate)
    ├─ Expert 4: EfficientNet-B5
    ├─ Expert 5: EfficientNet-B5 (duplicate)
    └─ Expert 6: EfficientNet-B6
    ↓
Each Expert:
    - Forward Features → Global Pooling → Classifier
    - Output: Softmax probability [P(real), P(fake)]
    ↓
Extract P(fake) from each expert
    ↓
Mean Aggregation: (P₀ + P₁ + ... + P₆) / 7
    ↓
Final Prediction (REAL if <0.5, FAKE if ≥0.5)
```

### Inference Strategy

1. **Image Analysis**
   - Load image as PIL RGB
   - Apply transformation: ToTensor → Normalize → Resize(512×512)
   - Forward through all 7 experts (using EMA weights)
   - Average the 7 fake probabilities
   - Return prediction and confidence

2. **Video Analysis**
   - Extract up to 100 frames (evenly spaced)
   - Process each frame independently
   - Aggregate frame-level predictions (simple mean)
   - Return final video prediction and frame-by-frame results

---

## Architecture Details

### Component Breakdown

#### 1. Expert Network Configuration

**Expert Roster (7 experts total):**

| Expert Index | Backbone Architecture | Parameters | Feature Dim | Pretrained Weights |
|--------------|----------------------|------------|-------------|-------------------|
| 0            | ConvNeXt-V2-Tiny     | ~28M       | 768         | FCMAE (ImageNet-1K) |
| 1            | ConvNeXt-V2-Tiny     | ~28M       | 768         | FCMAE (ImageNet-1K) |
| 2            | EfficientNet-B4      | ~19M       | 1792        | JFT-300M (Noisy Student) |
| 3            | EfficientNet-B4      | ~19M       | 1792        | JFT-300M (Noisy Student) |
| 4            | EfficientNet-B5      | ~30M       | 2048        | JFT-300M (Noisy Student) |
| 5            | EfficientNet-B5      | ~30M       | 2048        | JFT-300M (Noisy Student) |
| 6            | EfficientNet-B6      | ~43M       | 2304        | JFT-300M (Noisy Student) |

**Total Ensemble Parameters**: ~195M (sum of all experts)

**Note**: Experts 0-1, 2-3, and 4-5 use identical architectures but have independently trained weights, providing ensemble diversity through different initialization and training trajectories.

#### 2. ConvNeXt-V2-Tiny Expert

```python
class MFF_Expert_Convnext(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.name = 'Effnet'  # Legacy name
        self.baseline_extractor = timm.create_model(
            'convnextv2_tiny.fcmae_ft_in1k',
            pretrained=pretrained,
            num_classes=2
        )
    
    def forward(self, x):
        # Feature extraction
        x = self.baseline_extractor.forward_features(x)  # [B, 768, H/32, W/32]
        
        # Global average pooling
        feat = self.baseline_extractor.head.global_pool(x)[:, :, 0, 0]  # [B, 768]
        
        # Classification head
        x = self.baseline_extractor.head(x)  # [B, 2]
        
        return x, feat  # (logits, features)
```

**ConvNeXt-V2-Tiny Architecture:**

```text
Stem: Conv2D(3 → 96, 4×4, stride=4)
    ↓
Stage 1: 3 × ConvNeXtBlock (C=96)    # Output: [B, 96, H/4, W/4]
    ↓
Downsample: LayerNorm → Conv2D(96 → 192, 2×2, stride=2)
    ↓
Stage 2: 3 × ConvNeXtBlock (C=192)   # Output: [B, 192, H/8, W/8]
    ↓
Downsample: LayerNorm → Conv2D(192 → 384, 2×2, stride=2)
    ↓
Stage 3: 9 × ConvNeXtBlock (C=384)   # Output: [B, 384, H/16, W/16]
    ↓
Downsample: LayerNorm → Conv2D(384 → 768, 2×2, stride=2)
    ↓
Stage 4: 3 × ConvNeXtBlock (C=768)   # Output: [B, 768, H/32, W/32]
    ↓
Global Average Pooling               # Output: [B, 768]
    ↓
LayerNorm → Linear(768 → 2)          # Classification head
```

**ConvNeXtBlock Structure:**

```python
# ConvNeXt V2 Block
x_input = x  # [B, C, H, W]

# 1. Depthwise Convolution (7×7)
x = DepthwiseConv2D(C, 7×7, padding=3)(x)

# 2. LayerNorm + Linear (1×1 Conv)
x = LayerNorm()(x)
x = Linear(C → 4C)(x)  # Expansion

# 3. GELU Activation
x = GELU()(x)

# 4. Global Response Normalization (GRN) - NEW in V2
x = GRN()(x)

# 5. Linear Projection (1×1 Conv)
x = Linear(4C → C)(x)  # Contraction

# 6. Residual Connection
x = x_input + x
```

#### 3. EfficientNet Expert (B4/B5/B6)

```python
class MFF_Expert_EffB4(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.name = 'Effnet'
        self.baseline_extractor = timm.create_model(
            'tf_efficientnet_b4.ns_jft_in1k',
            pretrained=pretrained,
            num_classes=2
        )
    
    def forward(self, x):
        # Feature extraction
        x = self.baseline_extractor.forward_features(x)  # [B, 1792, H/32, W/32]
        
        # Global pooling
        x = self.baseline_extractor.global_pool(x)  # [B, 1792]
        
        # Dropout (if enabled)
        if self.baseline_extractor.drop_rate > 0.:
            x = F.dropout(x, p=self.baseline_extractor.drop_rate, 
                         training=self.baseline_extractor.training)
        
        feat = x  # Store features before classification
        
        # Classification
        x = self.baseline_extractor.classifier(x)  # [B, 2]
        
        return x, feat  # (logits, features)
```

**EfficientNet Architecture Summary:**

| Variant | Input Size | Depth Coef (α) | Width Coef (β) | Final Channels | Parameters |
|---------|-----------|---------------|---------------|----------------|------------|
| B4      | 380×380   | 1.8           | 1.4           | 1792           | ~19M       |
| B5      | 456×456   | 2.2           | 1.6           | 2048           | ~30M       |
| B6      | 528×528   | 2.6           | 1.8           | 2304           | ~43M       |

**Note**: All EfficientNet variants use the same 7-stage MBConv structure, scaled by compound coefficients.

#### 4. Exponential Moving Average (EMA)

```python
class ExponentialMovingAverage:
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float,  # 0.995 for all experts
        use_num_updates: bool = True
    ):
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach().cuda() for p in parameters]
    
    def update(self, parameters: Optional[Iterable[torch.nn.Parameter]] = None):
        """Update shadow parameters using EMA formula."""
        parameters = self._get_parameters(parameters)
        decay = self.decay
        
        # Adaptive decay based on number of updates
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        
        one_minus_decay = 1.0 - decay
        
        with torch.no_grad():
            for s_param, param in zip(self.shadow_params, parameters):
                # EMA update: shadow = shadow - (1 - decay) * (shadow - param)
                tmp = (s_param - param)
                tmp.mul_(one_minus_decay)
                s_param.sub_(tmp)
```

**EMA Formula:**

$$
θ̃ₜ = β · θ̃ₜ₋₁ + (1 - β) · θₜ

where:
- θ̃ₜ = EMA parameters at step t
- θₜ = Current model parameters
- β = decay = 0.995
$$

**Adaptive Decay:**

```math
β_adapted = min(0.995, (1 + num_updates) / (10 + num_updates))

Example:
- Step 0: β = min(0.995, 1/10) = 0.1
- Step 10: β = min(0.995, 11/20) = 0.55
- Step 100: β = min(0.995, 101/110) = 0.918
- Step 1000+: β = 0.995 (saturated)
```

**Inference with EMA:**

```python
def forward_expert(self, x, idx, isTrain=False):
    cur_net = self.experts[idx]
    cur_ema = self.ema_list[idx]
    
    if isTrain:
        # Training mode: use current parameters
        x, feat = cur_net(x)
    else:
        # Inference mode: use EMA parameters
        with cur_ema.average_parameters():
            x, feat = cur_net(x)
            x = F.softmax(x, dim=1)[:, 1]  # Extract P(fake)
    
    return x, feat
```

#### 5. MoE Fusion Layer

```python
class MFF_MoE(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.name = 'MFF_MoE'
        self.experts = nn.ModuleList()
        self.ema_list = []
        
        expert_details = [
            'MFF_Expert_Convnext',  # Expert 0
            'MFF_Expert_Convnext',  # Expert 1 (duplicate architecture)
            'MFF_Expert_EffB4',     # Expert 2
            'MFF_Expert_EffB4',     # Expert 3 (duplicate architecture)
            'MFF_Expert_EffB5',     # Expert 4
            'MFF_Expert_EffB5',     # Expert 5 (duplicate architecture)
            'MFF_Expert_EffB6',     # Expert 6
        ]
        
        for idx, cls_name in enumerate(expert_details):
            expert = getattr(sys.modules[__name__], cls_name)(pretrained=pretrained)
            ema = ExponentialMovingAverage(expert.parameters(), decay=0.995)
            self.experts.append(expert)
            self.ema_list.append(ema)
    
    def forward(self, x):
        # Forward through all experts
        x = [self.forward_expert(x, idx)[0] for idx in range(len(self.experts))]
        
        # Stack predictions: [7] → [B, 7]
        x = torch.stack(x, dim=1)
        
        # Average across experts: [B, 7] → [B]
        x = torch.mean(x, dim=1)
        
        return x  # Final fake probability
```

**Fusion Strategy:**

$$
P_final(fake) = (1/7) × Σᵢ₌₀⁶ P_expertᵢ(fake)

where:
- P_expertᵢ(fake) = Softmax[logits](:, 1)
- Simple arithmetic mean (no learnable weights)
$$

### Preprocessing Pipeline

```python
self.transform = transforms.Compose([
    transforms.ToTensor(),                      # [0, 255] → [0.0, 1.0]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],             # ImageNet statistics
        std=[0.229, 0.224, 0.225]
    ),
    transforms.Resize((512, 512), antialias=True),  # Resize to 512×512
])
```

**Normalization Formula:**

$$
x_norm = (x - mean) / std
$$

Per-channel:

- R: (x_R - 0.485) / 0.229
- G: (x_G - 0.456) / 0.224
- B: (x_B - 0.406) / 0.225

### Model Loading Mechanism

```python
def load(self, path):
    # 1. Load network weights
    weights_file = os.path.join(path, 'MFF-MoE-v1.pth')
    state_dict = torch.load(weights_file, map_location='cpu')
    
    # Filter out EMA-related keys
    filtered_state_dict = {
        k: v for k, v in state_dict.items() 
        if not k.startswith('ema_list') and not k.startswith('ema_state')
    }
    
    # Load network parameters
    self.load_state_dict(filtered_state_dict, strict=True)
    
    # 2. Load EMA weights
    ema_file = os.path.join(path, 'MFF-MoE-v1.state')
    if os.path.exists(ema_file):
        self.ema_state = torch.load(ema_file, map_location='cpu')
        for idx, expert in enumerate(self.experts):
            if idx in self.ema_state:
                self.ema_list[idx].load_state_dict(self.ema_state[idx])
```

**File Structure:**

```text
models/MFF-MoE-V1/
├── MFF-MoE-v1.pth       # Main model weights (all 7 experts)
└── MFF-MoE-v1.state     # EMA shadow parameters (per-expert)
```

---

## Input/Output Specifications

### Input Requirements

| Property | Value |
|----------|-------|
| **Media Type** | Image (JPEG, PNG, etc.) or Video (MP4, AVI, MOV, etc.) |
| **Image Resolution** | Any (resized to 512×512 internally) |
| **Color Space** | RGB (3 channels) |
| **Preprocessing** | Automatic (ToTensor, Normalize, Resize) |

### Output Schema

#### Image Analysis Result

```python
ImageAnalysisResult(
    prediction: str,              # "REAL" or "FAKE"
    confidence: float,             # 0.0 to 1.0
    processing_time: float,        # Seconds
    note: Optional[str],           # Error or warning message
    dimensions: Dict[str, int]     # {"width": W, "height": H}
)
```

#### Video Analysis Result

```python
VideoAnalysisResult(
    prediction: str,                         # "REAL" or "FAKE"
    confidence: float,                        # 0.0 to 1.0
    processing_time: float,                   # Seconds
    frames_analyzed: int,                     # Number of frames processed
    frame_predictions: List[FramePrediction], # Per-frame details
    metrics: Dict[str, Any]                   # {"final_average_score": float}
)
```

**Frame Prediction Structure:**

```python
FramePrediction(
    index: int,        # Frame number (0-indexed)
    score: float,      # Frame-level fake probability
    prediction: str    # "REAL" or "FAKE"
)
```

---

## Strengths and Limitations

### Strengths

1. **Ensemble Robustness**
   - 7 diverse experts provide redundancy
   - Reduces overfitting to specific manipulation types
   - Smooths out individual model errors

2. **Multi-Scale Feature Learning**
   - ConvNeXt (local features) + EfficientNet (global features)
   - Scales from B4 (19M) to B6 (43M) parameters
   - Captures both fine-grained and coarse patterns

3. **EMA Inference**
   - EMA weights provide more stable predictions
   - Reduces variance from training noise
   - Decay=0.995 balances responsiveness and smoothness

4. **Pretrained Backbones**
   - ConvNeXt-V2: FCMAE (masked autoencoder pretraining)
   - EfficientNet: JFT-300M + Noisy Student (semi-supervised)
   - Strong generalization from large-scale pretraining

5. **Multi-Modal Support**
   - Single unified model for images and videos
   - Consistent preprocessing across modalities

6. **Duplicate Experts**
   - Experts 0-1, 2-3, 4-5 share architectures but differ in weights
   - Provides ensemble diversity through different training trajectories

### Limitations

1. **Computational Cost**
   - 7 forward passes per input (no early stopping)
   - Total: ~195M parameters (~780 MB in FP32)
   - Inference time: 7× slower than single model

2. **Fixed Ensemble Weights**
   - Simple arithmetic mean (no learnable fusion)
   - Cannot adapt importance per expert
   - Equal weighting may not be optimal for all inputs

3. **Memory Requirements**
   - Must load all 7 experts simultaneously
   - Peak memory: ~3-4 GB GPU VRAM
   - EMA doubles parameter storage (model + shadow params)

4. **Frame Sampling in Video**
   - Samples up to 100 frames (fixed)
   - May miss manipulations in unsampled frames
   - No temporal modeling across frames

5. **Resolution Loss**
   - All inputs resized to 512×512
   - Higher resolution inputs lose detail
   - EfficientNet-B6 designed for 528×528 but uses 512×512

6. **No Attention Mechanism**
   - Equal weight to all experts
   - Cannot dynamically focus on most confident expert
   - No gating network

---

## Technical Deep Dive

### Ensemble Diversity Mechanisms

#### 1. Architectural Diversity

**ConvNeXt-V2 vs. EfficientNet:**

| Aspect | ConvNeXt-V2 | EfficientNet |
|--------|-------------|--------------|
| **Building Block** | ConvNeXtBlock (Depthwise Conv 7×7) | MBConv (Depthwise Conv 3×3 or 5×5) |
| **Activation** | GELU | Swish |
| **Normalization** | LayerNorm | BatchNorm |
| **Attention** | GRN (Global Response Norm) | SE (Squeeze-and-Excitation) |
| **Scaling** | Fixed depth/width | Compound scaling (α, β, γ) |
| **Receptive Field** | Larger (7×7 conv) | Smaller (3×3, 5×5 conv) |

**Complementary Feature Learning:**

- ConvNeXt: Captures local spatial patterns (larger receptive field)
- EfficientNet: Captures global context (deeper networks, SE attention)

#### 2. Weight Diversity (Duplicate Architectures)

**Why Duplicate Experts?**

```text
Expert 0: ConvNeXt-V2-Tiny (Seed A)
Expert 1: ConvNeXt-V2-Tiny (Seed B)

Despite identical architectures:
- Different random initialization
- Different training batch orders
- Different local minima convergence

Result: Complementary error patterns
```

**Ensemble Theory:**

```text
Expected Error = Bias² + Variance + Noise

Ensemble reduces variance by:
E[Var(ensemble)] = (1/N) × E[Var(single_model)]

For N=7 experts:
Variance reduction ≈ 7× (if errors uncorrelated)
```

#### 3. Scale Diversity (EfficientNet B4/B5/B6)

**Compound Scaling Differences:**

```python
# EfficientNet-B4
depth_coefficient = 1.8
width_coefficient = 1.4
resolution = 380×380

# EfficientNet-B5
depth_coefficient = 2.2
width_coefficient = 1.6
resolution = 456×456

# EfficientNet-B6
depth_coefficient = 2.6
width_coefficient = 1.8
resolution = 528×528
```

**Scale-Specific Strengths:**

- **B4** (19M params): Fast, captures coarse features
- **B5** (30M params): Balanced depth and width
- **B6** (43M params): Deepest, finest detail (but all resize to 512×512)

### EMA Deep Dive

#### Why EMA for Inference?

**Training vs. Inference Stability:**

```text
Training Parameters (θₜ):
- High variance due to SGD noise
- Fluctuates around optimal solution
- May overfit to recent batches

EMA Parameters (θ̃ₜ):
- Smooth average over training trajectory
- Reduces high-frequency noise
- More stable generalization
```

**Mathematical Derivation:**

$$
Recursive EMA:
θ̃ₜ = β · θ̃ₜ₋₁ + (1 - β) · θₜ

Expand recursively:
θ̃ₜ = (1 - β) · Σᵢ₌₀ᵗ βⁱ · θₜ₋ᵢ

For β = 0.995:
- Weight at t=0 (current): 1 - 0.995 = 0.005 (0.5%)
- Weight at t=100: 0.005 × 0.995¹⁰⁰ ≈ 0.003 (0.3%)
- Weight at t=1000: 0.005 × 0.995¹⁰⁰⁰ ≈ 0.00003 (0.003%)

Effective window: ~1400 steps (where weight > 1%)
$$

**Adaptive Decay Schedule:**

```python
β_adapted = min(β, (1 + num_updates) / (10 + num_updates))

Why adaptive?
- Early training: rapid changes → lower β (faster adaptation)
- Late training: convergence → higher β (smoother averaging)

Example timeline:
Step 0-10:   β ≈ 0.1-0.55   (responsive to large changes)
Step 10-100: β ≈ 0.55-0.92  (transitioning to stable)
Step 100+:   β = 0.995      (maximum smoothing)
```

### Inference Workflow

#### Single Image Analysis

```python
def _analyze_image(self, image: Image.Image) -> Tuple[float, Optional[str]]:
    # 1. Preprocessing
    image_tensor = self.transform(image)  # [3, 512, 512]
    image_tensor = image_tensor.unsqueeze(0).to(self.device)  # [1, 3, 512, 512]
    
    # 2. Forward through MoE
    with torch.no_grad():
        prob_fake = self.model(image_tensor).item()  # Scalar
    
    # 3. Return result
    return prob_fake, None
```

**Detailed Forward Pass:**

```python
def forward(self, x):
    expert_preds = []
    
    # Expert 0: ConvNeXt-V2-Tiny
    with self.ema_list[0].average_parameters():
        logits, feat = self.experts[0](x)  # logits: [B, 2]
        prob = F.softmax(logits, dim=1)[:, 1]  # [B] - P(fake)
        expert_preds.append(prob)
    
    # Expert 1: ConvNeXt-V2-Tiny (duplicate)
    with self.ema_list[1].average_parameters():
        logits, feat = self.experts[1](x)
        prob = F.softmax(logits, dim=1)[:, 1]
        expert_preds.append(prob)
    
    # ... (repeat for experts 2-6)
    
    # Stack and average
    expert_preds = torch.stack(expert_preds, dim=1)  # [B, 7]
    final_prob = torch.mean(expert_preds, dim=1)     # [B]
    
    return final_prob
```

#### Video Analysis Workflow

```python
def _run_video_analysis(self, video_path: str) -> VideoAnalysisResult:
    # 1. Extract frames
    frame_generator = extract_frames(video_path, num_frames=100)
    frames = list(frame_generator)  # Up to 100 PIL images
    
    # 2. Analyze each frame
    frame_scores = []
    frame_predictions = []
    
    for i, frame_pil in enumerate(frames):
        prob_fake, _ = self._analyze_image(frame_pil)
        frame_scores.append(prob_fake)
        frame_predictions.append(
            FramePrediction(
                index=i,
                score=prob_fake,
                prediction="FAKE" if prob_fake >= 0.5 else "REAL"
            )
        )
    
    # 3. Aggregate
    avg_score = np.mean(frame_scores)
    prediction = "FAKE" if avg_score >= 0.5 else "REAL"
    confidence = avg_score if prediction == "FAKE" else 1 - avg_score
    
    return VideoAnalysisResult(
        prediction=prediction,
        confidence=confidence,
        frames_analyzed=len(frames),
        frame_predictions=frame_predictions,
        metrics={"final_average_score": avg_score}
    )
```

### Memory and Performance Analysis

#### Model Size Breakdown

```python
Expert 0: ConvNeXt-V2-Tiny  = 28M params × 4 bytes = 112 MB
Expert 1: ConvNeXt-V2-Tiny  = 28M params × 4 bytes = 112 MB
Expert 2: EfficientNet-B4   = 19M params × 4 bytes = 76 MB
Expert 3: EfficientNet-B4   = 19M params × 4 bytes = 76 MB
Expert 4: EfficientNet-B5   = 30M params × 4 bytes = 120 MB
Expert 5: EfficientNet-B5   = 30M params × 4 bytes = 120 MB
Expert 6: EfficientNet-B6   = 43M params × 4 bytes = 172 MB

Total Model Weights: 788 MB (FP32)
EMA Shadow Params:   788 MB (FP32)
Total Storage:       1.58 GB (FP32)

With FP16:
Total Storage:       790 MB (FP16)
```

#### Computational Complexity

**FLOPs per Expert (512×512 input):**

| Expert | Backbone | FLOPs | Percentage |
|--------|----------|-------|------------|
| 0, 1   | ConvNeXt-V2-Tiny | ~4.5 GFLOPs | 14.8% each |
| 2, 3   | EfficientNet-B4  | ~5.3 GFLOPs | 17.4% each |
| 4, 5   | EfficientNet-B5  | ~10.4 GFLOPs | 34.2% each |
| 6      | EfficientNet-B6  | ~19.1 GFLOPs | 62.8% |

**Total FLOPs per Image:**

```text
Total = 2×4.5 + 2×5.3 + 2×10.4 + 19.1
      = 9.0 + 10.6 + 20.8 + 19.1
      = 59.5 GFLOPs per image
```

**Throughput Estimates (RTX 3090, ~35 TFLOPs FP32):**

```text
Single Image:
- Theoretical: 35 TFLOPs / 59.5 GFLOPs = 588 images/sec
- Practical (overhead): ~150-200 images/sec
- Latency: ~5-7 ms per image

Video (100 frames):
- Total: 100 × 59.5 GFLOPs = 5.95 TFLOPs
- Time: ~0.5-0.7 seconds (100 frames)
```

### Integration Example

```python
from src.config import MFFMoEV1Config
from src.ml.detectors.mff_moe_detector import MFFMoEDetectorV1

# 1. Load configuration
config = MFFMoEV1Config(
    class_name="MFFMoEDetectorV1",
    model_name="MFF-MOE-V1",
    model_path="models/MFF-MoE-V1/",  # Directory containing .pth and .state
    isImage=True,
    isVideo=True,
    isAudio=False,
    video_frames_to_sample=100
)

# 2. Initialize detector
detector = MFFMoEDetectorV1(config)
detector.load()

# 3. Analyze an image
image_result = detector.analyze(media_path="image.jpg")
print(f"Image Prediction: {image_result.prediction}")
print(f"Confidence: {image_result.confidence:.3f}")
print(f"Dimensions: {image_result.dimensions}")

# 4. Analyze a video
video_result = detector.analyze(media_path="video.mp4")
print(f"Video Prediction: {video_result.prediction}")
print(f"Confidence: {video_result.confidence:.3f}")
print(f"Frames Analyzed: {video_result.frames_analyzed}")
print(f"Average Score: {video_result.metrics['final_average_score']:.3f}")

# 5. Frame-level analysis
for frame_pred in video_result.frame_predictions[:10]:
    print(f"Frame {frame_pred.index}: {frame_pred.score:.3f} ({frame_pred.prediction})")
```

### Configuration Parameters

```yaml
MFF-MOE-V1:
  class_name: "MFFMoEDetectorV1"
  model_path: "models/MFF-MoE-V1/"
  isImage: true
  isVideo: true
  isAudio: false
  video_frames_to_sample: 100
```

**Parameter Descriptions:**

- **model_path**: Directory containing two files:
  - `MFF-MoE-v1.pth`: Main model weights (all 7 experts)
  - `MFF-MoE-v1.state`: EMA shadow parameters (per-expert dict)

- **video_frames_to_sample**: Maximum frames to extract from video
  - Default: 100 frames (evenly spaced)
  - Balances temporal coverage and processing time

---

## Summary

MFF-MoE (V1) is a **7-expert ensemble model** that combines ConvNeXt-V2 and EfficientNet backbones (B4/B5/B6) using a Mixture-of-Experts architecture with Exponential Moving Average inference. The model achieves robustness through architectural diversity, weight diversity (duplicate experts with different initializations), and scale diversity (multi-resolution backbones).

**Best Use Cases:**

- High-stakes detection requiring robustness
- Image and video analysis with consistent preprocessing
- Scenarios where computational cost is acceptable
- Production systems needing stable predictions (via EMA)

**Not Suitable For:**

- Real-time applications (7× forward passes)
- Resource-constrained environments (1.6 GB model + 3-4 GB VRAM)
- Long videos (samples only 100 frames)
- Applications requiring temporal modeling

The ensemble approach provides strong generalization by combining complementary feature representations from diverse backbones, with EMA inference ensuring stable and smooth predictions across varied inputs.
