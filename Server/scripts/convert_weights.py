import os
import torch
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Input, TimeDistributed, GlobalAveragePooling2D, LSTM, Dropout, Dense
from tensorflow.keras.models import Model

# Add the project root to the path to import our PyTorch model
import sys  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from src.ml.architectures.eyeblink_cnn_lstm import EyeblinkCnnLstm
from src.config import EyeblinkModelConfig # We need this to get model params

# --- CONFIGURATION ---
# ‼️ UPDATE THESE THREE PATHS ‼️
KERAS_WEIGHTS_PATH = "models/EyeBlink-CNN-LSTM-v1.keras"
PYTORCH_OUTPUT_PATH = "models/EyeBlink-CNN-LSTM-v1.pth"
CONFIG_YAML_PATH = "configs/config.yaml"

# --- 1. Keras Model Definition (Copied from your script) ---
def create_keras_model(img_size=(160, 160), sequence_length=5):
    cnn_base = Xception(weights="imagenet", include_top=False, input_shape=(*img_size, 3), pooling="avg")
    # Freeze the base
    for layer in cnn_base.layers:
        layer.trainable = False
        
    seq_input = Input(shape=(sequence_length, *img_size, 3))
    x = TimeDistributed(cnn_base)(seq_input)
    x = LSTM(128)(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation="sigmoid")(x)
    model = Model(seq_input, out)
    print("✅ Keras model architecture built.")
    return model

# --- 2. Weight Conversion Logic ---
def convert_keras_to_pytorch(keras_model, pytorch_model):
    """
    Copies weights from the trained Keras model to the PyTorch model.
    """
    print("🔄 Starting weight conversion...")
    
    # We only need to copy the weights for the layers we trained: LSTM and Dense
    keras_layers = [l for l in keras_model.layers if isinstance(l, (LSTM, Dense))]
    pytorch_modules = {
        'lstm': pytorch_model.lstm,
        'fc': pytorch_model.fc
    }

    # --- LSTM Layer Conversion ---
    keras_lstm = keras_layers[0]
    pytorch_lstm = pytorch_modules['lstm']
    
    keras_weights = keras_lstm.get_weights()
    # Keras: [kernel (input), recurrent_kernel (hidden), bias]
    # PyTorch: [weight_ih, weight_hh, bias_ih, bias_hh]
    
    # Keras kernel maps to PyTorch input-hidden weights
    pytorch_lstm.weight_ih_l0.data = torch.from_numpy(keras_weights[0].T)
    # Keras recurrent_kernel maps to PyTorch hidden-hidden weights
    pytorch_lstm.weight_hh_l0.data = torch.from_numpy(keras_weights[1].T)
    # Keras bias needs to be split for PyTorch
    keras_bias = keras_weights[2]
    pytorch_lstm.bias_ih_l0.data = torch.from_numpy(keras_bias)
    pytorch_lstm.bias_hh_l0.data = torch.zeros_like(pytorch_lstm.bias_hh_l0.data)
    print("  - LSTM weights converted.")

    # --- Dense (Fully Connected) Layer Conversion ---
    keras_fc = keras_layers[1]
    pytorch_fc = pytorch_modules['fc']
    
    keras_weights = keras_fc.get_weights()
    # Keras: [kernel, bias]
    # PyTorch: [weight, bias]
    # ‼️ IMPORTANT: Keras kernel must be transposed for PyTorch ‼️
    pytorch_fc.weight.data = torch.from_numpy(keras_weights[0].T)
    pytorch_fc.bias.data = torch.from_numpy(keras_weights[1])
    print("  - Dense layer weights converted.")
    
    print("✅ Weight conversion successful.")
    return pytorch_model

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Keras to PyTorch Weight Converter for Eyeblink Model ---")
    
    # Load Eyeblink config from YAML to get model parameters
    import yaml
    try:
        with open(CONFIG_YAML_PATH, 'r') as f:
            full_config = yaml.safe_load(f)
        eyeblink_raw_config = full_config['models']['EYEBLINK-CNN-LSTM-V1']
        
        # Use Pydantic to validate and structure the config
        eyeblink_pydantic_config = EyeblinkModelConfig(**eyeblink_raw_config)
        
        IMG_SIZE = eyeblink_pydantic_config.model_definition.img_size
        SEQ_LEN = eyeblink_pydantic_config.sequence_length

    except Exception as e:
        print(f"❌ Error loading configuration from {CONFIG_YAML_PATH}: {e}")
        exit()

    # 1. Build and load the Keras model with its trained weights
    keras_model = create_keras_model(img_size=IMG_SIZE, sequence_length=SEQ_LEN)
    print(f"  - Loading Keras weights from: {KERAS_WEIGHTS_PATH}")
    keras_model.load_weights(KERAS_WEIGHTS_PATH)
    
    # 2. Build the equivalent PyTorch model
    print("\n🔨 Building PyTorch model architecture...")
    pytorch_model = EyeblinkCnnLstm(eyeblink_pydantic_config.model_definition.model_dump())
    pytorch_model.eval() # Set to evaluation mode
    
    # 3. Perform the conversion
    pytorch_model_with_weights = convert_keras_to_pytorch(keras_model, pytorch_model)
    
    # 4. Save the PyTorch model's state dictionary
    print(f"\n💾 Saving PyTorch state dictionary to: {PYTORCH_OUTPUT_PATH}")
    torch.save(pytorch_model_with_weights.state_dict(), PYTORCH_OUTPUT_PATH)
    
    print("\n🎉 Conversion complete! You can now use the .pth file in your server.")