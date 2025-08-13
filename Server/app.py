from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import cv2
import numpy as np
import os
import time
from PIL import Image
from transformers import AutoProcessor
import torch.nn.functional as F
from werkzeug.utils import secure_filename
import uuid

from src.utils import load_config
from src.model_lstm import create_lstm_model

app = Flask(__name__)
CORS(app)

# Global variables for model
model = None
processor = None
config = None
device = None

def initialize_model():
    """Initialize the deepfake detection model"""
    global model, processor, config, device
    
    try:
        config = load_config('configs/config_lstm.yaml')
        device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading LSTM model from {config['best_model_save_path']}...")
        model = create_lstm_model(config)
        model.load_state_dict(torch.load(config['best_model_save_path'], map_location=device))
        model.to(device)
        model.eval()
        
        processor = AutoProcessor.from_pretrained(config['base_model_path'])
        
        print("Model initialized successfully!")
        return True
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        return False

def extract_frames(video_path, num_frames):
    """Extract frames from video for analysis"""
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            return []
        
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        
        cap.release()
        
        # Ensure we have exactly num_frames
        if len(frames) == 0:
            return []
        
        if len(frames) > num_frames:
            frames = frames[:num_frames]
        
        while len(frames) < num_frames:
            frames.append(frames[-1])
            
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return []
    
    return frames

def predict_video(video_path):
    """Predict if video is real or fake"""
    global model, processor, config, device
    
    if not model or not processor:
        raise Exception("Model not initialized")
    
    start_time = time.time()
    
    # Extract frames
    frames = extract_frames(video_path, config['num_frames_per_video'])
    
    if not frames:
        raise Exception("Could not extract frames from video")
    
    # Process frames
    inputs = processor(images=frames, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)
    
    # Make prediction
    with torch.no_grad():
        logits = model(pixel_values, num_frames_per_video=config['num_frames_per_video'])
        probabilities = F.softmax(logits, dim=1)
        confidence, predicted_class_id = torch.max(probabilities, dim=1)
    
    processing_time = time.time() - start_time
    
    # Map prediction to label
    label_map = {0: "REAL", 1: "FAKE"}
    predicted_label = label_map[predicted_class_id.item()]
    confidence_score = confidence.item()
    
    return {
        "prediction": predicted_label,
        "confidence": confidence_score,
        "processing_time": processing_time,
        "model_version": "siglip-lstm-v1"
    }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "not initialized",
        "timestamp": time.time()
    })

@app.route('/analyze', methods=['POST'])
def analyze_video():
    """Analyze video for deepfake detection"""
    try:
        # Check if model is loaded
        if not model or not processor:
            return jsonify({
                "success": False,
                "error": "Model not initialized"
            }), 500
        
        # Check if video file is present
        if 'video' not in request.files:
            return jsonify({
                "success": False,
                "error": "No video file provided"
            }), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({
                "success": False,
                "error": "No video file selected"
            }), 400
        
        # Get video ID from request
        video_id = request.form.get('videoId', str(uuid.uuid4()))
        
        # Save uploaded file temporarily
        filename = secure_filename(video_file.filename)
        temp_path = f"temp/{video_id}_{filename}"
        
        # Create temp directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)
        
        video_file.save(temp_path)
        
        try:
            # Analyze the video
            result = predict_video(temp_path)
            
            # Clean up temporary file
            os.remove(temp_path)
            
            return jsonify({
                "success": True,
                "video_id": video_id,
                **result
            })
            
        except Exception as analysis_error:
            # Clean up temporary file on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return jsonify({
                "success": False,
                "error": f"Analysis failed: {str(analysis_error)}"
            }), 500
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    global config
    
    if not config:
        return jsonify({
            "success": False,
            "error": "Model not initialized"
        }), 500
    
    return jsonify({
        "success": True,
        "model_info": {
            "name": config.get('project_name', 'Unknown'),
            "base_model": config.get('base_model_path', 'Unknown'),
            "num_frames": config.get('num_frames_per_video', 'Unknown'),
            "lstm_hidden_size": config.get('lstm_hidden_size', 'Unknown'),
            "lstm_layers": config.get('lstm_num_layers', 'Unknown'),
            "device": str(device) if device else "Unknown"
        }
    })

if __name__ == '__main__':
    print("Starting Drishtiksha AI Server...")
    
    # Initialize model
    if initialize_model():
        print("✅ Model initialized successfully!")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ Failed to initialize model. Please check your configuration.")
