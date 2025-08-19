# -*- coding: utf-8 -*-
import os
from imutils import face_utils
import tensorflow as tf
import dlib
import imutils
import numpy as np
import cv2
from scipy.spatial import distance as dist
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import (Input, TimeDistributed, GlobalAveragePooling2D,
                                     LSTM, Dropout, Dense)
from tensorflow.keras.models import Model

# --- Constants and Configuration ---
# ‼️ IMPORTANT: Update these paths before running ‼️
SHAPE_PREDICTOR_PATH = "D://Slantie@LDRP//7th Semester//NTRO//Website//Server//models//Face-Landmarks.dat"
MODEL_WEIGHTS_PATH = "D://Slantie@LDRP//7th Semester//NTRO//Website//Server//models//EyeBlink-CNN-LSTM-v1.keras"

# Model and Preprocessing settings
IMG_SIZE = (160, 160)
SEQ_LEN = 5

# Blink detection thresholds
BLINK_THRESH = 0.45   # Adjust as per dataset
SUCC_FRAME = 2        # Number of successive frames to count as a blink

# --- Model Builder ---
def create_model(img_size=IMG_SIZE, sequence_length=SEQ_LEN):
    """
    Builds the CNN + LSTM model for blink-based deepfake detection.
    """
    # CNN feature extractor (Xception without top layer)
    cnn_base = Xception(weights="imagenet", include_top=False, input_shape=(*img_size, 3))

    # Define the sequential input
    seq_input = Input(shape=(sequence_length, *img_size, 3))

    # Wrap the CNN base in a TimeDistributed layer
    x = TimeDistributed(cnn_base)(seq_input)

    # Flatten features for each frame
    x = TimeDistributed(GlobalAveragePooling2D())(x)

    # LSTM layer
    x = LSTM(128)(x)

    # Dropout
    x = Dropout(0.3)(x)

    # Final classification
    out = Dense(1, activation="sigmoid")(x)

    model = Model(seq_input, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print("✅ Model architecture built successfully.")
    return model

# --- Helper Functions ---
def calculate_EAR(eye):
    """
    Calculates the Eye Aspect Ratio (EAR) for a given set of eye landmarks.
    """
    y1 = dist.euclidean(eye[1], eye[5])
    y2 = dist.euclidean(eye[2], eye[4])
    x1 = dist.euclidean(eye[0], eye[3])
    if x1 == 0:
        return 0.0
    return (y1 + y2) / x1

def preprocess_frame(frame):
    """
    Resizes and normalizes a single frame for model prediction.
    """
    if frame is None:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(frame_rgb, IMG_SIZE)
    return resized_frame.astype(np.float32) / 255.0

# --- Core Processing Function ---
def process_and_predict_video(video_path, shape_predictor, model):
    """
    Processes a video to detect blinks, creates sequences of blink frames,
    and predicts whether the video is real or fake using the loaded model.
    """
    print(f"🎥 Starting processing for video: {os.path.basename(video_path)}...")

    detector = dlib.get_frontal_face_detector()
    (L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    cam = cv2.VideoCapture(video_path)
    if not cam.isOpened():
        print("❌ Error: Could not open video file.")
        return

    count_frame = 0
    all_blink_frames = []
    blink_frame_buffer = []

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=640)
        (frame_h, frame_w) = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) > 0:
            face = faces[0]  # Assume one face
            shape = shape_predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            lefteye = shape[L_start:L_end]
            righteye = shape[R_start:R_end]

            all_eye_landmarks = np.concatenate((lefteye, righteye))
            (x, y, w, h) = cv2.boundingRect(all_eye_landmarks)
            padding = 15
            eye_crop = frame[max(0, y - padding):min(frame_h, y + h + padding),
                             max(0, x - padding):min(frame_w, x + w + padding)]

            avg_EAR = (calculate_EAR(lefteye) + calculate_EAR(righteye)) / 2.0

            if avg_EAR < BLINK_THRESH:
                count_frame += 1
                if eye_crop.size > 0:
                    blink_frame_buffer.append(eye_crop)
            else:
                if count_frame >= SUCC_FRAME:
                    all_blink_frames.extend(blink_frame_buffer)
                count_frame = 0
                blink_frame_buffer = []

    cam.release()
    print(f"✅ Blink detection complete. Found {len(all_blink_frames)} frames across all blinks.")

    if len(all_blink_frames) < SEQ_LEN:
        print(f"⚠️ Warning: Only {len(all_blink_frames)} blink frames were detected. "
              f"Need at least {SEQ_LEN} to form a sequence. Cannot predict.")
        return

    processed_frames = [preprocess_frame(f) for f in all_blink_frames if f is not None]

    sequences = []
    for i in range(len(processed_frames) - SEQ_LEN + 1):
        seq = processed_frames[i:i + SEQ_LEN]
        sequences.append(seq)

    if not sequences:
        print("⚠️ Warning: Could not build any sequences. Prediction aborted.")
        return

    sequences_np = np.array(sequences)
    print(f"📊 Successfully built {len(sequences_np)} sequences for prediction.")

    predictions = model.predict(sequences_np, verbose=0).ravel()
    avg_confidence = float(np.mean(predictions))
    final_class = "REAL" if avg_confidence >= 0.5 else "FAKE"

    print("\n" + "="*34)
    print("   Final Video Classification   ")
    print("="*34)
    print(f"🎬 Predicted Class: {final_class}")
    print(f"📈 Confidence Score: {avg_confidence:.4f}")
    print("(Closer to 1.0 is REAL, closer to 0.0 is FAKE)")
    print("="*34 + "\n")

# --- Main Execution Block ---
if __name__ == "__main__":
    vid_path = "D://Slantie@LDRP//7th Semester//NTRO//Website//Server//assets//id1_id23_0009.mp4"
    # vid_path = "D:\\Slantie@LDRP\\7th Semester\\NTRO\\Website\\Server\\assets\\00000.mp4"
    # vid_path = "D:\\Slantie@LDRP\\7th Semester\\NTRO\\Website\\Server\\assets\\id0_0001.mp4"
    # vid_path = "D:\\Slantie@LDRP\\7th Semester\\NTRO\\Website\\Server\\assets\\id0_id1_0001.mp4"

    if not os.path.exists(vid_path):
        print(f"❌ Error: Video file not found at '{vid_path}'")
    elif not os.path.exists(SHAPE_PREDICTOR_PATH):
        print(f"❌ Error: dlib shape predictor not found at '{SHAPE_PREDICTOR_PATH}'")
    elif not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"❌ Error: Model weights file not found at '{MODEL_WEIGHTS_PATH}'")
    else:
        try:
            # 1. Build the model
            model = create_model()

            # 2. Load pre-trained weights
            print(f"🔄 Loading trained weights from: {MODEL_WEIGHTS_PATH}")
            model.load_weights(MODEL_WEIGHTS_PATH)
            print("✅ Weights loaded successfully.")

            # 3. Load dlib shape predictor
            shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

            # 4. Run prediction
            process_and_predict_video(vid_path, shape_predictor, model)

        except Exception as e:
            print(f"⚠️ An unexpected error occurred: {e}")
