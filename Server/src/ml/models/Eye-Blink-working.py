# -*- coding: utf-8 -*-
import os
import cv2
import dlib
import imutils
import numpy as np
import torch
import torch.nn as nn
import timm
from imutils import face_utils
from scipy.spatial import distance as dist

# ---------------- CONFIG ----------------
SHAPE_PREDICTOR_PATH = "models/Face-Landmarks.dat"
MODEL_WEIGHTS_PATH = "models/cnn_LSTM_Ext_DS.pth"

IMG_SIZE = (160, 160)
SEQ_LEN = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BLINK_THRESH = 0.45
SUCC_FRAME = 2


# ---------------- MODEL ----------------
class CNN_LSTM(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=1):
        super().__init__()
        self.cnn = timm.create_model("legacy_xception", pretrained=False, num_classes=0)
        feat_dim = self.cnn.num_features
        self.lstm = nn.LSTM(input_size=feat_dim, hidden_size=hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        feats = self.cnn(x)                      # [B*T, feat_dim]
        feats = feats.view(B, T, -1)             # [B, T, feat_dim]
        out, _ = self.lstm(feats)                # [B, T, hidden_dim]
        out = out[:, -1, :]                      # last timestep
        out = self.dropout(out)
        out = self.fc(out)
        return torch.sigmoid(out).squeeze(1)


# ---------------- HELPERS ----------------
def calculate_EAR(eye):
    y1 = dist.euclidean(eye[1], eye[5])
    y2 = dist.euclidean(eye[2], eye[4])
    x1 = dist.euclidean(eye[0], eye[3])
    return 0.0 if x1 == 0 else (y1 + y2) / x1


def preprocess_frame(frame):
    if frame is None:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, IMG_SIZE)
    arr = resized.astype(np.float32) / 255.0
    arr = torch.from_numpy(arr).permute(2, 0, 1)  # [C,H,W]
    return arr


# ---------------- PIPELINE ----------------
def process_and_predict_video(video_path, shape_predictor, model):
    print(f"🎥 Processing video: {os.path.basename(video_path)}...")

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
        (h, w) = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) > 0:
            face = faces[0]
            shape = shape_predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            lefteye = shape[L_start:L_end]
            righteye = shape[R_start:R_end]
            all_eye_landmarks = np.concatenate((lefteye, righteye))
            (x, y, ew, eh) = cv2.boundingRect(all_eye_landmarks)
            padding = 15
            eye_crop = frame[max(0, y-padding):min(h, y+eh+padding),
                             max(0, x-padding):min(w, x+ew+padding)]

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
    print(f"✅ Blink detection complete. Found {len(all_blink_frames)} frames.")

    if len(all_blink_frames) < SEQ_LEN:
        print(f"⚠️ Not enough blink frames ({len(all_blink_frames)} < {SEQ_LEN})")
        return

    processed_frames = [preprocess_frame(f) for f in all_blink_frames if f is not None]

    sequences = []
    for i in range(len(processed_frames) - SEQ_LEN + 1):
        seq = processed_frames[i:i+SEQ_LEN]
        sequences.append(torch.stack(seq))  # [T,C,H,W]

    if not sequences:
        print("⚠️ Could not build sequences. Prediction aborted.")
        return

    batch = torch.stack(sequences).to(DEVICE)  # [N,T,C,H,W]
    model.eval()
    with torch.no_grad():
        preds = model(batch).cpu().numpy()

    avg_prob = float(np.mean(preds))
    final_class = "FAKE" if avg_prob >= 0.5 else "REAL"

    print("\n" + "="*34)
    print("   Final Video Classification   ")
    print("="*34)
    print(f"🎬 Predicted Class: {final_class}")
    print(f"📈 Probability Score: {avg_prob:.4f}")
    print("(Closer to 1.0 is REAL, closer to 0.0 is FAKE)")
    print("="*34 + "\n")


# ---------------- MAIN ----------------
if __name__ == "__main__":
    vid_path_1 = "Fake 1.mp4"
    vid_path_2 = "Fake 2.mp4"
    vid_path_3 = "Real 1.mp4"

    if not os.path.exists(vid_path_1):
        print(f"❌ Video not found: {vid_path_1}")
    elif not os.path.exists(SHAPE_PREDICTOR_PATH):
        print(f"❌ Missing dlib shape predictor: {SHAPE_PREDICTOR_PATH}")
    elif not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"❌ Missing model weights: {MODEL_WEIGHTS_PATH}")
    else:
        try:
            # Build model
            model = CNN_LSTM().to(DEVICE)

            # Load weights (handle both old/new formats)
            checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE)
            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                model.load_state_dict(checkpoint["model_state"])
            else:
                model.load_state_dict(checkpoint)
            print("✅ Model weights loaded.")

            # Load shape predictor
            shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

            # Run inference
            process_and_predict_video(vid_path_1, shape_predictor, model)
            process_and_predict_video(vid_path_2, shape_predictor, model)
            process_and_predict_video(vid_path_3, shape_predictor, model)

        except Exception as e:
            print(f"⚠️ Unexpected error: {e}")