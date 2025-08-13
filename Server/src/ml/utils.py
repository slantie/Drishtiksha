# src/ml/utils.py

import cv2
import numpy as np
from PIL import Image
from typing import List

def extract_frames(video_path: str, num_frames: int) -> List[Image.Image]:
    """Extracts a specified number of frames evenly from a video for inference."""
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
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

        # Handle cases where fewer frames were extracted than required
        if not frames: return []
        while len(frames) < num_frames:
            frames.append(frames[-1]) # Duplicate the last frame
        
        return frames[:num_frames]

    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        return []