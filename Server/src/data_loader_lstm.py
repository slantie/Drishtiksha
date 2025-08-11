# /home/dell-pc-03/Deepfake/deepfake-detection/Raj/LSTM_video/src/data_loader_lstm.py

import os
import cv2
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoProcessor
from PIL import Image

class VideoFrameDataset(Dataset):
    """
    Dataset for loading video frames for deepfake detection.
    It extracts a fixed number of frames from each video.
    """
    def __init__(self, video_files, labels, processor, num_frames):
        self.video_files = video_files
        self.labels = labels
        self.processor = processor
        self.num_frames = num_frames
        self.label_map = {"real": 0, "fake": 1}

    def __len__(self):
        return len(self.video_files)

    def _extract_frames(self, video_path):
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): return []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < 1: return []
            
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
            for i in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            cap.release()

            # --- THIS IS THE CRITICAL FIX ---
            # Guarantee that exactly num_frames are returned.
            if len(frames) == 0:
                return []

            # If too many frames were extracted (rare), truncate the list.
            if len(frames) > self.num_frames:
                frames = frames[:self.num_frames]
            
            # If too few frames were extracted (common), pad by duplicating the last frame.
            while len(frames) < self.num_frames:
                frames.append(frames[-1])
            # --- END OF FIX ---

        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return []
        
        return frames

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]
        frames = self._extract_frames(video_path)
        if not frames:
            return None
        inputs = self.processor(images=frames, return_tensors="pt")
        label_id = self.label_map[label]
        return {
            'pixel_values': inputs['pixel_values'],
            'label': torch.tensor(label_id).long()
        }


def get_lstm_data_loaders(config):
    """Creates data loaders suitable for the LSTM model."""
    processor = AutoProcessor.from_pretrained(config['base_model_path'])
    
    real_videos = glob(os.path.join(config['real_data_path'], '*.mp4'))
    fake_videos = glob(os.path.join(config['fake_data_path'], '*.mp4'))

    video_files = real_videos + fake_videos
    labels = ["real"] * len(real_videos) + ["fake"] * len(fake_videos)

    train_files, val_files, train_labels, val_labels = train_test_split(
        video_files, labels, test_size=config['validation_split'], random_state=42, stratify=labels
    )

    train_dataset = VideoFrameDataset(train_files, train_labels, processor, config['num_frames_per_video'])
    val_dataset = VideoFrameDataset(val_files, val_labels, processor, config['num_frames_per_video'])

    def lstm_collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if not batch: return None, None
        pixel_values = torch.cat([item['pixel_values'] for item in batch], dim=0)
        labels = torch.stack([item['label'] for item in batch])
        return pixel_values, labels

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, 
        collate_fn=lstm_collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False, 
        collate_fn=lstm_collate_fn, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader