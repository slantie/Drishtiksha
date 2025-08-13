# src/training/data_loader_lstm.py

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
                if ret: frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            cap.release()
            if not frames: return []
            while len(frames) < self.num_frames: frames.append(frames[-1])
            return frames[:self.num_frames]
        except Exception as e:
            print(f"\nError processing video {video_path}: {e}")
            return []

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label_name = self.labels[idx]
        frames = self._extract_frames(video_path)
        if not frames: return None
        
        inputs = self.processor(images=frames, return_tensors="pt")
        label_id = self.label_map[label_name]
        
        return {
            'pixel_values': inputs['pixel_values'],
            'label': torch.tensor(label_id, dtype=torch.float32)
        }

def get_lstm_data_loaders(config):
    """Creates data loaders from the training configuration."""
    train_config = config['training']
    model_config = config['models'][train_config['target_model']]

    processor = AutoProcessor.from_pretrained(model_config['processor_path'])
    
    def _get_all_video_files(dir_paths):
        all_files = []
        for path in dir_paths:
            all_files.extend(glob(os.path.join(path, '**/*.mp4'), recursive=True))
        return all_files
    
    print("Gathering video files...")
    real_videos = _get_all_video_files(train_config['real_data_paths'])
    fake_videos = _get_all_video_files(train_config['fake_data_paths'])
    print(f"Found {len(real_videos)} real videos and {len(fake_videos)} fake videos.")

    video_files = real_videos + fake_videos
    labels = ["real"] * len(real_videos) + ["fake"] * len(fake_videos)
    
    train_files, val_files, train_labels, val_labels = train_test_split(
        video_files, labels, test_size=train_config['validation_split'], random_state=42, stratify=labels)
    
    train_dataset = VideoFrameDataset(train_files, train_labels, processor, model_config['num_frames'])
    val_dataset = VideoFrameDataset(val_files, val_labels, processor, model_config['num_frames'])
    
    def lstm_collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if not batch: return None, None
        pixel_values = torch.cat([item['pixel_values'] for item in batch], dim=0)
        labels = torch.stack([item['label'] for item in batch])
        return pixel_values, labels

    train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True, collate_fn=lstm_collate_fn, num_workers=train_config.get('num_workers', 4), pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False, collate_fn=lstm_collate_fn, num_workers=train_config.get('num_workers', 4), pin_memory=True)
    
    return train_loader, val_loader