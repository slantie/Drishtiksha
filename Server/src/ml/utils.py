# src/ml/utils.py

import cv2
import numpy as np
import logging
from PIL import Image
from typing import List

logger = logging.getLogger(__name__)

def extract_frames(video_path: str, num_frames: int) -> List[Image.Image]:
    """
    Extracts a specified number of frames evenly from a video.

    Args:
        video_path (str): The path to the video file.
        num_frames (int): The total number of frames to extract.

    Returns:
        A list of PIL.Image.Image objects. Returns an empty list on failure.
    """
    frames = []
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            logger.warning(f"Video file has no frames: {video_path}")
            return []

        # Ensure we don't request more frames than available, preventing errors
        num_frames_to_extract = min(total_frames, num_frames)
        frame_indices = np.linspace(0, total_frames - 1, num_frames_to_extract, dtype=int)

        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Convert frame from BGR (OpenCV default) to RGB for PIL
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        # If fewer frames were extracted than desired (e.g., video is too short),
        # duplicate the last frame to meet the required count for the model.
        if frames and len(frames) < num_frames:
            logger.debug(
                f"Duplicating last frame to meet model input size "
                f"({len(frames)} -> {num_frames})."
            )
            while len(frames) < num_frames:
                frames.append(frames[-1])

        return frames

    except Exception as e:
        logger.error(f"Error extracting frames from {video_path}: {e}", exc_info=True)
        return []
    finally:
        if cap:
            cap.release()