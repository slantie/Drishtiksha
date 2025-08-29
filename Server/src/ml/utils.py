# src/ml/utils.py

import cv2
import os
import numpy as np
import logging
from PIL import Image
from typing import List, Generator, Optional, Iterable

logger = logging.getLogger(__name__)


class VideoCaptureManager:
    """
    A robust context manager for cv2.VideoCapture.

    Ensures that the video capture object is always released, even if errors occur.
    Also handles the initial check for whether the video file can be opened.
    """
    def __init__(self, video_path: str):
        self.video_path = video_path
        self._cap = None

    def __enter__(self) -> Optional[cv2.VideoCapture]:
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            logger.error(f"Could not open video file: {self.video_path}")
            # Raise an error to be caught by the calling function.
            raise IOError(f"Could not open video file: {self.video_path}")
        return self._cap

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cap:
            self._cap.release()


def extract_frames(
    video_path: str,
    num_frames: int,
    specific_indices: Optional[List[int]] = None
) -> Generator[Image.Image, None, None]:
    """
    REFACTORED high-performance frame extractor.

    This function is a generator that yields frames one by one to conserve memory.
    It uses an optimized sequential-read approach for evenly spaced frames and
    seeking for specific indices.

    Args:
        video_path (str): The path to the video file.
        num_frames (int): The total number of frames to extract for even spacing.
        specific_indices (Optional[List[int]]): A list of specific frame indices to extract.

    Yields:
        PIL.Image.Image: The extracted frames, converted to RGB.

    Raises:
        IOError: If the video file cannot be opened or is corrupt.
    """
    try:
        with VideoCaptureManager(video_path) as cap:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < 1:
                logger.warning(f"Video file appears empty or corrupt: {video_path}")
                return

            if specific_indices:
                # --- Mode 1: Extract specific frames (seeking is necessary) ---
                indices_to_extract = sorted(list(set(specific_indices)))
                for i in indices_to_extract:
                    if i >= total_frames: continue
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if ret:
                        yield Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                # --- Mode 2: High-performance sequential read for evenly spaced frames ---
                num_to_get = min(total_frames, num_frames)
                indices_to_extract = set(np.linspace(0, total_frames - 1, num_to_get, dtype=int))
                
                for i in range(total_frames):
                    ret, frame = cap.read()
                    if not ret: break
                    if i in indices_to_extract:
                        yield Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    except (IOError, cv2.error) as e:
        logger.error(f"Error during frame extraction from {video_path}: {e}", exc_info=True)
        # Re-raise to allow the caller to handle the failure.
        raise IOError(f"Frame extraction failed for {video_path}") from e


def pad_frames(
    frames: Iterable[Image.Image],
    target_count: int
) -> List[Image.Image]:
    """
    Pads a list of frames to a target count by repeating the first frame.

    This is a model-specific utility, separated from the generic `extract_frames`.

    Args:
        frames (Iterable[Image.Image]): An iterable of frames to pad.
        target_count (int): The desired number of frames in the output list.

    Returns:
        List[Image.Image]: The padded list of frames.
    """
    frame_list = list(frames)
    if not frame_list:
        return []

    current_count = len(frame_list)
    if current_count >= target_count:
        return frame_list

    num_to_pad = target_count - current_count
    logger.debug(f"Padding with {num_to_pad} copies of the first frame to meet model input size.")
    
    first_frame = frame_list[0]
    padding = [first_frame] * num_to_pad
    frame_list.extend(padding)
    
    return frame_list