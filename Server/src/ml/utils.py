# src/ml/utils.py

import cv2
import os
import numpy as np
import logging
from PIL import Image
from typing import List, Generator, Optional

logger = logging.getLogger(__name__)

# FIX: The function signature is updated to accept the 'specific_indices' argument.
def extract_frames(
    video_path: str, 
    num_frames: int, 
    specific_indices: Optional[List[int]] = None
) -> Generator[Image.Image, None, None]:
    """
    Extracts frames from a video, either evenly spaced or at specific indices.
    This function is a generator, yielding frames one by one to conserve memory.

    Args:
        video_path (str): The path to the video file.
        num_frames (int): The total number of frames to extract if specific_indices is None.
        specific_indices (Optional[List[int]]): A list of specific frame indices to extract.

    Yields:
        PIL.Image.Image: The extracted frames.
    """
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            logger.warning(f"Video file appears to be empty or corrupt: {video_path}")
            return

        extracted_frames_buffer = []
        
        if specific_indices:
            # Mode 1: Extract specific frames requested by the caller
            indices_to_extract = sorted(list(set(specific_indices)))
        else:
            # Mode 2: Extract an evenly spaced sequence of frames
            num_to_get = min(total_frames, num_frames)
            indices_to_extract = np.linspace(0, total_frames - 1, num_to_get, dtype=int)

        for i in indices_to_extract:
            if i >= total_frames: continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                extracted_frames_buffer.append(pil_frame)
                yield pil_frame

        # Handle padding only when NOT extracting specific indices
        if not specific_indices and extracted_frames_buffer and len(extracted_frames_buffer) < num_frames:
            num_to_pad = num_frames - len(extracted_frames_buffer)
            logger.debug(
                f"Padding with {num_to_pad} copies of the first frame for '{os.path.basename(video_path)}' "
                f"to meet model input size ({len(extracted_frames_buffer)} -> {num_frames})."
            )
            first_frame = extracted_frames_buffer[0]
            for _ in range(num_to_pad):
                yield first_frame

    except Exception as e:
        logger.error(f"Error during frame extraction from {video_path}: {e}", exc_info=True)
    finally:
        if cap:
            cap.release()