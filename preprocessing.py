"""
Preprocessing functions for Pong environment frames.
Converts raw RGB frames into processed inputs suitable for neural networks.
"""

import numpy as np
import cv2


def rgb_to_grayscale(frame):
    """
    Convert RGB frame to grayscale.
    
    Args:
        frame: numpy array of shape (210, 160, 3) with RGB values 0-255
        
    Returns:
        Grayscale frame of shape (210, 160) with values 0-255
    """
    # Using standard grayscale conversion weights
    # 0.299*R + 0.587*G + 0.114*B
    grayscale = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
    return grayscale.astype(np.uint8)


def downsample_frame(frame, target_size=(80, 80)):
    """
    Downsample frame to smaller dimensions.
    
    Args:
        frame: numpy array of shape (H, W) or (H, W, C)
        target_size: tuple (height, width) for target size, default (80, 80)
        
    Returns:
        Downsampled frame of shape target_size
    """
    # Use cv2.resize for efficient downsampling
    # INTER_AREA is good for downsampling (better than INTER_LINEAR)
    resized = cv2.resize(frame, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    return resized


def frame_differencing(current_frame, previous_frame):
    """
    Compute frame difference to capture motion.
    current_frame - previous_frame
    
    Args:
        current_frame: numpy array of current frame
        previous_frame: numpy array of previous frame (same shape as current_frame)
        
    Returns:
        Frame difference (can have negative values, so we'll take absolute value)
    """
    # Compute difference
    diff = current_frame.astype(np.int16) - previous_frame.astype(np.int16)
    
    # Take absolute value to get magnitude of change
    diff = np.abs(diff)
    
    # Convert back to uint8
    return diff.astype(np.uint8)


def normalize_pixels(frame, scale=255.0):
    """
    Normalize pixel values to range [0, 1] or [-1, 1].
    
    Args:
        frame: numpy array with pixel values 0-255
        scale: normalization scale (255.0 for [0,1], or use different scaling)
        
    Returns:
        Normalized frame with float values
    """
    # Normalize to [0, 1]
    normalized = frame.astype(np.float32) / scale
    
    # Alternative: normalize to [-1, 1] (uncomment if preferred)
    # normalized = (frame.astype(np.float32) / scale) * 2.0 - 1.0
    
    return normalized


def preprocess_frame(frame, previous_frame=None, target_size=(80, 80), normalize=True):
    """
    Complete preprocessing pipeline for a single frame.
    
    Steps:
    1. Convert RGB to grayscale
    2. Downsample to target_size
    3. Frame differencing (if previous_frame provided)
    4. Normalize pixels (if normalize=True)
    
    Args:
        frame: numpy array of shape (210, 160, 3) - raw RGB frame
        previous_frame: optional, numpy array of same shape - previous frame for differencing
        target_size: tuple (height, width) for downsampling, default (80, 80)
        normalize: bool, whether to normalize pixel values to [0, 1]
        
    Returns:
        Preprocessed frame ready for neural network input
    """
    # Step 1: Convert to grayscale
    gray = rgb_to_grayscale(frame)
    
    # Step 2: Downsample
    downsampled = downsample_frame(gray, target_size)
    
    # Step 3: Frame differencing (if previous frame provided)
    if previous_frame is not None:
        # Preprocess previous frame first
        prev_gray = rgb_to_grayscale(previous_frame)
        prev_downsampled = downsample_frame(prev_gray, target_size)
        
        # Compute difference
        processed = frame_differencing(downsampled, prev_downsampled)
    else:
        # No previous frame, just use current frame
        processed = downsampled
    
    # Step 4: Normalize (if requested)
    if normalize:
        processed = normalize_pixels(processed)
    
    return processed


def preprocess_batch(frames, previous_frames=None, target_size=(80, 80), normalize=True):
    """
    Preprocess a batch of frames efficiently.
    
    Args:
        frames: numpy array of shape (batch_size, 210, 160, 3)
        previous_frames: optional, numpy array of same shape
        target_size: tuple for downsampling
        normalize: bool, whether to normalize
        
    Returns:
        Preprocessed batch of shape (batch_size, target_size[0], target_size[1])
    """
    batch_size = frames.shape[0]
    processed_batch = []
    
    for i in range(batch_size):
        prev_frame = previous_frames[i] if previous_frames is not None else None
        processed = preprocess_frame(frames[i], prev_frame, target_size, normalize)
        processed_batch.append(processed)
    
    return np.array(processed_batch)

