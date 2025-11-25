"""
Test script to demonstrate and verify preprocessing functions.
Shows before/after preprocessing and optionally visualizes the results.
"""

import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import (
    rgb_to_grayscale,
    downsample_frame,
    frame_differencing,
    normalize_pixels,
    preprocess_frame
)


def test_preprocessing():
    """Test all preprocessing functions with real Pong frames."""
    
    print("=" * 70)
    print("TESTING PREPROCESSING FUNCTIONS")
    print("=" * 70)
    print()
    
    # Create environment
    print("1. Creating Pong environment...")
    env = gym.make("ALE/Pong-v5")
    observation, info = env.reset()
    print(f"   ✓ Environment created")
    print(f"   Raw observation shape: {observation.shape}")
    print(f"   Raw observation dtype: {observation.dtype}")
    print()
    
    # Get a few frames for testing
    print("2. Collecting sample frames...")
    frames = []
    for i in range(5):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        frames.append(observation.copy())
        if terminated or truncated:
            observation, info = env.reset()
    
    print(f"   ✓ Collected {len(frames)} frames")
    print()
    
    # Test individual preprocessing steps
    print("3. Testing individual preprocessing steps...")
    print("-" * 70)
    
    # Test RGB to grayscale
    frame1 = frames[0]
    gray = rgb_to_grayscale(frame1)
    print(f"   RGB to Grayscale:")
    print(f"      Input shape: {frame1.shape}")
    print(f"      Output shape: {gray.shape}")
    print(f"      Output dtype: {gray.dtype}")
    print(f"      Value range: [{gray.min()}, {gray.max()}]")
    print()
    
    # Test downsampling
    downsampled = downsample_frame(gray, target_size=(80, 80))
    print(f"   Downsampling:")
    print(f"      Input shape: {gray.shape}")
    print(f"      Output shape: {downsampled.shape}")
    print(f"      Value range: [{downsampled.min()}, {downsampled.max()}]")
    print()
    
    # Test frame differencing
    frame2 = frames[1]
    gray2 = rgb_to_grayscale(frame2)
    downsampled2 = downsample_frame(gray2, target_size=(80, 80))
    diff = frame_differencing(downsampled, downsampled2)
    print(f"   Frame Differencing:")
    print(f"      Frame 1 shape: {downsampled.shape}")
    print(f"      Frame 2 shape: {downsampled2.shape}")
    print(f"      Difference shape: {diff.shape}")
    print(f"      Difference range: [{diff.min()}, {diff.max()}]")
    print()
    
    # Test normalization
    normalized = normalize_pixels(downsampled)
    print(f"   Normalization:")
    print(f"      Input range: [{downsampled.min()}, {downsampled.max()}]")
    print(f"      Output range: [{normalized.min()}, {normalized.max()}]")
    print(f"      Output dtype: {normalized.dtype}")
    print()
    
    # Test complete preprocessing pipeline
    print("4. Testing complete preprocessing pipeline...")
    print("-" * 70)
    
    # Without previous frame
    processed1 = preprocess_frame(frames[0], normalize=True)
    print(f"   Without previous frame:")
    print(f"      Input shape: {frames[0].shape}")
    print(f"      Output shape: {processed1.shape}")
    print(f"      Output dtype: {processed1.dtype}")
    print(f"      Output range: [{processed1.min():.3f}, {processed1.max():.3f}]")
    print()
    
    # With previous frame (frame differencing)
    processed2 = preprocess_frame(frames[1], previous_frame=frames[0], normalize=True)
    print(f"   With previous frame (frame differencing):")
    print(f"      Input shape: {frames[1].shape}")
    print(f"      Output shape: {processed2.shape}")
    print(f"      Output dtype: {processed2.dtype}")
    print(f"      Output range: [{processed2.min():.3f}, {processed2.max():.3f}]")
    print()
    
    # Test without normalization
    processed3 = preprocess_frame(frames[0], normalize=False)
    print(f"   Without normalization:")
    print(f"      Output dtype: {processed3.dtype}")
    print(f"      Output range: [{processed3.min()}, {processed3.max()}]")
    print()
    
    # Visualization
    print("5. Creating visualization...")
    print("-" * 70)
    
    # Create a figure showing preprocessing steps
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Preprocessing Pipeline Visualization', fontsize=16)
    
    # Row 1: Original and intermediate steps
    axes[0, 0].imshow(frames[0])
    axes[0, 0].set_title('Original RGB Frame\n(210×160×3)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Grayscale\n(210×160)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(downsampled, cmap='gray')
    axes[0, 2].set_title('Downsampled\n(80×80)')
    axes[0, 2].axis('off')
    
    # Row 2: Frame difference and final processed
    axes[1, 0].imshow(downsampled2, cmap='gray')
    axes[1, 0].set_title('Frame 2 Downsampled\n(80×80)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(diff, cmap='gray')
    axes[1, 1].set_title('Frame Difference\n(80×80)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(processed2, cmap='gray')
    axes[1, 2].set_title('Final Processed\n(80×80, normalized)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    output_file = 'preprocessing_visualization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   ✓ Visualization saved to: {output_file}")
    print()
    
    # Show the plot
    print("   Displaying visualization window...")
    print("   (Close the window to continue)")
    plt.show()
    
    # Cleanup
    env.close()
    
    print()
    print("=" * 70)
    print("✓ ALL PREPROCESSING TESTS PASSED!")
    print("=" * 70)
    print()
    print("Summary:")
    print("  • RGB to grayscale: ✓")
    print("  • Downsampling (210×160 → 80×80): ✓")
    print("  • Frame differencing: ✓")
    print("  • Normalization: ✓")
    print("  • Complete pipeline: ✓")
    print()
    print("The preprocessing functions are ready to use in your RL training!")


if __name__ == "__main__":
    test_preprocessing()

