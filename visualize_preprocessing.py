"""
Visualize the Pong environment with preprocessing applied.
Shows original frames and preprocessed frames side by side in real-time.
"""

import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from preprocessing import preprocess_frame

# Global variables for animation
fig = None
ax1 = None
ax2 = None
im1 = None
im2 = None
env = None
previous_frame = None

def init_visualization():
    """Initialize the visualization window."""
    global fig, ax1, ax2, im1, im2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('Pong Environment: Original vs Preprocessed', fontsize=16, fontweight='bold')
    
    # Initialize with black images
    im1 = ax1.imshow(np.zeros((210, 160, 3), dtype=np.uint8))
    ax1.set_title('Original Frame\n(210×160×3 RGB)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    im2 = ax2.imshow(np.zeros((80, 80), dtype=np.uint8), cmap='gray')
    ax2.set_title('Preprocessed Frame\n(80×80 Grayscale Difference, Normalized)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    
    return im1, im2

def update_frame(frame_num):
    """Update function for animation."""
    global env, previous_frame, im1, im2
    
    if env is None:
        return im1, im2
    
    # Take a random action
    action = env.action_space.sample()
    
    # Step the environment
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Update original frame
    im1.set_array(observation)
    
    # Preprocess the frame
    if previous_frame is not None:
        processed = preprocess_frame(observation, previous_frame=previous_frame, normalize=True)
    else:
        processed = preprocess_frame(observation, normalize=True)
    
    # Update preprocessed frame (convert back to 0-255 for display)
    # Enhance contrast for better visibility
    processed_display = (processed * 255).astype(np.uint8)
    
    # If the frame is mostly black (frame difference is small), enhance it
    if processed_display.max() < 50:  # Very dark frame
        # Stretch the contrast
        if processed_display.max() > processed_display.min():
            processed_display = ((processed_display - processed_display.min()) / 
                               (processed_display.max() - processed_display.min()) * 255).astype(np.uint8)
    
    im2.set_array(processed_display)
    
    # Update previous frame
    previous_frame = observation.copy()
    
    # Reset if episode ended
    if terminated or truncated:
        observation, info = env.reset()
        previous_frame = None
    
    return im1, im2

def visualize_preprocessing_live(num_frames=200):
    """Create a live visualization of preprocessing."""
    global env, previous_frame
    
    print("Creating Pong environment...")
    env = gym.make("ALE/Pong-v5")
    observation, info = env.reset()
    previous_frame = None
    
    print("Initializing visualization...")
    im1, im2 = init_visualization()
    
    print(f"Starting animation for {num_frames} frames...")
    print("Close the window to stop.")
    print()
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, 
        update_frame, 
        frames=num_frames,
        interval=50,  # 50ms = 20 FPS
        blit=True,
        repeat=True
    )
    
    plt.show()
    
    # Cleanup
    env.close()
    print("Visualization closed.")

def visualize_preprocessing_static(num_frames=10):
    """Create a static visualization showing multiple frames."""
    print("Creating Pong environment...")
    env = gym.make("ALE/Pong-v5")
    observation, info = env.reset()
    previous_frame = None
    
    print(f"Collecting {num_frames} frames...")
    frames = []
    processed_frames = []
    
    for i in range(num_frames):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        frames.append(observation.copy())
        
        # Preprocess
        if previous_frame is not None:
            processed = preprocess_frame(observation, previous_frame=previous_frame, normalize=True)
        else:
            processed = preprocess_frame(observation, normalize=True)
        
        processed_frames.append(processed)
        previous_frame = observation.copy()
        
        if terminated or truncated:
            observation, info = env.reset()
            previous_frame = None
    
    env.close()
    
    # Create visualization with 3 rows: original, raw preprocessed, enhanced
    fig, axes = plt.subplots(3, num_frames, figsize=(2*num_frames, 7))
    fig.suptitle('Original vs Preprocessed Frames (Raw and Enhanced)', fontsize=14, fontweight='bold')
    
    for i in range(num_frames):
        # Original frames (top row)
        axes[0, i].imshow(frames[i])
        axes[0, i].set_title(f'Frame {i+1}\nOriginal RGB', fontsize=8)
        axes[0, i].axis('off')
        
        # Preprocessed frames - raw (middle row)
        processed_display = (processed_frames[i] * 255).astype(np.uint8)
        axes[1, i].imshow(processed_display, cmap='gray')
        min_val = processed_frames[i].min()
        max_val = processed_frames[i].max()
        axes[1, i].set_title(f'Frame {i+1}\nRaw (range: [{min_val:.3f}, {max_val:.3f}])', fontsize=8)
        axes[1, i].axis('off')
        
        # Preprocessed frames - enhanced (bottom row)
        if processed_display.max() > processed_display.min():
            enhanced = ((processed_display - processed_display.min()) / 
                       (processed_display.max() - processed_display.min()) * 255).astype(np.uint8)
        else:
            enhanced = processed_display
        axes[2, i].imshow(enhanced, cmap='gray')
        axes[2, i].set_title(f'Frame {i+1}\nEnhanced Contrast', fontsize=8)
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    output_file = 'preprocessing_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Static visualization saved to: {output_file}")
    
    # Show the plot
    print("Displaying visualization...")
    plt.show()
    
    print("Done!")

if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("PREPROCESSING VISUALIZATION")
    print("=" * 70)
    print()
    print("Choose visualization mode:")
    print("  1. Live animation (real-time, interactive)")
    print("  2. Static comparison (multiple frames side-by-side)")
    print()
    
    choice = input("Enter choice (1 or 2, default=2): ").strip()
    
    if choice == "1":
        print()
        print("Starting live animation...")
        print("(This will open a window showing real-time preprocessing)")
        print()
        visualize_preprocessing_live(num_frames=200)
    else:
        print()
        print("Creating static comparison...")
        print()
        visualize_preprocessing_static(num_frames=10)

