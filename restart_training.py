"""
Restart Training Script

This script cleans up old checkpoints and restarts training from scratch.
Use this when you want to start fresh with new hyperparameters or fixes.
"""

import os
import shutil

def restart_training():
    """Clean up old checkpoints and prepare for fresh training."""
    
    checkpoint_dir = "checkpoints"
    
    print("=" * 70)
    print("  RESTARTING TRAINING")
    print("=" * 70)
    
    # Check if checkpoint directory exists
    if os.path.exists(checkpoint_dir):
        # Count files
        files = os.listdir(checkpoint_dir)
        num_files = len(files)
        
        if num_files > 0:
            print(f"\nFound {num_files} files in {checkpoint_dir}/")
            print("Files will be deleted:")
            for f in files[:5]:  # Show first 5
                print(f"  - {f}")
            if num_files > 5:
                print(f"  ... and {num_files - 5} more")
            
            response = input("\nDelete these files and restart? (yes/no): ").strip().lower()
            
            if response in ['yes', 'y']:
                # Delete directory
                shutil.rmtree(checkpoint_dir)
                print(f"✓ Deleted {checkpoint_dir}/")
                
                # Recreate empty directory
                os.makedirs(checkpoint_dir)
                print(f"✓ Created fresh {checkpoint_dir}/")
                
                print("\n" + "=" * 70)
                print("  READY TO TRAIN")
                print("=" * 70)
                print("\nRun: python train.py")
                print()
            else:
                print("\nCancelled. No files were deleted.")
        else:
            print(f"\n{checkpoint_dir}/ is already empty. Ready to train!")
            print("\nRun: python train.py")
    else:
        # Create directory
        os.makedirs(checkpoint_dir)
        print(f"\n✓ Created {checkpoint_dir}/")
        print("\n" + "=" * 70)
        print("  READY TO TRAIN")
        print("=" * 70)
        print("\nRun: python train.py")
        print()

if __name__ == "__main__":
    restart_training()
