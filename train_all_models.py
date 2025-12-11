#!/usr/bin/env python3
"""
Script to train all model variants sequentially.
Runs: base, dueling, mha, dueling_mha
"""
import subprocess
import sys
import time
from datetime import datetime

MODELS = ["base", "dueling", "mha", "dueling_mha"]

def run_training(model_name: str) -> bool:
    """Run training for a specific model."""
    print("\n" + "="*80)
    print(f"Starting training for model: {model_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    try:
        # Run training script
        result = subprocess.run(
            [sys.executable, "train.py", "--model", model_name],
            check=True,
            cwd="."
        )
        
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        
        print("\n" + "="*80)
        print(f"✓ Completed training for model: {model_name}")
        print(f"  Duration: {hours}h {minutes}m {seconds}s")
        print("="*80 + "\n")
        
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print("\n" + "="*80)
        print(f"✗ Training failed for model: {model_name}")
        print(f"  Error code: {e.returncode}")
        print(f"  Duration before failure: {elapsed:.1f}s")
        print("="*80 + "\n")
        return False
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print("\n" + "="*80)
        print(f"⚠ Training interrupted for model: {model_name}")
        print(f"  Duration before interruption: {elapsed:.1f}s")
        print("="*80 + "\n")
        return False


def main():
    """Run all models sequentially."""
    print("\n" + "="*80)
    print("DQN Model Training Suite")
    print("="*80)
    print(f"Models to train: {', '.join(MODELS)}")
    print(f"Total models: {len(MODELS)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    overall_start = time.time()
    results = {}
    
    for i, model in enumerate(MODELS, 1):
        print(f"\n[{i}/{len(MODELS)}] Processing model: {model}")
        
        success = run_training(model)
        results[model] = success
        
        if not success:
            print(f"\n⚠ Warning: Model {model} failed. Continuing with next model...")
        
        # Small delay between models
        if i < len(MODELS):
            print("\nWaiting 5 seconds before starting next model...\n")
            time.sleep(5)
    
    # Final summary
    overall_elapsed = time.time() - overall_start
    hours = int(overall_elapsed // 3600)
    minutes = int((overall_elapsed % 3600) // 60)
    seconds = int(overall_elapsed % 60)
    
    print("\n" + "="*80)
    print("Training Suite Complete")
    print("="*80)
    print(f"Total duration: {hours}h {minutes}m {seconds}s")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults:")
    for model, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {model:15s} - {status}")
    print("="*80 + "\n")
    
    # Exit with error if any model failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Training suite interrupted by user. Exiting...")
        sys.exit(130)

