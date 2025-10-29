#!/usr/bin/env python3
"""
TextAttack Tutorial Automation Script

This script automates the end-to-end TextAttack tutorial workflow:
1. Setup dependencies and NLTK data
2. Peek at the dataset
3. Train a model
4. Evaluate the trained model
5. Run adversarial attacks

Run this script from the project root directory after activating your virtual environment.
"""

import subprocess
import sys
import os
import glob
from pathlib import Path


def run_command(command, description, shell=False):
    """Run a command and handle errors."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*50)
    
    try:
        if shell:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        else:
            result = subprocess.run(command.split(), check=True, capture_output=True, text=True)
        
        print("✓ Success!")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed with exit code {e.returncode}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        return False


def find_latest_model_path(outputs_dir="./outputs"):
    """Find the most recently created model directory."""
    if not os.path.exists(outputs_dir):
        print(f"Outputs directory '{outputs_dir}' not found.")
        return None
    
    # Find all subdirectories in outputs
    model_dirs = []
    for root, dirs, files in os.walk(outputs_dir):
        for dir_name in dirs:
            if "best_model" in dir_name:
                full_path = os.path.join(root, dir_name)
                model_dirs.append((full_path, os.path.getctime(full_path)))
    
    if not model_dirs:
        print("No trained model directories found.")
        return None
    
    # Return the most recently created model
    latest_model = max(model_dirs, key=lambda x: x[1])[0]
    print(f"Found latest model: {latest_model}")
    return latest_model


def main():
    """Main execution function."""
    print("TextAttack Tutorial Automation Script")
    print("=====================================")
    
    # Configuration
    config = {
        "dataset": "rotten_tomatoes",
        "model_name": "distilbert-base-uncased",
        "num_labels": 2,
        "max_length": 64,
        "train_batch_size": 8,
        "eval_batch_size": 16,
        "num_epochs": 3,
        "eval_examples": 50,
        "attack_examples": 20,
        "attack_recipe": "textfooler"
    }
    
    # Step 1: Peek at dataset
    print("\nStep 1: Peeking at dataset")
    peek_cmd = f"textattack peek-dataset --dataset-from-huggingface {config['dataset']}"
    if not run_command(peek_cmd, "Peek dataset"):
        print("Failed to peek at dataset. Continuing...")
    
    # Step 2: Train model
    print("\nStep 2: Training model")
    train_cmd = f"""python -m textattack train \
        --model-name-or-path {config['model_name']} \
        --dataset {config['dataset']} \
        --model-num-labels {config['num_labels']} \
        --model-max-length {config['max_length']} \
        --per-device-train-batch-size {config['train_batch_size']} \
        --per-device-eval-batch-size {config['eval_batch_size']} \
        --num-epochs {config['num_epochs']}"""
    
    if not run_command(train_cmd, "Train model", shell=True):
        print("Training failed. Cannot proceed with evaluation and attacks.")
        sys.exit(1)
    
    # Step 3: Find the trained model path
    print("\nStep 3: Locating trained model")
    model_path = find_latest_model_path()
    if not model_path:
        print("Could not find trained model. Please check the outputs directory.")
        sys.exit(1)
    
    # Step 4: Evaluate model
    print("\nStep 4: Evaluating model")
    eval_cmd = f"""python -m textattack eval \
        --num-examples {config['eval_examples']} \
        --model {model_path} \
        --dataset-from-huggingface {config['dataset']} \
        --dataset-split test"""
    
    if not run_command(eval_cmd, "Evaluate model", shell=True):
        print("Evaluation failed. Continuing to attack step...")
    
    # Step 5: Run adversarial attacks
    print("\nStep 5: Running adversarial attacks")
    attack_cmd = f"""python -m textattack attack \
        --recipe {config['attack_recipe']} \
        --num-examples {config['attack_examples']} \
        --model {model_path} \
        --dataset-from-huggingface {config['dataset']} \
        --dataset-split test"""
    
    if not run_command(attack_cmd, "Run attacks", shell=True):
        print("Attack failed.")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("🎉 TextAttack tutorial completed successfully!")
    print(f"Model saved at: {model_path}")
    print("="*50)


if __name__ == "__main__":
    # Change to the script's directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    main()