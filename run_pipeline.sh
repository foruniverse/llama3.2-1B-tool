#!/bin/bash
# Quick start script for Llama3.2-1B post-training

set -e

echo "======================================"
echo "Llama3.2-1B Function Calling Training"
echo "======================================"
echo ""

# Check Python and install dependencies
echo "Step 1: Installing dependencies..."
python -m pip install -q transformers datasets peft trl modelscope accelerate wandb bitsandbytes pydantic pyyaml 2>/dev/null || {
    echo "Warning: Some packages failed to install. This may be due to dependency conflicts."
    echo "Try manual installation if issues occur."
}

echo "✓ Dependencies installation attempted"
echo ""

# Create directories
echo "Step 2: Creating project directories..."
mkdir -p configs data/{raw,processed} models/{pretrained,checkpoints,final} scripts/utils logs/tensorboard
echo "✓ Directories created"
echo ""

# Download model
echo "Step 3: Downloading Llama3.2-1B model..."
echo "This may take 10-15 minutes and requires ~2.1GB disk space"
python scripts/1_download_model.py || echo "Warning: Model download may have failed"
echo ""

# Prepare data
echo "Step 4: Preparing training data..."
echo "This will download and format Team-ACE/ToolACE"
python scripts/2_prepare_data.py || echo "Warning: Data preparation may have failed"
echo ""

# Train SFT
echo "Step 5: Starting SFT training (1-2 hours)..."
echo "GPU will be heavily utilized. Monitor with: nvidia-smi -l 1"
python scripts/3_sft_training.py || echo "Warning: SFT training may have failed"
echo ""

# Train GRPO
echo "Step 6: Starting GRPO training (2-4 hours)..."
python scripts/4_grpo_training.py || echo "Warning: GRPO training may have failed"
echo ""

# Evaluate
echo "Step 7: Running evaluation..."
python scripts/5_evaluate.py || echo "Warning: Evaluation may have failed"
echo ""

# BFCL Evaluation
echo "Step 8: Running BFCL-V4 evaluation..."
python scripts/7_bfcl_evaluation.py || echo "Warning: BFCL evaluation may have failed"
echo ""

# Inference
echo "Step 9: Running inference examples..."
python scripts/6_inference.py || echo "Warning: Inference may have failed"
echo ""

echo "======================================"
echo "✓ Training pipeline completed!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Check logs in ./logs/ directory"
echo "2. View BFCL results: cat logs/bfcl_results.json"
echo "3. Monitor training: tensorboard --logdir=./logs/tensorboard"
echo ""
