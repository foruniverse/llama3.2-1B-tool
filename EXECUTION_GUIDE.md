# Execution Reference Guide

## Complete Training Pipeline

### Option 1: Run All Steps Automatically
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

### Option 2: Run Steps Individually

#### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Or use uv
uv sync

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=./models/huggingface_cache
export MODELSCOPE_CACHE=./models/modelscope_cache
```

#### 2. Model Download (15 min)
```bash
python scripts/1_download_model.py
```
Output: `./models/pretrained/Llama-3.2-1B-Instruct/` (~2.1GB)

#### 3. Data Preparation (10 min)
```bash
python scripts/2_prepare_data.py
```
Output: `./data/processed/{train,validation}/`

#### 4. SFT Training (1-2 hours)
```bash
python scripts/3_sft_training.py

# Monitor with TensorBoard
tensorboard --logdir=./logs/tensorboard --port=6006
```
Output: `./models/checkpoints/sft/`

#### 5. GRPO Training (2-4 hours)
```bash
python scripts/4_grpo_training.py
```
Output: `./models/final/`

#### 6. Evaluation (10 min)
```bash
python scripts/5_evaluate.py
```
Output: `./logs/evaluation_results.json`

#### 7. BFCL-V4 Evaluation (30 min)
```bash
python scripts/7_bfcl_evaluation.py
```
Output: `./logs/bfcl_results.json`

#### 8. Inference Examples
```bash
python scripts/6_inference.py
```

## Configuration Files

### Modify Training Parameters

Edit `configs/sft_config.yaml` for SFT:
```yaml
training:
  num_train_epochs: 3
  per_device_train_batch_size: 2
  learning_rate: 2.0e-4

lora:
  lora_r: 16
  lora_alpha: 32
```

Edit `configs/grpo_config.yaml` for GRPO:
```yaml
grpo:
  num_generations: 4
  beta: 0.01
  max_new_tokens: 256
```

Edit `configs/bfcl_config.yaml` for evaluation:
```yaml
bfcl:
  test_dimensions:
    - format_sensitivity
    - hallucination
    - single_turn
```

## Monitoring and Logging

### Check Logs
```bash
# Real-time training log
tail -f logs/sft_training.log
tail -f logs/grpo_training.log

# View evaluation results
cat logs/evaluation_results.json
cat logs/bfcl_results.json
```

### TensorBoard Visualization
```bash
tensorboard --logdir=./logs/tensorboard --port=6006
```
Access at: http://localhost:6006

### GPU Monitoring
```bash
# Real-time GPU usage
nvidia-smi -l 1

# GPU memory profiling
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,nounits -l 1
```

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
# Edit configs/sft_config.yaml or grpo_config.yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 8  # Increase this

# Or enable QLoRA (int4 quantization)
# in configs - additional changes needed
```

### Model Download Issues
```bash
# Set HF mirror
export HF_ENDPOINT=https://hf-mirror.com

# Or download manually from Modelscope
# https://modelscope.cn/models/AI-ModelScope/Llama-3.2-1B-Instruct
```

### CUDA Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU only (very slow)
export CUDA_VISIBLE_DEVICES=-1
```

## Results Interpretation

### Evaluation Results (`logs/evaluation_results.json`)
```json
{
  "metrics": {
    "average_loss": 1.2,
    "perplexity": 3.5,
    "num_eval_samples": 1000
  }
}
```

### BFCL Results (`logs/bfcl_results.json`)
```json
{
  "results": {
    "format_sensitivity": {"score": 0.92},
    "hallucination": {"score": 0.95},
    "single_turn": {"score": 0.89}
  },
  "overall_score": 0.92
}
```

## Model Deployment

### Inference with LoRA
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("./models/pretrained/Llama-3.2-1B-Instruct")
model = PeftModel.from_pretrained(base_model, "./models/final")
tokenizer = AutoTokenizer.from_pretrained("./models/pretrained/Llama-3.2-1B-Instruct")

# Generate
inputs = tokenizer("Your prompt", return_tensors="pt")
outputs = model.generate(**inputs, max_length=256)
print(tokenizer.decode(outputs[0]))
```

### Merge LoRA Weights (Optional)
```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained("./models/final")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./models/merged")
```

### Export to GGUF (Optional)
```bash
# Using llama.cpp tools
python ./llama.cpp/convert.py ./models/merged --outtype f16
```

## Performance Optimization

### Memory Optimization
- ✓ FP16: 2x memory reduction
- ✓ LoRA: 70% parameter reduction
- ✓ Gradient Checkpointing: Trade compute for memory

### Speed Optimization
- Batch size: Balance between speed and memory
- Gradient accumulation: Simulate larger batches
- Mixed precision: Trade accuracy for speed

### Multi-GPU Training
```bash
# Modify scripts to use distributed training
# In training args: use distributed_data_parallel

export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 scripts/3_sft_training.py
```

## Support and Resources

- TRL Documentation: https://huggingface.co/docs/trl
- PEFT Documentation: https://huggingface.co/docs/peft
- BFCL Leaderboard: https://gorilla.cs.berkeley.edu/leaderboard.html
- Llama Models: https://www.llama.com/

## Next Steps After Training

1. **Fine-tune hyperparameters** based on BFCL results
2. **Test on real function calling tasks**
3. **Deploy to production** with optimized model
4. **Monitor performance** in production
5. **Iterate and improve** based on feedback
