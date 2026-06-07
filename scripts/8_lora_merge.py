import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

def merge_lora():
    project_root = Path(__file__).parent.parent
    base_model_path = project_root / "models" / "pretrained" / "AI-ModelScope" / "Llama-3.2-1B-Instruct"
    lora_path = project_root / "models" / "checkpoints" / "sft" / "checkpoint-300"
    output_dir = project_root / "models" / "final-300"
    
    print(f"Loading tokenizer from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    print(f"Loading base model from {base_model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu"  # Usually better to load in CPU for saving RAM before merging, or "auto" if it fits.
    )
    
    print(f"Loading LoRA weights from {lora_path}...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    print("Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done! The merged model is saved at:", output_dir)

if __name__ == "__main__":
    print(AutoTokenizer)
    merge_lora()
