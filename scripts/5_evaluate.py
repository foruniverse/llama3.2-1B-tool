#!/usr/bin/env python3
"""Basic evaluation script for trained models."""

import sys
import os
from pathlib import Path
from typing import Dict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.utils import setup_logging, load_yaml_config, save_json, get_device_info

logger = setup_logging("./logs/evaluate.log")


def evaluate():
    """Run basic evaluation on validation set."""
    logger.info("Starting evaluation...")
    logger.info(f"Device info: {get_device_info()}")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        from datasets import load_from_disk
        from torch.utils.data import DataLoader
    except ImportError as e:
        logger.error(f"Required packages not installed: {e}")
        return False

    try:
        # Load model and tokenizer
        logger.info("Loading trained model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "./models/pretrained/Llama-3.2-1B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        model = PeftModel.from_pretrained(base_model, "./models/final", is_trainable=False)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            "./models/pretrained/Llama-3.2-1B-Instruct",
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load validation dataset
        logger.info("Loading validation dataset...")
        eval_dataset = load_from_disk("./data/processed/validation")

        # Calculate perplexity
        logger.info("Calculating perplexity...")
        total_loss = 0.0
        num_batches = 0

        dataloader = DataLoader(eval_dataset, batch_size=4, shuffle=False)

        with torch.no_grad():
            for batch in dataloader:
                input_ids = torch.tensor(batch["input_ids"]).to(model.device)
                labels = torch.tensor(batch["labels"]).to(model.device)

                outputs = model(input_ids=input_ids, labels=labels)
                total_loss += outputs.loss.item()
                num_batches += 1

                if num_batches % 10 == 0:
                    logger.info(f"Processed {num_batches} batches...")

        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        logger.info(f"Average Loss: {avg_loss:.4f}")
        logger.info(f"Perplexity: {perplexity:.4f}")

        # Save results
        results = {
            "task": "basic_evaluation",
            "metrics": {
                "average_loss": float(avg_loss),
                "perplexity": float(perplexity),
                "num_eval_samples": len(eval_dataset),
            }
        }

        save_json(results, "./logs/evaluation_results.json")
        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to ./logs/evaluation_results.json")

        return True

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = evaluate()
    sys.exit(0 if success else 1)
