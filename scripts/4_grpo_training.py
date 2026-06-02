#!/usr/bin/env python3
"""GRPO Training script for Llama3.2-1B."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.utils import setup_logging, ensure_dir, load_yaml_config, get_device_info

logger = setup_logging("./logs/grpo_training.log")


def train_grpo():
    """Run GRPO training."""
    logger.info("Starting GRPO training...")
    logger.info(f"Device info: {get_device_info()}")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import LoraConfig, get_peft_model, PeftModel
        from trl import GRPOTrainer, GRPOConfig
        from datasets import load_from_disk
    except ImportError as e:
        logger.error(f"Required packages not installed: {e}")
        return False

    try:
        # Load configuration
        config = load_yaml_config("./configs/grpo_config.yaml")

        # Set environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["WANDB_PROJECT"] = "llama3.2-function-calling"

        # Load base model
        logger.info("Loading base model...")
        model_path = config['model']['model_name_or_path']

        base_model = AutoModelForCausalLM.from_pretrained(
            "./models/pretrained/Llama-3.2-1B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load SFT model (with LoRA)
        logger.info(f"Loading SFT model from: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path, is_trainable=True)

        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "./models/pretrained/Llama-3.2-1B-Instruct",
            trust_remote_code=True,
            use_fast=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Prepare datasets
        logger.info("Loading processed datasets...")
        train_dataset = load_from_disk("./data/processed/train")
        eval_dataset = load_from_disk("./data/processed/validation")

        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Eval dataset size: {len(eval_dataset)}")

        # GRPO Configuration
        grpo_config = GRPOConfig(
            output_dir=config['training']['output_dir'],
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            num_train_epochs=config['training']['num_train_epochs'],
            per_device_train_batch_size=config['training']['per_device_train_batch_size'],
            gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
            learning_rate=config['training']['learning_rate'],
            lr_scheduler_type=config['training']['lr_scheduler_type'],
            logging_steps=config['logging']['logging_steps'],
            save_steps=config['logging']['save_steps'],
            eval_steps=config['logging']['eval_steps'],
            save_total_limit=config['logging']['save_total_limit'],
            fp16=config['optimization']['fp16'],
            gradient_checkpointing=config['optimization']['gradient_checkpointing'],
            report_to=config['logging']['report_to'],
            seed=config['seed'],

            # GRPO specific parameters
            num_generations=config['grpo']['num_generations'],
            temperature=config['grpo']['temperature'],
            top_p=config['grpo']['top_p'],
            beta=config['grpo']['beta'],
            max_new_tokens=config['grpo']['max_new_tokens'],
        )

        # Initialize trainer
        logger.info("Initializing GRPOTrainer...")
        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            args=grpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # Train
        logger.info("Starting GRPO training loop...")
        trainer.train()

        # Save final model
        logger.info(f"Saving final model to {config['training']['output_dir']}")
        trainer.save_model(config['training']['output_dir'])
        tokenizer.save_pretrained(config['training']['output_dir'])

        logger.info("GRPO training completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Error during GRPO training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = train_grpo()
    sys.exit(0 if success else 1)
