#!/usr/bin/env python3
"""SFT Training script for Llama3.2-1B."""

import sys
import os
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.utils import setup_logging, ensure_dir, load_yaml_config, get_device_info

logger = setup_logging("./logs/sft_training.log")


def train_sft():
    """Run SFT training."""
    logger.info("Starting SFT training...")
    logger.info(f"Device info: {get_device_info()}")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import LoraConfig, get_peft_model
        from trl import SFTTrainer
        from datasets import load_from_disk
    except ImportError as e:
        logger.error(f"Required packages not installed: {e}")
        return False

    try:
        # Load configuration
        config = load_yaml_config("./configs/sft_config.yaml")

        # Set environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["WANDB_PROJECT"] = "llama3.2-function-calling"

        # Load model
        logger.info(f"Loading model: {config['model']['model_name']}")
        model = AutoModelForCausalLM.from_pretrained(
            config['model']['cache_dir'] + "/Llama-3.2-1B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            config['model']['cache_dir'] + "/Llama-3.2-1B-Instruct",
            trust_remote_code=True,
            use_fast=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Configure LoRA
        logger.info("Configuring LoRA...")
        peft_config = LoraConfig(
            r=config['lora']['lora_r'],
            lora_alpha=config['lora']['lora_alpha'],
            lora_dropout=config['lora']['lora_dropout'],
            bias=config['lora']['lora_bias'],
            task_type=config['lora']['task_type'],
            target_modules=config['lora']['lora_target_modules'],
        )

        model = get_peft_model(model, peft_config)
        logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        # Prepare datasets
        logger.info("Loading processed datasets...")
        train_dataset = load_from_disk("./data/processed/train")
        eval_dataset = load_from_disk("./data/processed/validation")

        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Eval dataset size: {len(eval_dataset)}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=config['training']['output_dir'],
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            num_train_epochs=config['training']['num_train_epochs'],
            per_device_train_batch_size=config['training']['per_device_train_batch_size'],
            per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
            gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
            learning_rate=config['training']['learning_rate'],
            lr_scheduler_type=config['training']['lr_scheduler_type'],
            warmup_ratio=config['training']['warmup_ratio'],
            weight_decay=config['training']['weight_decay'],
            logging_steps=config['logging']['logging_steps'],
            save_steps=config['logging']['save_steps'],
            eval_steps=config['logging']['eval_steps'],
            save_total_limit=config['logging']['save_total_limit'],
            fp16=config['optimization']['fp16'],
            gradient_checkpointing=config['optimization']['gradient_checkpointing'],
            report_to=config['logging']['report_to'],
            seed=config['seed'],
            dataloader_pin_memory=True,
            dataloader_num_workers=4,
        )

        # Initialize trainer
        logger.info("Initializing SFTTrainer...")
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="input_ids",
            max_seq_length=config['optimization']['max_seq_length'],
        )

        # Train
        logger.info("Starting training loop...")
        trainer.train()

        # Save model
        logger.info(f"Saving model to {config['training']['output_dir']}")
        trainer.save_model(config['training']['output_dir'])
        tokenizer.save_pretrained(config['training']['output_dir'])

        logger.info("SFT training completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = train_sft()
    sys.exit(0 if success else 1)
