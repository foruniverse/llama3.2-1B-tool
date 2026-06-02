#!/usr/bin/env python3
"""Prepare training data from HuggingFace dataset."""

import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.utils import setup_logging, ensure_dir, load_yaml_config

logger = setup_logging("./logs/prepare_data.log")


def prepare_data():
    """Load and prepare training data."""
    logger.info("Starting data preparation...")

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError:
        logger.error("Required packages not installed. Please run: pip install datasets transformers")
        return False

    try:
        # Load configuration
        sft_config = load_yaml_config("./configs/sft_config.yaml")
        dataset_name = sft_config["data"]["dataset_name"]
        validation_split = sft_config["data"]["validation_split_percentage"] / 100

        logger.info(f"Loading dataset: {dataset_name}")

        # Load dataset from HuggingFace
        dataset = load_dataset(dataset_name)

        # Get the train split
        if "train" in dataset:
            train_data = dataset["train"]
        else:
            train_data = dataset

        logger.info(f"Dataset size: {len(train_data)}")

        # Load tokenizer
        model_name = sft_config["model"]["model_name"]
        logger.info(f"Loading tokenizer for: {model_name}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "./models/pretrained/AI-ModelScope/Llama-3.2-1B-Instruct",
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"Could not load from local path: {e}. Using fallback.")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Process function
        def preprocess_function(examples):
            """Preprocess examples for training."""
            # Combine instruction, input, and output for function calling
            texts = []
            for i in range(len(examples.get("instruction", []))):
                instruction = examples.get("instruction", [""])[i]
                input_text = examples.get("input", [""])[i]
                output_text = examples.get("output", [""])[i]

                # Format for function calling task
                if input_text:
                    text = f"{instruction}\n{input_text}\n{output_text}"
                else:
                    text = f"{instruction}\n{output_text}"

                texts.append(text)

            # Tokenize
            tokenized = tokenizer(
                texts,
                truncation=True,
                max_length=sft_config["training"]["max_seq_length"],
                padding="max_length",
            )

            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        logger.info("Processing dataset...")

        # Apply preprocessing
        processed_dataset = train_data.map(
            preprocess_function,
            batched=True,
            batch_size=32,
            remove_columns=train_data.column_names,
            num_proc=sft_config["data"]["preprocessing_num_workers"],
        )

        # Split dataset
        logger.info(f"Splitting dataset (validation: {validation_split*100}%)")
        split_dataset = processed_dataset.train_test_split(
            test_size=validation_split,
            seed=42
        )

        # Save processed data
        output_dir = "./data/processed"
        ensure_dir(output_dir)

        logger.info(f"Saving processed data to {output_dir}")
        split_dataset["train"].save_to_disk(f"{output_dir}/train")
        split_dataset["test"].save_to_disk(f"{output_dir}/validation")

        logger.info(f"Training samples: {len(split_dataset['train'])}")
        logger.info(f"Validation samples: {len(split_dataset['test'])}")
        logger.info("Data preparation completed successfully!")

        return True

    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = prepare_data()
    sys.exit(0 if success else 1)
