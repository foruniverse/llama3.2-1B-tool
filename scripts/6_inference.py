#!/usr/bin/env python3
"""Inference script for function calling with trained model."""

import sys
from pathlib import Path
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.utils import setup_logging, get_device_info

logger = setup_logging("./logs/inference.log")


class FunctionCallingInference:
    """Inference class for function calling tasks."""

    def __init__(self, model_path: str = "./models/final", base_model_path: str = "./models/pretrained/Llama-3.2-1B-Instruct"):
        """Initialize inference model."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
        except ImportError as e:
            logger.error(f"Required packages not installed: {e}")
            raise

        logger.info(f"Loading base model from {base_model_path}...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        logger.info(f"Loading LoRA model from {model_path}...")
        self.model = PeftModel.from_pretrained(self.base_model, model_path, is_trainable=False)
        self.model.eval()

        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = self.model.device
        logger.info(f"Model loaded on device: {self.device}")

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0, top_p: float = 1.0) -> str:
        """Generate response for given prompt."""
        import torch

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        return response

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts."""
        responses = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}...")
            response = self.generate(prompt, **kwargs)
            responses.append(response)
        return responses


def main():
    """Main inference function."""
    logger.info("Starting inference...")
    logger.info(f"Device info: {get_device_info()}")

    try:
        # Initialize model
        inference = FunctionCallingInference()

        # Example function calling prompts
        test_prompts = [
            """Available functions:
- search(query: str): Search for information
- get_weather(location: str): Get weather for location

User: What is the weather in Beijing?
Response:""",

            """Available functions:
- calculate(expression: str): Calculate mathematical expression
- convert_units(value: float, from_unit: str, to_unit: str): Convert units

User: Convert 100 kilometers to miles
Response:""",

            """Available functions:
- send_email(to: str, subject: str, body: str): Send an email
- get_emails(): Get recent emails

User: Send an email to john@example.com with subject "Meeting" and body "Let's meet tomorrow"
Response:""",
        ]

        logger.info(f"Running inference on {len(test_prompts)} prompts...")

        # Generate responses
        responses = inference.batch_generate(
            test_prompts,
            max_new_tokens=256,
            temperature=0.0,
            top_p=1.0,
        )

        # Display results
        logger.info("\n" + "="*80)
        logger.info("INFERENCE RESULTS")
        logger.info("="*80)

        for i, (prompt, response) in enumerate(zip(test_prompts, responses), 1):
            logger.info(f"\n--- Example {i} ---")
            logger.info(f"Prompt:\n{prompt}")
            logger.info(f"Response:\n{response}")

        logger.info("\n" + "="*80)
        logger.info("Inference completed successfully!")

        return True

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
