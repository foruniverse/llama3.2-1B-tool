#!/usr/bin/env python3
"""SFT Training script for Llama3.2-1B."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.utils import setup_logging, load_yaml_config, get_device_info

logger = setup_logging("./logs/sft_training.log")


def resolve_model_path(config: dict) -> str:
    """Prefer a downloaded local ModelScope model, otherwise use model_name."""
    cache_dir = Path(config["model"].get("cache_dir", "./models/pretrained"))
    path = cache_dir / "AI-ModelScope" / "Llama-3.2-1B-Instruct"

    if (path / "config.json").exists():
        return str(path)

    return config["model"]["model_name"]


def get_torch_dtype(torch_module, bf16: bool, fp16: bool):
    """Pick model load dtype from precision config."""
    if bf16:
        return torch_module.bfloat16
    if fp16:
        return torch_module.float16
    return torch_module.float32


def precision_flags(torch_module, config: dict) -> tuple[bool, bool]:
    """Return safe bf16/fp16 Trainer flags for the current GPU."""
    bf16 = bool(config["optimization"].get("bf16"))
    fp16 = bool(config["optimization"].get("fp16"))

    if bf16 and torch_module.cuda.is_available() and not torch_module.cuda.is_bf16_supported():
        logger.warning("bf16 is configured but not supported by this GPU; falling back to fp16.")
        return False, True

    return bf16, fp16


def drop_text_column(dataset):
    """Keep only tensor columns needed by the trainer."""
    if "text" in dataset.column_names:
        return dataset.remove_columns(["text"])
    return dataset


def train_sft():
    """Run SFT training."""
    logger.info("Starting SFT training...")
    logger.info(f"Device info: {get_device_info()}")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from transformers import Trainer
        from transformers import default_data_collator
        from peft import LoraConfig, get_peft_model
        from trl import SFTTrainer
        from datasets import load_from_disk
        # 【配置选项】如果需要4/8位量化微调（QLoRA），可取消注释下方库
        # from transformers import BitsAndBytesConfig
    except ImportError as e:
        logger.error(f"Required packages not installed: {e}")
        return False

    class LossOnlySFTTrainer(SFTTrainer):
        """SFTTrainer without extra logits-based metrics to fit 8GB GPUs."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            inputs["use_cache"] = False
            return Trainer.compute_loss(
                self,
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

    try:
        # Load configuration
        config = load_yaml_config("./configs/sft_config.yaml")
        bf16, fp16 = precision_flags(torch, config)
        debug_max_steps = int(os.environ.get("SFT_MAX_STEPS", "-1"))
        debug_run = debug_max_steps > 0

        # 【配置选项】GPU卡号设置
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # 使用多张卡

        # 【配置选项】Wandb 模型训练追踪平台设置
        os.environ["WANDB_PROJECT"] = "llama3.2-function-calling"
        # os.environ["WANDB_DISABLED"] = "true" # 禁用 Wandb 追踪

        # ==========================================
        # 1. 加载模型（包含所有主流微调配置项）
        # ==========================================
        model_path = resolve_model_path(config)
        logger.info(f"Loading model: {model_path}")
        
        # 【配置选项】如果想启用 QLoRA 4-bit 量化减小显存，可取消注释下方配置
        # qlora_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     bnb_4bit_use_double_quant=True,
        # )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # 【配置选项】模型精度选择
            dtype=get_torch_dtype(torch, bf16, fp16),
            
            # 【配置选项】显卡分配策略
            device_map="auto",                      # 自动分配
            # device_map={"":"cuda:0"},             # 强制单卡训练（多卡环境推荐）
            
            trust_remote_code=True,
            
            # 【配置选项】注意力机制优化（加速并省显存）
            attn_implementation="flash_attention_2", # 推荐（需要GPU支持及安装 flash-attn）
            # attn_implementation="sdpa",             # PyTorch自带的缩放点积注意力
            
            # 【配置选项】启用量化时取消下方注释
            # quantization_config=qlora_config,
        )
        if config['optimization']['gradient_checkpointing']:
            model.config.use_cache = False

        # ==========================================
        # 2. 加载分词器
        # ==========================================
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,                          # 使用Fast版本分词器加速
            # padding_side="right",                 # 【配置选项】指定填充方向（SFT通常用 right，生成/推理用 left）
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # ==========================================
        # 3. 配置 LoRA 参数
        # ==========================================
        logger.info("Configuring LoRA...")
        peft_config = LoraConfig(
            r=config['lora']['lora_r'],             # 秩大小：常规8或16；复杂任务（如Function Calling）设为32或64
            lora_alpha=config['lora']['lora_alpha'], # 缩放系数：通常设为 r 的 2 倍
            lora_dropout=config['lora']['lora_dropout'], # 防止过拟合的丢弃率
            bias=config['lora']['lora_bias'],       # 偏置参数更新策略：通常为 'none'
            task_type=config['lora']['task_type'],   # 任务类型：因果语言模型为 CAUSAL_LM
            
            # 【配置选项】微调目标模块选择
            target_modules=config['lora']['lora_target_modules'], # 默认从yaml读取
            # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # 全量模块微调（推荐，效果最好）
        )

        model = get_peft_model(model, peft_config)
        logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        # Prepare datasets
        logger.info("Loading processed datasets...")
        train_dataset = drop_text_column(load_from_disk("./data/processed/train"))
        eval_dataset = drop_text_column(load_from_disk("./data/processed/validation"))

        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Eval dataset size: {len(eval_dataset)}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=config['training']['output_dir'], # 模型保存路径
            do_train=True,                          # 执行训练
            do_eval=not debug_run,                  # 执行评估
            num_train_epochs=config['training']['num_train_epochs'], # 训练轮数
            max_steps=debug_max_steps,

            # 根据显存做 trade off
            per_device_train_batch_size=config['training']['per_device_train_batch_size'],  # 训练 batch_size
            per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],    # 评估 batch_size
            gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],  # 梯度累积步数， 逻辑 batch_size = gas * 物理 batch_size


            # [优化器和学习率]
            learning_rate=config['training']['learning_rate'], # 学习率 lora 一般在 5e-5 到 2e-4 之间
            warmup_ratio=config['training']['warmup_ratio'], # 预热步数比例  
            weight_decay=config['training']['weight_decay'], # 权重衰减系数
            max_grad_norm=config['training']['max_grad_norm'],
            lr_scheduler_type=config['training']['lr_scheduler_type'],  # 学习率衰减策略
            optim=config['optimization']['optim'],
            # lr_scheduler_type="cosine",           # 【配置选项】余弦退火（最常用）
            # lr_scheduler_type="linear",           # 【配置选项】线性衰减

            # 【日志、保存与评估策略】
            logging_steps=config['logging']['logging_steps'], # 每隔多少步打印一次日志
            save_steps=config['logging']['save_steps'],       # 每隔多少步保存一次模型
            eval_steps=config['logging']['eval_steps'],       # 每隔多少步进行一次验证集评估
            eval_strategy="no" if debug_run else "steps", # 按步数评估
            save_strategy="no" if debug_run else "steps", # 按步数保存模型
            save_total_limit=config['logging']['save_total_limit'], # 最多保留几个权重检查点（旧的会被自动删除）
            
            # 【计算精度与性能优化】
            fp16=fp16,                              # 是否启用 fp16 混合精度
            bf16=bf16,                              # 是否启用 bf16（如果GPU支持，强烈建议替代fp16）
            gradient_checkpointing=config['optimization']['gradient_checkpointing'], # 激活梯度检查点（时间换空间，大幅降显存）
            
            # 【第三方平台报告】
            report_to="none" if debug_run else config['logging']['report_to'], # 默认从yaml读取，如 "wandb" 或 "tensorboard"
            # report_to="none",                       # 【配置选项】不向任何平台报告
            
            seed=config['seed'],                      # 随机种子，确保实验可复现
            dataloader_pin_memory=True,               # 锁页内存，加速CPU到GPU的数据传输
            dataloader_num_workers=4,                 # 多线程数据加载
            remove_unused_columns=False,              # PEFT forward signature can hide attention_mask
            prediction_loss_only=True,                 # Avoid storing logits during eval
        )

        # Initialize trainer
        logger.info("Initializing SFTTrainer...")
        trainer = LossOnlySFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=default_data_collator,
        )

        # Train
        logger.info("Starting training loop...")
        trainer.train()

        # Save model
        if debug_run:
            logger.info("Debug run completed; skipping final model save.")
        else:
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
