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
        # 【配置选项】如果需要4/8位量化微调（QLoRA），可取消注释下方库
        # from transformers import BitsAndBytesConfig
    except ImportError as e:
        logger.error(f"Required packages not installed: {e}")
        return False

    try:
        # Load configuration
        config = load_yaml_config("./configs/sft_config.yaml")

        # 【配置选项】GPU卡号设置
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # 使用多张卡

        # 【配置选项】Wandb 模型训练追踪平台设置
        os.environ["WANDB_PROJECT"] = "llama3.2-function-calling"
        # os.environ["WANDB_DISABLED"] = "true" # 禁用 Wandb 追踪

        # ==========================================
        # 1. 加载模型（包含所有主流微调配置项）
        # ==========================================
        logger.info(f"Loading model: {config['model']['model_name']}")
        
        # 【配置选项】如果想启用 QLoRA 4-bit 量化减小显存，可取消注释下方配置
        # qlora_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     bnb_4bit_use_double_quant=True,
        # )

        model = AutoModelForCausalLM.from_pretrained(
            config['model']['cache_dir'] + "/AI-ModelScope/Llama-3.2-1B-Instruct",
            # 【配置选项】模型精度选择
            torch_dtype=torch.float16,              # 默认混合精度
            # torch_dtype=torch.bfloat16,           # 推荐（RTX 3090/4090, A100 等安培架构及以上GPU使用，更稳定）
            # torch_dtype=torch.float32,            # 全精度（显存消耗极大，不推荐）
            
            # 【配置选项】显卡分配策略
            device_map="auto",                      # 自动分配
            # device_map={"":"cuda:0"},             # 强制单卡训练（多卡环境推荐）
            
            trust_remote_code=True,
            
            # 【配置选项】注意力机制优化（加速并省显存）
            # attn_implementation="flash_attention_2", # 推荐（需要GPU支持及安装 flash-attn）
            # attn_implementation="sdpa",             # PyTorch自带的缩放点积注意力
            
            # 【配置选项】启用量化时取消下方注释
            # quantization_config=qlora_config,
        )

        # ==========================================
        # 2. 加载分词器
        # ==========================================
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            config['model']['cache_dir'] + "/Llama-3.2-1B-Instruct",
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
        train_dataset = load_from_disk("./data/processed/train")
        eval_dataset = load_from_disk("./data/processed/validation")

        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Eval dataset size: {len(eval_dataset)}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=config['training']['output_dir'], # 模型保存路径
            overwrite_output_dir=True,              # 允许覆盖输出目录
            do_train=True,                          # 执行训练
            do_eval=True,                           # 执行评估
            num_train_epochs=config['training']['num_train_epochs'], # 训练轮数

            # 根据显存做 trade off
            per_device_train_batch_size=config['training']['per_device_train_batch_size'],  # 训练 batch_size
            per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],    # 评估 batch_size
            gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],  # 梯度累积步数， 逻辑 batch_size = gas * 物理 batch_size

            # [优化器和学习率]
            learning_rate=config['training']['learning_rate'], # 学习率 lora 一般在 5e-5 到 2e-4 之间
            warmup_ratio=config['training']['warmup_ratio'], # 预热步数比例  
            weight_decay=config['training']['weight_decay'], # 权重衰减系数
            lr_scheduler_type=config['training']['lr_scheduler_type'],  # 学习率衰减策略
            # lr_scheduler_type="cosine",           # 【配置选项】余弦退火（最常用）
            # lr_scheduler_type="linear",           # 【配置选项】线性衰减

            # 【日志、保存与评估策略】
            logging_steps=config['logging']['logging_steps'], # 每隔多少步打印一次日志
            save_steps=config['logging']['save_steps'],       # 每隔多少步保存一次模型
            eval_steps=config['logging']['eval_steps'],       # 每隔多少步进行一次验证集评估
            # evaluation_strategy="steps",          # 【配置选项】按步数评估（新版HF中建议使用 eval_strategy）
            # save_strategy="steps",                # 【配置选项】按步数保存模型
            save_total_limit=config['logging']['save_total_limit'], # 最多保留几个权重检查点（旧的会被自动删除）
            
            # 【计算精度与性能优化】
            fp16=config['optimization']['fp16'],                       # 是否启用 fp16 混合精度
            # bf16=True,                                               # 【配置选项】是否启用 bf16（如果GPU支持，强烈建议替代fp16）
            gradient_checkpointing=config['optimization']['gradient_checkpointing'], # 激活梯度检查点（时间换空间，大幅降显存）
            
            # 【第三方平台报告】
            report_to=config['logging']['report_to'], # 默认从yaml读取，如 "wandb" 或 "tensorboard"
            # report_to="none",                       # 【配置选项】不向任何平台报告
            
            seed=config['seed'],                      # 随机种子，确保实验可复现
            dataloader_pin_memory=True,               # 锁页内存，加速CPU到GPU的数据传输
            dataloader_num_workers=4,                 # 多线程数据加载
        )

        # Initialize trainer
        logger.info("Initializing SFTTrainer...")
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            max_seq_length=config['optimization']['max_seq_length'],

            # 【避坑提示】如果你的数据集是已经分词处理好的（包含input_ids），请注释掉下面这行 dataset_text_field
            dataset_text_field="input_ids", 
            
            # 【配置选项】如果数据集是原始文本（未分词），可取消注释下方配置
            # dataset_text_field="text",             # 指定数据集中存储原始文本的字段名
            
            # 【配置选项】高效数据拼装选项
            # packing=True,                          # 将多条短文本拼接至 max_seq_length 长度，大幅提高长文本训练效率
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
