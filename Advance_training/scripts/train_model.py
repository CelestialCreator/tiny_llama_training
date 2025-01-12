# scripts/train_model.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_from_disk
from peft import LoraConfig
from trl import SFTTrainer
import logging
import os
import psutil
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def log_system_info():
    """Log system information"""
    memory = psutil.virtual_memory()
    logger.info(f"System Memory: Total={memory.total/1e9:.1f}GB, Available={memory.available/1e9:.1f}GB")
    logger.info(f"CPU Count: {psutil.cpu_count()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        logger.info("Apple M1 GPU (MPS) is available")
    logger.info(f"PyTorch Version: {torch.__version__}")

def tokenize_function(examples, tokenizer):
    """Tokenize the dataset examples"""
    try:
        # Combine instruction, input, and output
        prompts = []
        for i in range(len(examples['instruction'])):
            prompt = f"<|user|>\n{examples['instruction'][i]}"
            if examples['input'][i]:
                prompt += f"\nContext: {examples['input'][i]}"
            prompt += f"\n<|assistant|>\n{examples['output'][i]}\n<|end|>"
            prompts.append(prompt)

        tokenized = tokenizer(
            prompts,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt"
        )
        
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    except Exception as e:
        logger.error(f"Error in tokenization: {str(e)}")
        raise

def finetune_tinyllama(data_path, base_model_id, output_dir):
    try:
        logger.info("Starting fine-tuning process")
        log_system_info()
        
        # Load dataset
        logger.info(f"Loading dataset from {data_path}")
        dataset = load_from_disk(data_path)
        logger.info(f"Dataset loaded with {len(dataset)} examples")
        
        # Initialize tokenizer and model
        logger.info("Loading tokenizer and model")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Device selection
        device_map = "mps" if torch.backends.mps.is_available() else "auto"
        logger.info(f"Using device: {device_map}")
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map=device_map
        )

        # Tokenize dataset
        logger.info("Tokenizing dataset")
        tokenized_dataset = dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )

        # Configure LoRA
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=1,
            logging_steps=1,
            save_strategy="epoch",
            push_to_hub=False,
            fp16=False,
            optim="adamw_torch",
        )

        # Initialize trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            peft_config=peft_config,
            processing_class=tokenizer
        )

        # Train
        logger.info("Starting training")
        trainer.train()
        
        # Save
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs('logs', exist_ok=True)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'artifacts/tinyllama_finetuned_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    finetune_tinyllama(
        'data/prepared_data',
        'TinyLlama/TinyLlama-1.1B-Chat-v0.3',
        output_dir
    )