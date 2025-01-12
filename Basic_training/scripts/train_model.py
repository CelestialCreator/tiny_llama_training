import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_from_disk
from peft import LoraConfig
from trl import SFTTrainer

def tokenize_function(examples, tokenizer):
    # Tokenize the texts
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Set labels to be the same as input_ids for causal language modeling
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

def finetune_tinyllama(data_path, base_model_id, output_dir):
    # Load dataset
    dataset = load_from_disk(data_path)
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
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
        logging_steps=10,
        save_strategy="epoch",
        push_to_hub=False,
        fp16=False
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        peft_config=peft_config,
        processing_class=tokenizer
    )

    # Train and save
    trainer.train()
    trainer.save_model(output_dir)

if __name__ == "__main__":
    finetune_tinyllama(
        'data/prepared_data', 
        'TinyLlama/TinyLlama-1.1B-Chat-v0.3', 
        './tinyllama_finetuned'
    )