# TinyLlama Fine-Tuning and Testing Pipeline

This repository contains the scripts to prepare data, fine-tune the TinyLlama language model using LoRA (Low-Rank Adaptation), and test the fine-tuned model. Below, you'll find a detailed explanation of the code and steps to run the pipeline effectively.

---

## Table of Contents
1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Pipeline Explanation](#pipeline-explanation)
    - [1. Data Preparation](#1-data-preparation)
    - [2. Fine-Tuning](#2-fine-tuning)
    - [3. Testing](#3-testing)
4. [Running the Pipeline](#running-the-pipeline)
5. [Sample Testing](#sample-testing)
6. [Logging and Metrics](#logging-and-metrics)

---

## Overview
This project demonstrates how to:
- Prepare conversation-style data for training TinyLlama.
- Fine-tune the model using LoRA for efficient adaptation.
- Test the fine-tuned model and evaluate its performance.

The pipeline uses popular libraries like Hugging Face Transformers, PyTorch, and LoRA.

---

## Requirements

Ensure you have the following dependencies installed:

```bash
pip install transformers torch datasets peft trl
```

---

## Pipeline Explanation

### 1. Data Preparation
**File:** `prepare_data()`

This script prepares conversation-style data for training by:
- Reading a text file (`slack.txt`) containing conversations.
- Formatting the data into a structure suitable for training.
- Adding special tokens (`<|user|>`, `<|assistant|>`, `<|end|>`) for clear distinction between roles.
- Saving the prepared dataset to disk for further processing.

#### Code Highlights:
- **Role Mapping:** Determines the speaker role as `user` or `assistant` based on keywords in the speaker's name (e.g., `CEO`, `CTO`).
- **Conversation Parsing:** Groups messages into conversations, distinguishing them with gaps.
- **Output Format:**
  - Adds special tokens like `<|user|>` and `<|assistant|>` for each message.
  - Saves the formatted dataset using `datasets.Dataset`.

#### Usage:
```bash
python prepare_data.py
```
The input is `data/slack.txt`, and the output is saved as `data/prepared_data`.

---

### 2. Fine-Tuning
**File:** `finetune_tinyllama()`

This script fine-tunes the TinyLlama model using LoRA for parameter-efficient training. It leverages Hugging Face's `transformers` and `trl` libraries for streamlined fine-tuning.

#### Key Features:
- **Tokenizer:** Prepares input texts with truncation and padding.
- **LoRA Configuration:**
  - `r`: Rank for decomposition.
  - `lora_alpha`: Scaling factor.
  - `lora_dropout`: Dropout rate to prevent overfitting.
- **Training Arguments:** Configures batch size, learning rate, number of epochs, and logging.
- **Trainer:** Uses `SFTTrainer` from `trl` to integrate LoRA into training.

#### Usage:
```bash
python finetune_tinyllama.py
```
The script loads data from `data/prepared_data` and fine-tunes the base model (`TinyLlama/TinyLlama-1.1B-Chat-v0.3`). The fine-tuned model is saved to `./tinyllama_finetuned`.

---

### 3. Testing
**File:** `ModelTester`

This script allows:
- Interactive testing of the fine-tuned model.
- Automated testing using predefined sample questions.

#### Key Features:
- **Device Selection:** Automatically selects GPU (CUDA or MPS) or CPU based on availability.
- **Resource Logging:** Logs memory usage, processing time, and system details for debugging.
- **Prompt Formatting:** Uses the same token format as in training (`<|user|>`, `<|assistant|>`).

#### Usage:
Run the script interactively:
```bash
python model_tester.py
```
Enter a prompt to generate a response, or type `quit` to exit.

You can also test predefined questions:
```bash
python model_tester.py --test-samples
```
The results will be saved to a timestamped file (e.g., `test_results_20250113_150000.txt`).

---

## Running the Pipeline

### Step 1: Data Preparation
Prepare the data for training:
```bash
python prepare_data.py
```
Ensure your conversation data is in `data/slack.txt`.

### Step 2: Fine-Tuning
Fine-tune the model on the prepared dataset:
```bash
python finetune_tinyllama.py
```
The fine-tuned model will be saved in `./tinyllama_finetuned`.

### Step 3: Testing
Test the fine-tuned model interactively:
```bash
python model_tester.py
```
Or test predefined questions:
```bash
python model_tester.py --test-samples
```

---

## Sample Testing

Predefined questions for testing include:
- "What are the key features of this deployment?"
- "What is the deployment strategy being used?"
- "Who is responsible for DevOps in the team?"

Each question includes expected elements in the response, such as "blue-green deployment" or "centralized logging system."

---

## Logging and Metrics

### Metrics Captured:
- **Processing Time:** Time taken to generate a response.
- **Memory Usage:** Tracks memory used during generation.
- **Response Length:** Measures the number of words in the response.
- **System Info:** Logs OS, Python version, and available memory.

### Log Outputs:
Logs are written to `model_testing.log` and include:
- GPU/CPU utilization.
- Errors or warnings encountered during generation.
- Summary statistics for automated tests.

---

## Summary
This pipeline provides a complete workflow for fine-tuning and testing a conversational AI model using TinyLlama. With its modular design and extensive logging, it is easy to adapt to different datasets and use cases. For questions or issues, feel free to reach out!

