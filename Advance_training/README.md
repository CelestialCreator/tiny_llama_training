# README for TinyLlama Scripts

This document provides details on the functionality of the scripts included in the project. It also outlines prerequisites, directory setup, and commands for running the scripts on both M1 (Apple Silicon) and NVIDIA GPUs.

---

## Directory Structure

Ensure the following directories are created before running the scripts:

- `data`: For storing input and output data.
- `logs`: For storing log files generated during script execution.
- `artifacts`: For saving fine-tuned models and related artifacts.

Create the directories if they don't exist:

```bash
mkdir -p data logs artifacts
```

---

## Prerequisites

### Required Python Packages

Install the required Python packages using the following command:

```bash
pip install transformers torch datasets peft trl psutil
```

---

## Scripts Overview

### 1. `scripts/prepare_data.py`

This script processes raw conversation data from a text file to create a structured dataset for training language models.

#### Key Features:

- Extracts roles and names from text lines, e.g., "Akshay (Platform Engineer)".
- Generates role-based QA pairs for fine-tuning.
- Processes conversation context into training examples.

#### Input/Output:

- **Input File**: `data/slack.txt` (conversation data in plain text).
- **Output File**: `data/prepared_data` (processed dataset).

#### Run Command:

```bash
python scripts/prepare_data.py
```

---

### 2. `scripts/test_model.py`

This script tests a fine-tuned TinyLlama model by generating responses to sample instructions or user-provided questions.

#### Key Features:

- Supports interactive testing mode for custom queries.
- Logs system information and response metrics.
- Includes sample questions for automated testing.

#### Run Command:

```bash
python scripts/test_model.py
```

#### GPU Configuration:

- **M1 (Apple Silicon):** Leverages MPS (Metal Performance Shaders).
- **NVIDIA GPU:** Automatically uses CUDA if available.

---

### 3. `scripts/train_model.py`

This script fine-tunes the TinyLlama model using the processed dataset.

#### Key Features:

- Configures LoRA (Low-Rank Adaptation) for efficient fine-tuning.
- Uses a pre-trained base model (`TinyLlama/TinyLlama-1.1B-Chat-v0.3`).
- Saves the fine-tuned model in the `artifacts` directory.

#### Input/Output:

- **Input File**: `data/prepared_data` (processed dataset).
- **Output Directory**: `artifacts/tinyllama_finetuned_<timestamp>`.

#### Run Command:

```bash
python scripts/train_model.py
```

---

## Special Notes

### Setting Up Logs Directory

Ensure that the `logs` directory exists to store log files. Create it if it doesn’t exist:

```bash
mkdir -p logs
```

### Running on M1 (Apple Silicon)

For Apple Silicon devices, the scripts automatically detect and use MPS. If you encounter issues, ensure you have the latest versions of PyTorch and related libraries:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Running on NVIDIA GPU

For NVIDIA GPUs, ensure you have CUDA installed and the appropriate PyTorch version:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Example Workflow

1. **Prepare Data:**

```bash
python scripts/prepare_data.py
```

2. **Train Model:**

```bash
python scripts/train_model.py
```

3. **Test Model:**

```bash
python scripts/test_model.py
```

---
