# scripts/prepare_data.py

import json
from datasets import Dataset
import logging
import os
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_preparation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_role_info(text):
    """Extract name and role from text like 'Akshay (Platform Engineer)'"""
    try:
        if '(' in text and ')' in text:
            name = text.split('(')[0].strip()
            role = text.split('(')[1].split(')')[0].strip()
            return name, role
        return text.strip(), "Unknown"
    except Exception as e:
        logger.error(f"Error extracting role info from '{text}': {str(e)}")
        return text.strip(), "Unknown"

def prepare_data(input_file, output_file):
    logger.info(f"Starting data preparation from {input_file}")
    
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Initialize data structures
    role_info = {}
    formatted_data = []
    current_conversation = []
    
    # First pass: Extract all role information
    logger.info("Extracting role information...")
    for line in lines:
        if ':' in line:
            speaker_part = line.split(':', 1)[0].strip()
            name, role = extract_role_info(speaker_part)
            if role != "Unknown":
                role_info[name] = role
                logger.info(f"Found role: {name} - {role}")

    # Second pass: Create training examples
    logger.info("Creating training examples...")
    
    # Type 1: Role-based QA pairs
    logger.info("Generating role-based QA pairs...")
    for name, role in role_info.items():
        # Company name extraction
        company = role.split(',')[0].split('of ')[-1] if 'of' in role else "the company"
        
        # Various question formats for the same information
        qa_pairs = [
            {
                "instruction": f"Who is the {role} of {company}?",
                "input": "",
                "output": f"{name} is the {role}.",
                "type": "role_query"
            },
            {
                "instruction": f"What is {name}'s role?",
                "input": "",
                "output": f"{name} is the {role}.",
                "type": "person_query"
            },
            {
                "instruction": f"What position does {name} hold?",
                "input": "",
                "output": f"{name} holds the position of {role}.",
                "type": "position_query"
            }
        ]
        formatted_data.extend(qa_pairs)

    # Type 2: Conversation flow
    logger.info("Processing conversation flow...")
    conversation_text = ""
    context_window = []
    
    for line in lines:
        line = line.strip()
        if line:
            if ':' in line:
                speaker_part, message = line.split(':', 1)
                name, role = extract_role_info(speaker_part)
                
                # Add to current context window
                context_window.append({
                    "speaker": name,
                    "role": role,
                    "message": message.strip()
                })
                
                # Generate conversation examples when we have enough context
                if len(context_window) >= 3:
                    context_text = "\n".join([
                        f"{turn['speaker']} ({turn['role']}): {turn['message']}"
                        for turn in context_window[:-1]
                    ])
                    
                    formatted_data.append({
                        "instruction": "Continue the conversation based on the context:",
                        "input": context_text,
                        "output": f"{context_window[-1]['speaker']} ({context_window[-1]['role']}): {context_window[-1]['message']}",
                        "type": "conversation_flow"
                    })
                    
                    # Slide the window
                    context_window = context_window[-2:]
            else:
                # Continuation of previous message
                if context_window:
                    context_window[-1]["message"] += " " + line

    # Create dataset
    logger.info(f"Created {len(formatted_data)} training examples")
    dataset = Dataset.from_list(formatted_data)
    
    # Save dataset
    logger.info(f"Saving dataset to {output_file}")
    dataset.save_to_disk(output_file)
    
    # Log examples of each type
    for type_ in ["role_query", "person_query", "conversation_flow"]:
        example = next((item for item in formatted_data if item["type"] == type_), None)
        if example:
            logger.info(f"\nExample of {type_}:")
            logger.info(f"Instruction: {example['instruction']}")
            logger.info(f"Input: {example['input']}")
            logger.info(f"Output: {example['output']}")

    return dataset

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    input_file = 'data/slack.txt'
    output_file = 'data/prepared_data'
    
    dataset = prepare_data(input_file, output_file)