import json
from datasets import Dataset

def prepare_data(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    formatted_data = []
    conversations = []
    current_conversation = []

    for line in lines:
        line = line.strip()
        if line:
            if ':' in line:
                speaker, message = line.split(':', 1)
                speaker = speaker.strip()
                message = message.strip()
                current_conversation.append({
                    "role": "user" if "CEO" not in speaker and "CTO" not in speaker else "assistant",
                    "content": f"{speaker}: {message}"
                })
            else:
                if current_conversation:
                    current_conversation[-1]["content"] += " " + line.strip()

        # Add conversation when there's a significant gap or at the end
        if not line and current_conversation:
            conversations.append(current_conversation)
            current_conversation = []

    if current_conversation:
        conversations.append(current_conversation)

    # Format data for TinyLlama
    for conv in conversations:
        text = ""
        for turn in conv:
            prefix = "<|user|>\n" if turn["role"] == "user" else "<|assistant|>\n"
            text += prefix + turn["content"] + "\n"
        text += "<|end|>"
        
        formatted_data.append({
            "text": text,
            "input_ids": None,  # Will be filled by tokenizer
            "labels": None      # Will be filled by tokenizer
        })

    dataset = Dataset.from_list(formatted_data)
    dataset.save_to_disk(output_file)

if __name__ == "__main__":
    prepare_data('data/slack.txt', 'data/prepared_data')