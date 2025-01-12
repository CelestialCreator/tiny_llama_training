
For Advance_training/README.md:
```markdown
# TinyLlama Advanced Training

Advanced implementation with role-based queries and conversation flow.

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
Directory Structure
CopyAdvance_training/
├── artifacts/         # Trained models
├── data/             # Training data
├── logs/             # Execution logs
├── scripts/          # Implementation scripts
└── requirements.txt  # Dependencies
Usage

Data Preparation:

bashCopypython scripts/prepare_data.py

Training:

bashCopypython scripts/train_model.py

Testing:

bashCopypython scripts/test_model.py
