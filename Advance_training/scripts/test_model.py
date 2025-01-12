# scripts/test_model.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import psutil
import logging
import platform
from datetime import datetime
import os
from pathlib import Path

# Set up project paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
os.makedirs(os.path.join(PROJECT_ROOT, 'logs'), exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, 'logs', 'model_testing.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelTester:
    def __init__(self, model_path):
        logger.info(f"Initializing ModelTester with model from: {model_path}")
        
        # Device selection with proper checks
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using CUDA GPU")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using Apple M1 MPS")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU - No GPU available")
        
        # Load tokenizer and model
        start_time = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Optimize model loading for M1
        model_kwargs = {
            'torch_dtype': torch.float32,
            'trust_remote_code': True,
        }
        
        if self.device == torch.device("mps"):
            # Add M1-specific optimizations
            model_kwargs.update({
                'low_cpu_mem_usage': True,
                'use_cache': False
            })
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")
        
        self.log_system_info()

    def log_system_info(self):
        memory = psutil.virtual_memory()
        logger.info(f"System Information:")
        logger.info(f"OS: {platform.system()} {platform.version()}")
        logger.info(f"Python Version: {platform.python_version()}")
        logger.info(f"PyTorch Version: {torch.__version__}")
        logger.info(f"Device being used: {self.device}")
        logger.info(f"Total RAM: {memory.total / (1024**3):.2f} GB")
        logger.info(f"Available RAM: {memory.available / (1024**3):.2f} GB")
        logger.info(f"CPU Usage: {psutil.cpu_percent()}%")

    def generate_response(self, instruction, context="", max_new_tokens=50, temperature=0.7):
        try:
            # Format prompt based on training format
            if context:
                formatted_prompt = f"<|user|>\n{instruction}\nContext: {context}\n<|assistant|>\n"
            else:
                formatted_prompt = f"<|user|>\n{instruction}\n<|assistant|>\n"
            
            logger.info(f"\nProcessing instruction: {instruction}")
            if context:
                logger.info(f"With context: {context}")
            
            start_time = time.time()
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    num_return_sequences=1,
                    early_stopping=True
                )
            
            outputs = outputs.cpu()
            
            # Clean up response to remove prompt tokens
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(formatted_prompt, "").strip()
            response = response.replace("<|user|>", "").replace("<|assistant|>", "").replace("<|end|>", "").strip()
            
            # Calculate metrics
            end_time = time.time()
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            processing_time = end_time - start_time
            memory_used = final_memory - initial_memory
            
            # Log metrics
            logger.info(f"Response generated in {processing_time:.2f} seconds")
            logger.info(f"System Memory used: {memory_used:.2f} MB")
            logger.info(f"Response length: {len(response.split())} words")
            
            return response, {
                'processing_time': processing_time,
                'memory_used': memory_used,
                'response_length': len(response.split()),
                'device': str(self.device)
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return str(e), None

def test_sample_questions(tester):
    test_questions = [
        {
            "instruction": "Who is the CTO of Mavonic?",
            "context": "",
            "expected": "Should mention Madhar as CTO"
        },
        {
            "instruction": "What is Yogesh's role in the team?",
            "context": "",
            "expected": "Should mention Dev Lead"
        },
        {
            "instruction": "What are the key features of this deployment?",
            "context": "",
            "expected": "Should mention API optimization, logging framework, and security updates"
        },
        {
            "instruction": "How are the logs being handled in the system?",
            "context": "",
            "expected": "Should mention centralized logging and S3 archival"
        },
        {
            "instruction": "Continue the conversation based on the context:",
            "context": "Akshay (Platform Engineer): Morning team! All pre-prod validations are complete.",
            "expected": "Should provide relevant deployment-related response"
        }
    ]
    
    logger.info("\n=== Starting Sample Questions Test ===")
    
    results = []
    total_time = 0
    total_memory = 0
    
    for test in test_questions:
        logger.info(f"\nTesting Question: {test['instruction']}")
        logger.info(f"Expected response should contain: {test['expected']}")
        
        response, metrics = tester.generate_response(test['instruction'], test['context'])
        
        if metrics:
            total_time += metrics['processing_time']
            total_memory += metrics['memory_used']
            
        results.append({
            'instruction': test['instruction'],
            'context': test['context'],
            'response': response,
            'metrics': metrics,
            'expected': test['expected']
        })
    
    # Log summary statistics
    logger.info("\n=== Test Summary ===")
    logger.info(f"Total questions processed: {len(test_questions)}")
    logger.info(f"Average processing time: {total_time/len(test_questions):.2f} seconds")
    logger.info(f"Average memory used: {total_memory/len(test_questions):.2f} MB")
    
    return results

if __name__ == "__main__":
    # Initialize tester with the specific model path
    model_path = os.path.join(PROJECT_ROOT, 'artifacts', 'tinyllama_finetuned_20250112_201332')
    tester = ModelTester(model_path)
    
    # Interactive testing mode
    logger.info("\n=== Starting Interactive Testing Mode ===")
    print("\nEnter 'quit' to exit")
    print("You can provide both instruction and context (optional)")
    
    while True:
        instruction = input("\nEnter your instruction/question: ")
        if instruction.lower() == 'quit':
            break
            
        context = input("Enter context (press Enter if none): ")
        
        response, metrics = tester.generate_response(instruction, context)
        print(f"\nResponse: {response}")
        if metrics:
            print(f"\nMetrics:")
            print(f"Processing time: {metrics['processing_time']:.2f} seconds")
            print(f"Memory used: {metrics['memory_used']:.2f} MB")
            print(f"Response length: {metrics['response_length']} words")
    
    # Run capability tests
    print("\nRunning sample questions test...")
    results = test_sample_questions(tester)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(PROJECT_ROOT, 'logs', f'test_results_{timestamp}.txt')
    
    with open(results_file, 'w') as f:
        f.write("=== Test Results ===\n\n")
        for result in results:
            f.write(f"Instruction: {result['instruction']}\n")
            if result['context']:
                f.write(f"Context: {result['context']}\n")
            f.write(f"Expected: {result['expected']}\n")
            f.write(f"Response: {result['response']}\n")
            if result['metrics']:
                f.write(f"Processing time: {result['metrics']['processing_time']:.2f} seconds\n")
                f.write(f"Memory used: {result['metrics']['memory_used']:.2f} MB\n")
            f.write("\n" + "="*50 + "\n")