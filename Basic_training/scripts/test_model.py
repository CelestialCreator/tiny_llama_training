from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import psutil
import logging
import platform
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_testing.log'),
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
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        
        # Move model to appropriate device
        self.model = self.model.to(self.device)
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")
        
        # Log GPU info if available
        if torch.cuda.is_available():
            logger.info(f"GPU Model: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")
        
        # Log system info
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

    def generate_response(self, prompt, max_length=100, temperature=0.7):
        try:
            # Format prompt
            formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
            logger.info(f"\nProcessing prompt: {prompt}")
            
            # Track resources and time
            start_time = time.time()
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Generate response with proper device placement
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():  # Add this for inference
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Move outputs back to CPU for decoding
            outputs = outputs.cpu()
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Calculate metrics
            end_time = time.time()
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            processing_time = end_time - start_time
            memory_used = final_memory - initial_memory
            
            # Log device-specific metrics
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**2
                logger.info(f"GPU Memory used: {gpu_memory:.2f} MB")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("MPS (M1 GPU) is being used")
            
            logger.info(f"Response generated in {processing_time:.2f} seconds")
            logger.info(f"System Memory used: {memory_used:.2f} MB")
            logger.info(f"Final response length: {len(response.split())} words")
            
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
    # Sample questions based on your training data
    test_questions = [
        {
            "question": "What are the key features of this deployment?",
            "expected": "Should mention API optimization, logging framework, and security updates"
        },
        {
            "question": "What is the deployment strategy being used?",
            "expected": "Should mention blue-green deployment strategy"
        },
        {
            "question": "Who is responsible for DevOps in the team?",
            "expected": "Should mention Yatin as DevOps"
        },
        {
            "question": "What is the deployment window?",
            "expected": "Should mention 2 hours deployment window"
        },
        {
            "question": "How are the logs being handled?",
            "expected": "Should mention centralized logging system and S3 archival"
        }
    ]
    
    logger.info("\n=== Starting Sample Questions Test ===")
    
    results = []
    total_time = 0
    total_memory = 0
    
    for test in test_questions:
        logger.info(f"\nTesting Question: {test['question']}")
        logger.info(f"Expected response should contain: {test['expected']}")
        
        response, metrics = tester.generate_response(test['question'])
        
        if metrics:
            total_time += metrics['processing_time']
            total_memory += metrics['memory_used']
            
        results.append({
            'question': test['question'],
            'response': response,
            'metrics': metrics
        })
        
    # Log summary statistics
    logger.info("\n=== Test Summary ===")
    logger.info(f"Total questions processed: {len(test_questions)}")
    logger.info(f"Average processing time: {total_time/len(test_questions):.2f} seconds")
    logger.info(f"Average memory used: {total_memory/len(test_questions):.2f} MB")
    
    return results

if __name__ == "__main__":
    # Initialize tester
    model_path = './tinyllama_finetuned'
    tester = ModelTester(model_path)
    
    # Run interactive mode
    logger.info("\n=== Starting Interactive Testing Mode ===")
    print("\nEnter 'quit' to exit")
    
    while True:
        user_input = input("\nEnter your question: ")
        if user_input.lower() == 'quit':
            break
            
        response, metrics = tester.generate_response(user_input)
        print(f"\nResponse: {response}")
        if metrics:
            print(f"\nMetrics:")
            print(f"Processing time: {metrics['processing_time']:.2f} seconds")
            print(f"Memory used: {metrics['memory_used']:.2f} MB")
            print(f"Response length: {metrics['response_length']} words")
    
    # Run sample questions test
    print("\nRunning sample questions test...")
    results = test_sample_questions(tester)
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'test_results_{timestamp}.txt', 'w') as f:
        for result in results:
            f.write(f"\nQuestion: {result['question']}\n")
            f.write(f"Response: {result['response']}\n")
            if result['metrics']:
                f.write(f"Processing time: {result['metrics']['processing_time']:.2f} seconds\n")
                f.write(f"Memory used: {result['metrics']['memory_used']:.2f} MB\n")
            f.write("\n" + "="*50 + "\n")
            