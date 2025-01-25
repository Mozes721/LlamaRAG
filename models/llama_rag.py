from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class LlamaRAG:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        """
        Initialize the Llama model for local use.
        
        Args:
            model_name (str): Name of the Hugging Face model to load.
        """
        # Load Llama tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"  # Automatically manages device placement
        )

        # Device handling for inputs only
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate_response(self, prompt: str, max_length: int = 200) -> str:
        """
        Generate a response using the Llama model.
        
        Args:
            prompt (str): The input prompt for the model.
            max_length (int): Maximum length of the generated response.
        
        Returns:
            str: The generated response.
        """
        # Tokenize the input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move input tensors to the same device as the model
        inputs = inputs.to(self.device)
        
        # Generate response using the model
        outputs = self.model.generate(inputs.input_ids, max_length=max_length)
        
        # Decode and return the response
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def handle_query(self, query: str) -> str:
        """
        Handle user queries and generate a response using the Llama model.
        
        Args:
            query (str): The user's query.
        
        Returns:
            str: The model's response to the query.
        """
        # Generate a response directly using the model
        prompt = (
            f"The user asked: '{query}'. "
            "Please provide a helpful and polite response to the user."
        )
        return self.generate_response(prompt)