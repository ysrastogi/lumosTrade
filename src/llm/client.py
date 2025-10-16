from google import genai
from google.genai import types
from config.settings import settings
from typing import Union, List, Optional, Dict, Any

class GeminiClient: 
    
    def __init__(self, model: str = "gemini-2.5-flash", embedding_model: str = "gemini-embedding-001"):
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = model
        self.embedding_model = embedding_model
    
    def generate(self, 
                 prompt: Union[str, List[str]], 
                 temperature: float = None,
                 system_instruction: str = None,
                 disable_thinking: bool = False,
                 top_p: float = None,
                 top_k: int = None,
                 max_output_tokens: int = None,
                 **kwargs) -> str:
        

        if isinstance(prompt, str):
            contents = prompt
        else:
            contents = prompt
            
        # Build configuration
        config_args = {}
        
        if temperature is not None:
            config_args["temperature"] = temperature
            
        if system_instruction is not None:
            config_args["system_instruction"] = system_instruction
            
        if disable_thinking:
            config_args["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            
        if top_p is not None:
            config_args["top_p"] = top_p
            
        if top_k is not None:
            config_args["top_k"] = top_k
            
        if max_output_tokens is not None:
            config_args["max_output_tokens"] = max_output_tokens
        
        config = types.GenerateContentConfig(**config_args) if config_args else None
        
        generate_args = {"model": self.model, "contents": contents}
        if config:
            generate_args["config"] = config
            
        for key, value in kwargs.items():
            if key not in generate_args:
                generate_args[key] = value
        
        response = self.client.models.generate_content(**generate_args)
        return response.text
    
    def embed(self, 
              contents: Union[str, List[str]], 
              model: str = None,
              **kwargs) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the given content(s) using the specified model.
        
        Args:
            contents: A string or list of strings to embed
            model: The embedding model to use (defaults to self.embedding_model)
            **kwargs: Additional parameters to pass to the embed_content method
            
        Returns:
            For a single input string: A list of floats representing the embedding
            For multiple input strings: A list of lists of floats representing embeddings for each input
        """
        embed_model = model or self.embedding_model
        
        embed_args = {"model": embed_model, "contents": contents}
        for key, value in kwargs.items():
            if key not in embed_args:
                embed_args[key] = value
        
        result = self.client.models.embed_content(**embed_args)
        
        # If input was a single string, return just the first embedding
        if isinstance(contents, str):
            return result.embeddings[0]
        
        # If input was a list of strings, return all embeddings
        return result.embeddings


gemini = GeminiClient()