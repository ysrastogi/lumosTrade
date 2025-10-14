from google import genai
from google.genai import types
from config.settings import settings
from typing import Union, List, Optional, Dict, Any

class GeminiClient: 
    
    def __init__(self, model: str = "gemini-2.5-flash"):
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = model
    
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

gemini = GeminiClient()