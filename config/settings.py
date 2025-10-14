from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os
from typing import Optional

load_dotenv()

class Settings(BaseSettings):
    deriv_auth_token: str = os.environ.get("DERIV_AUTH_TOKEN", "")
    gemini_api_key: str = os.environ.get("GEMINI_API_KEY", "")
    qdrant_api_key: Optional[str] = os.environ.get("QDRANT_API_KEY", None)
    api_host: str = os.environ.get("API_HOST", "")
    api_port: int = int(os.environ.get("API_PORT", 9000))
    ws_host: str = os.environ.get("WS_HOST", "")
    ws_port: int = int(os.environ.get("WS_PORT", 8765))
    redis_url: str = os.environ.get("REDIS_URL", "redis://localhost:6380/0")

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow" 


settings = Settings()
