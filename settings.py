from pydantic import BaseSettings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

class Settings(BaseSettings):
    deriv_auth_token: str
    gemini_api_key: str

    class Config:
        env_file = ".env"

settings = Settings()