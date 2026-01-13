from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List
import json


class Settings(BaseSettings):
    # API Settings
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Stroke Monitoring API"
    DEBUG: bool = False

    # CORS - list of allowed origins
    CORS_ORIGINS: str = '["http://localhost:3000"]'

    # Model Settings
    MODEL_PATH: str = "./models/multimodal_model.pth"
    MODEL_DEVICE: str = "cpu"

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 10
    RATE_LIMIT_WINDOW: int = 60

    # Security
    API_KEY: str | None = None

    @property
    def cors_origins_list(self) -> List[str]:
        return json.loads(self.CORS_ORIGINS)

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
