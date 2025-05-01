from __future__ import annotations

from dotenv import find_dotenv
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

from .models import BoxDetectorSettings
from .models import TextDetectorSettings

# test in local
load_dotenv(find_dotenv('.env'), override=True)


class Settings(BaseSettings):
    box_detector: BoxDetectorSettings
    text_detector: TextDetectorSettings

    class Config:
        env_nested_delimiter = '__'
