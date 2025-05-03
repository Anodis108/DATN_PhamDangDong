from __future__ import annotations

from dotenv import find_dotenv
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

from .models import BoxDetectorSettings
from .models import HeightCalculatorSettings
from .models import HeightPredictorSettings
from .models import PoseDetectorSettings
# test in local
load_dotenv(find_dotenv('.env'), override=True)


class Settings(BaseSettings):
    box_detector: BoxDetectorSettings
    height_predictor: HeightPredictorSettings
    height_calculator: HeightCalculatorSettings
    pose_detector: PoseDetectorSettings

    class Config:
        env_nested_delimiter = '__'
