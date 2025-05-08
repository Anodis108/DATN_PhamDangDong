from __future__ import annotations

from dotenv import find_dotenv
from dotenv import load_dotenv
from pydantic import HttpUrl
from pydantic_settings import BaseSettings

from .models import WriteCSVSettings

# test in local
load_dotenv(find_dotenv('.env'), override=True)


class Settings(BaseSettings):
    host_box_detector: HttpUrl
    host_pose_detector: HttpUrl
    host_height_calculator: HttpUrl
    host_height_predictor: HttpUrl

    write_csv: WriteCSVSettings

    class Config:
        env_nested_delimiter = '__'
