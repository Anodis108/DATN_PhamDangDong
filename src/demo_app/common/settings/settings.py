from __future__ import annotations

from dotenv import find_dotenv
from dotenv import load_dotenv
from pydantic import HttpUrl
from pydantic_settings import BaseSettings

from .models import AppSettings

# from .models import ChromaDB
# from .models import PostgresSettings
# test in local
load_dotenv(find_dotenv('.env'), override=True)


class Settings(BaseSettings):
    host_height_service: HttpUrl

    app: AppSettings

    class Config:
        env_nested_delimiter = '__'
