from __future__ import annotations

from common.bases import BaseModel


class PostgresSettings(BaseModel):
    username: str
    password: str
    host: str
    db: str
