from __future__ import annotations

from common.bases import BaseModel


class CardAlignSettings(BaseModel):
    baseimg_path: str
    per_match: float
