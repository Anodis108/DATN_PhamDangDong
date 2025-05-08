from __future__ import annotations

from common.bases import BaseModel


class AppSettings(BaseModel):
    camera_path: str
    img_logo_path: str
    save_dir: str
