from __future__ import annotations

from typing import Any
from typing import List

from common.bases import BaseModel


class APIOutput(BaseModel):
    info_text: List[Any]
