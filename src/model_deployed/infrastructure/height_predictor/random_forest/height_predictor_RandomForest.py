from __future__ import annotations

from functools import cached_property
from typing import List

import joblib
import numpy as np
from common.logs.logs import get_logger
from common.settings import Settings
from sklearn.ensemble import RandomForestRegressor

from ..base_pred import HeightPredictorModel
from ..base_pred import HeightPredictorModelInput
from ..base_pred import HeightPredictorModelOutput

logger = get_logger(__name__)


class HeightPredictorModelRandomForest(HeightPredictorModel):
    settings: Settings

    @cached_property
    def model_loaded(self) -> RandomForestRegressor:
        return joblib.load(self.settings.height_predictor.model_path_random_forest)


    def process(self, inputs: HeightPredictorModelInput) -> HeightPredictorModelOutput:
        pred = self.forward(inputs.x)
        return HeightPredictorModelOutput(pred=pred)

    def forward(self, x: List[List[float]]) -> List[float]:
        model = self.model_loaded
        x_np = np.array(x)  # shape: (n_samples, n_features)
        preds = model.predict(x_np)
        return preds.tolist()
