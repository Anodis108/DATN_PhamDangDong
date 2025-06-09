from __future__ import annotations

from functools import cached_property
from typing import List

import torch
import numpy as np
from torch import nn
from common.logs.logs import get_logger
from common.settings import Settings

from ..base_pred import HeightPredictorModel
from ..base_pred import HeightPredictorModelInput
from ..base_pred import HeightPredictorModelOutput

logger = get_logger(__name__)


class HeightNet(nn.Module):
    def __init__(self, input_size=6):
        super(HeightNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output(x)


class HeightPredictorModelNN(HeightPredictorModel):
    settings: Settings

    @cached_property
    def model_loaded(self) -> HeightNet:
        model = HeightNet(input_size=6)
        model = torch.load(self.settings.height_predictor.model_path_height_net, weights_only=False)
        # model.load_state_dict(state_dict)
        model.eval()
        return model

    def process(self, inputs: HeightPredictorModelInput) -> HeightPredictorModelOutput:
        pred = self.forward(inputs.x)
        return HeightPredictorModelOutput(pred=pred)

    def forward(self, x: List[List[float]]) -> List[float]:
        model = self.model_loaded
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(x_tensor).squeeze(1).numpy()
        return outputs.tolist()
