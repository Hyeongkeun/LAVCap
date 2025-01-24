# Copyright 2023-2024 Xiaomi Corporation and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


class EnsembleModel(torch.nn.Module):
    def __init__(self, models) -> None:
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.front_end = self.models[0].front_end

    def forward_spectrogram(self, x):
        ys = []
        for model in self.models:
            y = model.forward_spectrogram(x)
            ys.append(y)
        return torch.stack(ys, dim=-1).mean(-1)

    def forward(self, x):
        ys = []
        for model in self.models:
            y = model(x)
            ys.append(y)
        return torch.stack(ys, dim=-1).mean(-1)
