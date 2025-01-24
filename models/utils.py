# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
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
from transformers import StoppingCriteria


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

### Optimal Transport (Sinkhorn) in Log-space
def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, num_sinkhorn_iter=50) -> torch.Tensor:
        """ Perform Sinkhorn Normalization in Log-space for stability"""
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(num_sinkhorn_iter):
            u = log_mu - torch.logsumexp(Z + v.unsqueeze(0), dim=1)
            v = log_nu - torch.logsumexp(Z + u.unsqueeze(1), dim=0)
        return Z + u.unsqueeze(1) + v.unsqueeze(0)


@torch.no_grad()
def log_optimal_transport(scores: torch.Tensor, num_sinkhorn_iter=50) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability, following ``SuperGlue: Learning Feature Matching with Graph Neural Networks`` """
    m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    # hk OT
    log_mu = - ms.log().expand(m)
    log_nu = - ns.log().expand(n)
    Q = log_sinkhorn_iterations(scores, log_mu, log_nu, num_sinkhorn_iter)

    return Q