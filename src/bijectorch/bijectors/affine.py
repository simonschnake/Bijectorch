from typing import Tuple

import torch
from torch import Tensor

from bijectorch.bijectors.coupling import Coupling


class AffineCoupling(Coupling):

    def _trans_forward_and_log_det(self, x: Tensor, params: Tensor) -> Tuple[Tensor, Tensor]:
        s, t = params.chunk(2, dim=-1)

        s_fac = self._scaling_factor.exp()
        s = torch.tanh(s / s_fac) * s_fac
        
        x = (x + t) * torch.exp(s)

        logdet = s.sum(dim=list(range(1, x.dim())))
        
        return x, logdet

    def _trans_inverse_and_log_det(self, y: Tensor, params: Tensor) -> Tuple[Tensor, Tensor]:
        s, t = params.chunk(2, dim=-1)

        s_fac = self._scaling_factor.exp()
        s = torch.tanh(s / s_fac) * s_fac
        
        y = y / torch.exp(s) - t

        logdet = -s.sum(dim=list(range(1, y.dim())))
        
        return y, logdet