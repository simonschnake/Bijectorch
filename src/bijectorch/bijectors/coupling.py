import abc
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from bijectorch.bijectors import bijector as base


class Coupling(base.Bijector, metaclass=abc.ABCMeta):

    def __init__(self, event_shape: torch.Size, mask: Tensor, condition_network: nn.Module) -> None:

        super().__init__(len(event_shape))
        self._scaling_factor = nn.Parameter(torch.zeros([d // 2 for d in event_shape]))
        self._condition_network = condition_network
        self.register_buffer('_mask', mask)


    def _mask_tensor(self, x: Tensor):
        x_trans = x[..., self._mask]
        x_const = x[..., ~self._mask]
        return x_trans, x_const

    @abc.abstractmethod
    def _trans_forward_and_log_det(self, x: Tensor, params: Tensor) -> Tuple[Tensor, Tensor]:
        """forward transformation"""

    @abc.abstractmethod
    def _trans_inverse_and_log_det(self, y: Tensor, params: Tensor) -> Tuple[Tensor, Tensor]:
        """inverse transformation"""

    def forward_and_log_det(self, x: Tensor, z: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Computes y = f(x) and log|det J(f)(x)|."""

        x_trans, x_const = self._mask_tensor(x)

        if z is None:
            params: Tensor = self._condition_network(x_const)
        else:
            params: Tensor = self._condition_network(x_const, params)

        x_trans, logdet = self._trans_forward_and_log_det(x_trans, params)

        output = torch.empty_like(x)
        output[..., self._mask] = x_trans
        output[..., ~self._mask] = x_const
        return output, logdet


    def inverse_and_log_det(self, y: Tensor, z: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Computes y = f(x) and log|det J(f)(x)|."""

        y_trans, y_const = self._mask_tensor(y)

        if z is None:
            params: Tensor = self._condition_network(y_const)
        else:
            params: Tensor = self._condition_network(y_const, params)

        y_trans, logdet = self._trans_inverse_and_log_det(y_trans, params)

        output = torch.empty_like(y)
        output[..., self._mask] = y_trans
        output[..., ~self._mask] = y_const
        return output, logdet
