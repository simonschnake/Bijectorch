"""Wrapper for inverting a Distorch Bijector."""

from typing import Optional, Tuple
from torch import Tensor
from bijectorch.bijectors import bijector as base

class Inverse(base.Bijector):
    """A bijector that inverts a given bijector.

    That is, if `bijector` implements the transformation `f`, `Inverse(bijector)`
    implements the inverse transformation `f^{-1}`.

    The inversion is performed by swapping the forward with the corresponding
    inverse methods of the given bijector.
    """

    def __init__(self, bijector: base.Bijector):
        """Initializes an Inverse bijector.
        Args:
          bijector: the bijector to be inverted. It can be a distrax bijector, a TFP
            bijector, or a callable to be wrapped by `Lambda`.
        """
        self._bijector = bijector

        super().__init__(
            event_ndims_in=self._bijector.event_ndims_out,
            event_ndims_out=self._bijector.event_ndims_in,
            is_constant_jacobian=self._bijector.is_constant_jacobian,
            is_constant_log_det=self._bijector.is_constant_log_det)

    @property
    def bijector(self) -> base.Bijector:
      """The base bijector that was the input to `Inverse`."""
      return self._bijector

    def forward(self, x: Tensor, z: Optional[Tensor] = None) -> Tensor:
      """Computes y = f(x)."""
      return self._bijector.inverse(x)

    def inverse(self, y: Tensor, z: Optional[Tensor] = None) -> Tensor:
      """Computes x = f^{-1}(y)."""
      return self._bijector.forward(y)

    def forward_log_det_jacobian(self, x: Tensor, z: Optional[Tensor] = None) -> Tensor:
      """Computes log|det J(f)(x)|."""
      return self._bijector.inverse_log_det_jacobian(x)

    def inverse_log_det_jacobian(self, y: Tensor, z: Optional[Tensor] = None) -> Tensor:
      """Computes log|det J(f^{-1})(y)|."""
      return self._bijector.forward_log_det_jacobian(y)

    def forward_and_log_det(self, x: Tensor, z: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
      """Computes y = f(x) and log|det J(f)(x)|."""
      return self._bijector.inverse_and_log_det(x)

    def inverse_and_log_det(self, y: Tensor, z: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
      """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
      return self._bijector.forward_and_log_det(y)

    @property
    def name(self) -> str:
        """Name of the bijector."""
        return self.__class__.__name__ + self._bijector.name

    def same_as(self, other: base.Bijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        if type(other) is Inverse:  # pylint: disable=unidiomatic-typecheck
            return self.bijector.same_as(other.bijector)
        return False    