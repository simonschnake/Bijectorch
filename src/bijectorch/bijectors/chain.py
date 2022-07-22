"""Chain Bijector for composing a sequence of Bijector transformations."""

from typing import List, Optional, Sequence, Tuple

from torch import Tensor

from bijectorch.bijectors import bijector as base


class Chain(base.Bijector):
    """Composition of a sequence of bijectors into a single bijector.
    Bijectors are composable: if `f` and `g` are bijectors, then `g o f` is also
    a bijector. Given a sequence of bijectors `[f1, ..., fN]`, this class
    implements the bijector defined by `fN o ... o f1`.
    NOTE: the bijectors are applied in reverse order from the order they appear in
    the sequence. For example, consider the following code where `f` and `g` are
    two bijectors:
    ```
    layers = []
    layers.append(f)
    layers.append(g)
    bijector = distorch.Chain(layers)
    y = bijector.forward(x)
    ```
    The above code will transform `x` by first applying `g`, then `f`, so that
    `y = f(g(x))`.
    """

    def __init__(self, bijectors: Sequence[base.Bijector]):
        """Initializes a Chain bijector.
        Args:
          bijectors: a sequence of bijectors to be composed into one. 
            The sequence must contain at least one bijector.
        """
        if not bijectors:
          raise ValueError("The sequence of bijectors cannot be empty.")
        self._bijectors = bijectors

        # Check that neighboring bijectors in the chain have compatible dimensions
        for i, (outer, inner) in enumerate(zip(self._bijectors[:-1],
                                               self._bijectors[1:])):
          if outer.event_ndims_in != inner.event_ndims_out:
            raise ValueError(
                f"The chain of bijector event shapes are incompatible. Bijector "
                f"{i} ({outer.name}) expects events with {outer.event_ndims_in} "
                f"dimensions, while Bijector {i+1} ({inner.name}) produces events "
                f"with {inner.event_ndims_out} dimensions.")

        is_constant_jacobian = all(b.is_constant_jacobian for b in self._bijectors)
        is_constant_log_det = all(b.is_constant_log_det for b in self._bijectors)
        super().__init__(
            event_ndims_in=self._bijectors[-1].event_ndims_in,
            event_ndims_out=self._bijectors[0].event_ndims_out,
            is_constant_jacobian=is_constant_jacobian,
            is_constant_log_det=is_constant_log_det)

    @property
    def bijectors(self) -> Sequence[base.Bijector]:
      """The list of bijectors in the chain."""
      return self._bijectors

    def forward(self, x: Tensor, z: Optional[Tensor] = None) -> Tensor:
      """Computes y = f(x, z)."""
      for bijector in reversed(self._bijectors):
        x = bijector.forward(x, z)
      return x

    def inverse(self, y: Tensor, z: Optional[Tensor] = None) -> Tensor:
      """Computes x = f^{-1}(y, z)."""
      for bijector in self._bijectors:
        y = bijector.inverse(y, z)
      return y
    
    def forward_and_log_det(self, x: Tensor, z: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
      """Computes y = f(x, z) and log|det J(f)(x, z)|."""
      x, log_det = self._bijectors[-1].forward_and_log_det(x, z)
      for bijector in reversed(self._bijectors[:-1]):
        x, ld = bijector.forward_and_log_det(x, z)
        log_det += ld
      return x, log_det
    
    def inverse_and_log_det(self, y: Tensor, z: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
      """Computes x = f^{-1}(y, z) and log|det J(f^{-1})(y, z)|."""
      y, log_det = self._bijectors[0].inverse_and_log_det(y, z)
      for bijector in self._bijectors[1:]:
        y, ld = bijector.inverse_and_log_det(y, z)
        log_det += ld
      return y, log_det
    
    def same_as(self, other: base.Bijector) -> bool:
      """Returns True if this bijector is guaranteed to be the same as `other`."""
      if type(other) is Chain:  # pylint: disable=unidiomatic-typecheck
        if len(self.bijectors) != len(other.bijectors):
          return False
        for bij1, bij2 in zip(self.bijectors, other.bijectors):
          if not bij1.same_as(bij2):
            return False
        return True
      elif len(self.bijectors) == 1:
        return self.bijectors[0].same_as(other)
    
      return False