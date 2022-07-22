"""Bijector abstract base class."""

import abc
from typing import Optional, Tuple

from torch import Tensor, nn


class Bijector(nn.Module, metaclass=abc.ABCMeta):
    """Differentiable bijection that knows to compute its Jacobian determinant.
    
    A bijector implements a differentiable and bijective transformation `f`, whose 
    inverse is also differentiable (`f` is called a "diffeomorphism"). A bijector
    can be used to transform a continuous random variable `X` to a continuous
    random variable `Y = f(X)` in the context of `TransformedDistribution`.
    The bijector has the possibilty to be conditional on a variabel z.
        
    - `forward_and_log_det(x)` (required)
    - `inverse_and_log_det(y)` (required)
    
    The remaining methods are defined in terms of the above by default.
    Subclass requirements:
    
    - Subclasses must ensure that `f` is differentiable and bijective, and that
      their methods correctly implement `f^{-1}`, `J(f)` and `J(f^{-1})`. Distorch
      will assume these properties hold, and will make no attempt to verify them.
    - Distorch assumes that `f` acts on tensor-valued variables called "events", and
      that the bijector operates on batched events. Specifically, Distorch assumes
      the following:
    
        * `f` acts on events of shape [M1, ..., Mn] and returns events of shape
          [L1, ..., Lq]. `n` is referred to as `event_ndims_in`, and `q` as
          `event_ndims_out`. `event_ndims_in` and `event_ndims_out` must be static
          properties of the bijector, and must be known to it at construction time.
        * The bijector acts on batched events of shape [N1, ..., Nk, M1, ..., Mn],
          where [N1, ..., Nk] are batch dimensions, and returns batched events of
          shape [K1, ..., Kp, L1, ..., Lq], where [K1, ..., Kp] are (possibly
          different) batch dimensions. Distorch requires that bijectors always
          broadcast against batched events, that is, that they apply `f` identically
          to each event. Distorch also allows for events to broadcast against batched
          bijectors, meaning that multiple instantiations of `f` are applied to the
          same event, although this is not a subclass requirement.
      """

    def __init__(
      self, event_ndims_in: int,
      event_ndims_out: Optional[int] = None,
      is_constant_jacobian: bool = False,
      is_constant_log_det: Optional[bool] = None):
      """Initializes a Bijector.
      Args:
        event_ndims_in: Number of input event dimensions. The bijector acts on
          events of shape [M1, ..., Mn], where `n == event_ndims_in`.
        event_ndims_out: Number of output event dimensions. The bijector returns
          events of shape [L1, ..., Lq], where `q == event_ndims_out`. If None, it
          defaults to `event_ndims_in`.
        is_constant_jacobian: Whether the Jacobian is promised to be constant
          (which is the case if and only if the bijector is affine). A value of
          False will be interpreted as "we don't know whether the Jacobian is
          constant", rather than "the Jacobian is definitely not constant". Only
          set to True if you're absolutely sure the Jacobian is constant; if
          you're not sure, set to False.
        is_constant_log_det: Whether the Jacobian determinant is promised to be
          constant (which is the case for, e.g., volume-preserving bijectors). If
          None, it defaults to `is_constant_jacobian`. Note that the Jacobian
          determinant can be constant without the Jacobian itself being constant.
          Only set to True if you're absoltely sure the Jacobian determinant is
          constant; if you're not sure, set to None.
      """
      if event_ndims_out is None:
        event_ndims_out = event_ndims_in
      if event_ndims_in < 0:
        raise ValueError(
            f"`event_ndims_in` can't be negative. Got {event_ndims_in}.")
      if event_ndims_out < 0:
        raise ValueError(
            f"`event_ndims_out` can't be negative. Got {event_ndims_out}.")
      if is_constant_log_det is None:
        is_constant_log_det = is_constant_jacobian
      if is_constant_jacobian and not is_constant_log_det:
        raise ValueError("The Jacobian is said to be constant, but its "
                         "determinant is said not to be, which is impossible.")
      self._event_ndims_in = event_ndims_in
      self._event_ndims_out = event_ndims_out
      self._is_constant_jacobian = is_constant_jacobian
      self._is_constant_log_det = is_constant_log_det

      
    def forward(self, x: Tensor, z: Optional[Tensor] = None) -> Tensor:
      """Computes y = f(x, z)."""
      y, _ = self.forward_and_log_det(x, z)
      return y
  
    def inverse(self, y: Tensor, z: Optional[Tensor] = None) -> Tensor:
        """Computes x = f^{-1}(y)."""
        x, _ = self.inverse_and_log_det(y, z)
        return x
  
    def forward_log_det_jacobian(self, x: Tensor, z: Optional[Tensor] = None) -> Tensor:
        """Computes log|det J(f)(x)|."""
        _, logdet = self.forward_and_log_det(x, z)
        return logdet
  
    def inverse_log_det_jacobian(self, y: Tensor, z: Optional[Tensor] = None) -> Tensor:
        """Computes log|det J(f^{-1})(y)|."""
        _, logdet = self.inverse_and_log_det(y, z)
        return logdet
  
    @abc.abstractmethod
    def forward_and_log_det(self, x: Tensor, z: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Computes y = f(x) and log|det J(f)(x)|."""
  
    @abc.abstractmethod
    def inverse_and_log_det(self, y: Tensor, z: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
  
    @property
    def event_ndims_in(self) -> int:
        """Number of input event dimensions."""
        return self._event_ndims_in
  
    @property
    def event_ndims_out(self) -> int:
        """Number of output event dimensions."""
        return self._event_ndims_out
  
    @property
    def is_constant_jacobian(self) -> bool:
        """Whether the Jacobian is promised to be constant."""
        return self._is_constant_jacobian
  
    @property
    def is_constant_log_det(self) -> bool:
        """Whether the Jacobian determinant is promised to be constant."""
        return self._is_constant_log_det
  
    @property
    def name(self) -> str:
        """Name of the bijector."""
        return self.__class__.__name__
  
    def _check_forward_input_shape(self, x: Tensor) -> None:
        """Checks that the input `x` to a forward method has valid shape."""
        x_ndims = len(x.shape)
        if x_ndims < self.event_ndims_in:
            raise ValueError(
            f"Bijector {self.name} has `event_ndims_in=={self.event_ndims_in}`,"
            f" but the input has only {x_ndims} array dimensions.")
  
    def _check_inverse_input_shape(self, y: Tensor) -> None:
        """Checks that the input `y` to an inverse method has valid shape."""
        y_ndims = len(y.shape)
        if y_ndims < self.event_ndims_out:
            raise ValueError(
            f"Bijector {self.name} has `event_ndims_out=={self.event_ndims_out}`,"
            f" but the input has only {y_ndims} array dimensions.")
  
    @abc.abstractmethod
    def same_as(self, other: "Bijector") -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""