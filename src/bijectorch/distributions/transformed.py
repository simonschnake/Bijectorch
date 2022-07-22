"""Distribution representing a Bijector applied to a Distribution."""

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.distributions import Distribution

from bijectorch.bijectors import bijector as base

# Note that there is already a implementation in `torch`. 
# Here we redefine it to work with the bijectors.

class TransformedDistribution(Distribution):
    """Distribution of a random variable transformed by a bijective function.
    Let `X` be a continuous random variable and `Y = f(X)` be a random variable
    transformed by a differentiable bijection `f` (a "bijector"). Given the
    distribution of `X` (the "base distribution") and the bijector `f`, this class
    implements the distribution of `Y` (also known as the pushforward of the base
    distribution through `f`).
    The probability density of `Y` can be computed by:
    `log p(y) = log p(x) - log|det J(f)(x)|`
    where `p(x)` is the probability density of `X` (the "base density") and
    `J(f)(x)` is the Jacobian matrix of `f`, both evaluated at `x = f^{-1}(y)`.
    Sampling from a Transformed distribution involves two steps: sampling from the
    base distribution `x ~ p(x)` and then evaluating `y = f(x)`. The first step
    is agnostic to the possible batch dimensions of the bijector `f(x)`. For
    example:
    ``` TODO: update example
      dist = distrax.Normal(loc=0., scale=1.)
      bij = distrax.ScalarAffine(shift=jnp.asarray([3., 3., 3.]))
      transformed_dist = distrax.Transformed(distribution=dist, bijector=bij)
      samples = transformed_dist.sample(seed=0, sample_shape=())
      print(samples)  # [2.7941577, 2.7941577, 2.7941577]
    ```
    Note: the `batch_shape`, `event_shape`, and `dtype` properties of the
    transformed distribution.
    """

    def __init__(self, distribution: Distribution, bijector: base.Bijector):
        """Initializes a Transformed distribution.
        Args:
          distribution: the base distribution. a torch distribution.
          bijector: a differentiable bijective transformation.
        """
        super().__init__()

        self._distribution = distribution
        self._bijector = bijector

        if len(distribution.event_shape) != bijector.event_ndims_in:
            raise ValueError(
                f"Base distribution '{distribution.__class__.__name__}' has event shape "
                f"{distribution.event_shape}, but bijector '{bijector.name}' expects "
                f"events to have {bijector.event_ndims_in} dimensions. Perhaps use "
                f"`distrax.Block` or `distrax.Independent`?")

        self._batch_shape: Tuple[int, ...] = self._distribution.batch_shape
        self._event_shape: Tuple[int, ...] = self._distribution.batch_shape

    @property
    def distribution(self) -> Distribution:
        """The base distribution."""
        return self._distribution

    @property
    def bijector(self) -> base.Bijector:
        """The bijector representing the transformation."""
        return self._bijector

    @property
    def has_rsample(self):
        return self._distribution.has_rsample

    @torch.no_grad()
    def sample(self, sample_shape= torch.Size(), z: Optional[Tensor] = None) -> Tensor:
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched. Samples first from
        base distribution and applies `transform()` for every transform in the
        list.
        """

        # the base distribution is not conditional
        x: Tensor = self._distribution.sample(sample_shape) 
        y = self._bijector.forward(x, z)
        return y

    def rsample(self, sample_shape=torch.Size(), z: Optional[Tensor] = None):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched. Samples first from base distribution and applies
        `transform()` for every transform in the list.
        """
        x = self._distribution.rsample(sample_shape)
        y = self._bijector.forward(x, z)
        return y

    def log_prob(self, value: Tensor, z: Optional[Tensor] = None) -> Tensor:
        """See `Distribution.log_prob`."""
        x, ildj_y = self.bijector.inverse_and_log_det(value, z)
        lp_x = self.distribution.log_prob(x)
        lp_y = lp_x + ildj_y
        return lp_y

    def mean(self) -> Tensor:
        """Calculates the mean."""
        if self.bijector.is_constant_jacobian:
            return self.bijector.forward(self.distribution.mean())
        else:
            raise NotImplementedError(
                "`mean` is not implemented for this transformed distribution, "
                "because its bijector's Jacobian is not known to be constant.")

    def mode(self) -> Tensor:
        """Calculates the mode."""
        if self.bijector.is_constant_log_det:
            return self.bijector.forward(self.distribution.mode())
        else:
            raise NotImplementedError(
                "`mode` is not implemented for this transformed distribution, "
                "because its bijector's Jacobian determinant is not known to be "
                "constant.")
