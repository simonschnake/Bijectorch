"""Rational-quadratic spline bijector."""

from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from bijectorch.bijectors.coupling import Coupling


def _normalize_bin_sizes(
    unnormalized_bin_sizes: Tensor, total_size: float,
    min_bin_size: float) -> Tensor:
    """Make bin sizes sum to `total_size` and be no less than `min_bin_size`."""
    num_bins = unnormalized_bin_sizes.shape[-1]
    if num_bins * min_bin_size > total_size:
        raise ValueError(
            f'The number of bins ({num_bins}) times the minimum bin size'
            f' ({min_bin_size}) cannot be greater than the total bin size'
            f' ({total_size}).')
    bin_sizes = F.softmax(unnormalized_bin_sizes, dim=-1)
    return bin_sizes * (total_size - num_bins * min_bin_size) + min_bin_size

def _normalize_knot_slopes(
    unnormalized_knot_slopes: Tensor,
    min_knot_slope: float) -> Tensor:
    """Make knot slopes be no less than `min_knot_slope`."""
    # The offset is such that the normalized knot slope will be equal to 1
    # whenever the unnormalized knot slope is equal to 0.
    if min_knot_slope >= 1.:
        raise ValueError(f'The minimum knot slope must be less than 1; got' f' {min_knot_slope}.')

    min_knot_slope = torch.as_tensor(min_knot_slope)
    offset = torch.log(torch.exp(1. - min_knot_slope) - 1.)
    return F.softplus(unnormalized_knot_slopes + offset) + min_knot_slope

def _safe_quadratic_root(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """Implement a numerically stable version of the quadratic formula."""
    # This is not a general solution to the quadratic equation, as it assumes
    # b ** 2 - 4. * a * c is known a priori to be positive (and which of the two
    # roots is to be used, see https://arxiv.org/abs/1906.04032).
    # There are two sources of instability:
    # (a) When b ** 2 - 4. * a * c -> 0, sqrt gives NaNs in gradient.
    # We clip sqrt_diff to have the smallest float number.
    sqrt_diff = b ** 2 - 4. * a * c
    safe_sqrt = torch.sqrt(torch.clamp(sqrt_diff, torch.finfo(sqrt_diff.dtype).tiny))
    # If sqrt_diff is non-positive, we set sqrt to 0. as it should be positive.
    safe_sqrt = torch.where(sqrt_diff > 0., safe_sqrt, 0.)
    # (b) When 4. * a * c -> 0. We use the more stable quadratic solution
    # depending on the sign of b.
    # See https://people.csail.mit.edu/bkph/articles/Quadratics.pdf (eq 7 and 8).
    # Solution when b >= 0
    numerator_1 = 2. * c
    denominator_1 = -b - safe_sqrt
    # Solution when b < 0
    numerator_2 = - b + safe_sqrt
    denominator_2 = 2 * a
    # Choose the numerically stable solution.
    numerator = torch.where(b >= 0, numerator_1, numerator_2)
    denominator = torch.where(b >= 0, denominator_1, denominator_2)
    return numerator / denominator


class RationalQuadraticSpline(Coupling):
    """A rational-quadratic spline bijector.
    Implements the spline bijector introduced by:
    > Durkan et al., Neural Spline Flows, https://arxiv.org/abs/1906.04032, 2019.
    This bijector is a monotonically increasing spline operating on an interval
    [a, b], such that f(a) = a and f(b) = b. Outside the interval [a, b], the
    bijector defaults to a linear transformation whose slope matches that of the
    spline at the nearest boundary (either a or b). The range boundaries a and b
    are hyperparameters passed to the constructor.
    The spline on the interval [a, b] consists of `num_bins` segments, on each of
    which the spline takes the form of a rational quadratic (ratio of two
    quadratic polynomials). The first derivative of the bijector is guaranteed to
    be continuous on the whole real line. The second derivative is generally not
    continuous at the knot points (bin boundaries).
    The spline is parameterized by the bin sizes on the x and y axis, and by the
    slopes at the knot points. All spline parameters are passed to the constructor
    as an unconstrained array `params` of shape `[..., 3 * num_bins + 1]`. The
    spline parameters are extracted from `params`, and are reparameterized
    internally as appropriate. The number of bins is a hyperparameter, and is
    implicitly defined by the last dimension of `params`.
    This bijector is applied elementwise. Given some input `x`, the parameters
    `params` and the input `x` are broadcast against each other. For example,
    suppose `x` is of shape `[N, D]`. Then:
    - If `params` is of shape `[3 * num_bins + 1]`, the same spline is identically
      applied to each element of `x`.
    - If `params` is of shape `[D, 3 * num_bins + 1]`, the same spline is applied
      along the first axis of `x` but a different spline is applied along the
      second axis of `x`.
    - If `params` is of shape `[N, D, 3 * num_bins + 1]`, a different spline is
      applied to each element of `x`.
    - If `params` is of shape `[M, N, D, 3 * num_bins + 1]`, `M` different splines
      are applied to each element of `x`, and the output is of shape `[M, N, D]`.
    """

    def __init__(
        self, event_shape: torch.Size, mask: Tensor, condition_network: nn.Module,
        range_min: float, range_max: float, boundary_slopes: str = 'unconstrained',
        min_bin_size: float = 1e-4, min_knot_slope: float = 1e-4):
        """Initializes a RationalQuadraticSpline bijector.
        Args:
          params: array of shape `[..., 3 * num_bins + 1]`, the unconstrained
            parameters of the bijector. The number of bins is implicitly defined by
            the last dimension of `params`. The parameters can take arbitrary
            unconstrained values; the bijector will reparameterize them internally
            and make sure they obey appropriate constraints. If `params` is the
            all-zeros array, the bijector becomes the identity function everywhere
            on the real line.
          range_min: the lower bound of the spline's range. Below `range_min`, the
            bijector defaults to a linear transformation.
          range_max: the upper bound of the spline's range. Above `range_max`, the
            bijector defaults to a linear transformation.
          boundary_slopes: controls the behaviour of the slope of the spline at the
            range boundaries (`range_min` and `range_max`). It is used to enforce
            certain boundary conditions on the spline. Available options are:
            - 'unconstrained': no boundary conditions are imposed; the slopes at the
              boundaries can vary freely.
            - 'lower_identity': the slope of the spline is set equal to 1 at the
              lower boundary (`range_min`). This makes the bijector equal to the
              identity function for values less than `range_min`.
            - 'upper_identity': similar to `lower_identity`, but now the slope of
              the spline is set equal to 1 at the upper boundary (`range_max`). This
              makes the bijector equal to the identity function for values greater
              than `range_max`.
            - 'identity': combines the effects of 'lower_identity' and
              'upper_identity' together. The slope of the spline is set equal to 1
              at both boundaries (`range_min` and `range_max`). This makes the
              bijector equal to the identity function outside the interval
              `[range_min, range_max]`.
            - 'circular': makes the slope at `range_min` and `range_max` be the
              same. This implements the "circular spline" introduced by:
              > Rezende et al., Normalizing Flows on Tori and Spheres,
              > https://arxiv.org/abs/2002.02428, 2020.
              This option should be used when the spline operates on a circle
              parameterized by an angle in the interval `[range_min, range_max]`,
              where `range_min` and `range_max` correspond to the same point on the
              circle.
          min_bin_size: The minimum bin size, in either the x or the y axis. Should
            be a small positive number, chosen for numerical stability. Guarantees
            that no bin in either the x or the y axis will be less than this value.
          min_knot_slope: The minimum slope at each knot point. Should be a small
            positive number, chosen for numerical stability. Guarantess that no knot
            will have a slope less than this value.
        """
        super().__init__(event_shape, mask, condition_network)

        self.range_min = range_min
        self.range_max = range_max

        self.total_size = range_max - range_min

        self.boundary_slopes = boundary_slopes
        self.min_bin_size = min_bin_size
        self.min_knot_slope = min_knot_slope

        if range_min >= range_max:
            raise ValueError(
                f'`range_min` must be less than `range_max`. Got'
                f' `range_min={range_min}` and `range_max={range_max}`.')
        
        if min_bin_size <= 0.:
            raise ValueError(
                f'The minimum bin size must be positive; got'
                f' {min_bin_size}.')

        if min_knot_slope <= 0.:
            raise ValueError(
                f'The minimum knot slope must be positive; got'
                f' {min_knot_slope}.')

    def _calculate_bins(self, params: Tensor) -> Tensor:
        """
        The input params consists of the unnormalized bin_widths, bin_heights
        """
        if params.shape[-1] % 3 != 1 or params.shape[-1] < 4:
          raise ValueError(f'The last dimension of `params` must have size'
                           f' `3 * num_bins + 1` and `num_bins` must be at least 1.'
                           f' Got size {params.shape[-1]}.')

        K = (params.shape[-1] - 1) // 3
        # Extract unnormalized parameters.
        unnormalized_bin_widths = params[..., :K]
        unnormalized_bin_heights = params[..., K : 2 * K]
        unnormalized_knot_slopes = params[..., 2 * K:]

        # Normalize bin sizes and compute bin positions on the x and y axis.
        bin_widths = _normalize_bin_sizes(unnormalized_bin_widths, self.total_size, self.min_bin_size)
        bin_heights = _normalize_bin_sizes(unnormalized_bin_heights, self.total_size, self.min_bin_size)

        x_pos = self.range_min + torch.cumsum(bin_widths[..., :-1], axis=-1)
        y_pos = self.range_min + torch.cumsum(bin_heights[..., :-1], axis=-1)

        pad_shape = x_pos.shape[:-1] + (1,)
        pad_below = torch.full(pad_shape, self.range_min, dtype=x_pos.dtype)
        pad_above = torch.full(pad_shape, self.range_max, dtype=x_pos.dtype)

        x_pos = torch.cat([pad_below, x_pos, pad_above], axis=-1)
        y_pos = torch.cat([pad_below, y_pos, pad_above], axis=-1)

        # Normalize knot slopes and enforce requested boundary conditions.
        knot_slopes = _normalize_knot_slopes(unnormalized_knot_slopes, self.min_knot_slope)
 
        if self.boundary_slopes == 'unconstrained':
          self._knot_slopes = knot_slopes
        elif self.boundary_slopes == 'lower_identity':
          ones = torch.ones(pad_shape, self._dtype)
          self._knot_slopes = torch.concatenate([ones, knot_slopes[..., 1:]], axis=-1)
        elif self.boundary_slopes == 'upper_identity':
          ones = torch.ones(pad_shape, self._dtype)
          self._knot_slopes = torch.concatenate(
              [knot_slopes[..., :-1], ones], axis=-1)
        elif self.boundary_slopes == 'identity':
          ones = torch.ones(pad_shape, self._dtype)
          self._knot_slopes = torch.concatenate(
              [ones, knot_slopes[..., 1:-1], ones], axis=-1)
        elif self.boundary_slopes == 'circular':
          self._knot_slopes = torch.concatenate(
              [knot_slopes[..., :-1], knot_slopes[..., :1]], axis=-1)
        else:
          raise ValueError(f'Unknown option for boundary slopes:'
                           f' `{self.boundary_slopes}`.')

        params = torch.stack([x_pos, y_pos, knot_slopes], axis=1)

        return params
    
    def _trans_forward_and_log_det(self, x: Tensor, params: Tensor) -> Tuple[Tensor, Tensor]:
        """Applies a rational-quadratic spline to a scalar.
            Args:
                x: a scalar (0-dimensional array). The scalar `x` can be any real number; it
                   will be transformed by the spline if it's in the closed interval
                   `[x_pos[0], x_pos[-1]]`, and it will be transformed linearly if it's
                   outside that interval.
                params: 

                array of shape [num_bins + 1], the bin boundaries on the x axis.
                   y_pos: array of shape [num_bins + 1], the bin boundaries on the y axis.
          knot_slopes: array of shape [num_bins + 1], the slopes at the knot points.
        Returns:
              A tuple of two scalars: the output of the transformation and the log of the
          absolute first derivative at `x`.
        """
        # Search to find the right bin. NOTE: The bins are sorted, so we could use
        # binary search, but this is more GPU/TPU friendly.
        # The following implementation avoids indexing for faster TPU computation.

        params = self._calculate_bins(params)

        below_range: Tensor = x <= params[..., 0, :1]
        above_range: Tensor = x >= params[..., 0, -1:]

        k = torch.searchsorted(params[..., 0, :].contiguous(), x) - 1

        # If x does not fall into any bin, we use the first spline in the following
        # computations to avoid numerical issues.
        k = torch.where(k < 0, 0, k)
        k = torch.where(k >= params.size(-1) - 1, k - 1, k)

        x_k = params[..., 0, :].gather(-1, k)
        w = params[..., 0, :].gather(-1, k + 1) - x_k

        y_k = params[..., 1, :].gather(-1, k)
        h = params[..., 1, :].gather(-1, k + 1) - y_k

        d_k = params[..., 2, :].gather(-1, k)
        d_kp1 = params[..., 2, :].gather(-1, k + 1)

        s = h / w
        xi = (x - x_k) / w
        # `xi` should be in range [0, 1] to avoid NaNs later. This can happen because
        # of small floating point issues or when x is outside of the range of bins.
        # To avoid all problems, we restrict z in [0, 1].
        xi = torch.clamp(xi, 0., 1.)
        sq_xi = xi * xi
        xi1mxi = xi - sq_xi # ξ(1-ξ)

        numerator = h * (s * sq_xi + d_k * xi1mxi)
        denominator = s + (d_kp1 + d_k - 2. * s) * xi1mxi

        y = y_k + numerator/denominator

        
        # Compute log det Jacobian.
        # The logdet is a sum of 3 logs. It is easy to see that the inputs of the
        # first two logs are guaranteed to be positive because we ensured that z is in
        # [0, 1]. This is also true of the log(denominator) because:
        # denominator
        # == s + (d_kp1 + d_k - 2 * s) * xi*(1-xi)
        # >= s - 2 * s * xi * (1-xi)
        # >= s - 2 * s * (1/4)
        # == s / 2
        logdet = (
            2. * torch.log(s) 
            + torch.log(d_kp1 * sq_xi + 2. * s * xi1mxi + d_k * (1 - xi) * (1 - xi))
            - 2. * torch.log(denominator))

        # If x is outside the spline range, we default to a linear transformation.
        y = torch.where(below_range, (x - params[..., 0, :1]) * params[..., 2, :1] + params[..., 1, :1], y)
        y = torch.where(above_range, (x - params[..., 0, -1:]) * params[..., 2, -1:] + params[..., 1, -1:], y)

        logdet = torch.where(below_range, torch.log(params[..., 2, :1]), logdet)
        logdet = torch.where(above_range, torch.log(params[..., 2, -1:]), logdet)

        return y, logdet




    def _trans_inverse_and_log_det(self, y: Tensor, params: Tensor) -> Tuple[Tensor, Tensor]:
        """Applies the inverse of a rational-quadratic spline to a scalar.
        Args:
          y: a scalar (0-dimensional array). The scalar `y` can be any real number; it
            will be transformed by the spline if it's in the closed interval
            `[y_pos[0], y_pos[-1]]`, and it will be transformed linearly if it's
            outside that interval.
          x_pos: array of shape [num_bins + 1], the bin boundaries on the x axis.
          y_pos: array of shape [num_bins + 1], the bin boundaries on the y axis.
          knot_slopes: array of shape [num_bins + 1], the slopes at the knot points.
        Returns:
          A tuple of two scalars: the output of the inverse transformation and the log
          of the absolute first derivative of the inverse at `y`.
        """
        # Search to find the right bin. NOTE: The bins are sorted, so we could use
        # binary search, but this is more GPU/TPU friendly.
        # The following implementation avoids indexing for faster TPU computation.

        params = self._calculate_bins(params)

        below_range: Tensor = y <= params[..., 1, :1]
        above_range: Tensor = y >= params[..., 1, -1:]

        k = torch.searchsorted(params[..., 1, :].contiguous(), y) - 1

        # If x does not fall into any bin, we use the first spline in the following
        # computations to avoid numerical issues.
        k = torch.where(k < 0, 0, k)
        k = torch.where(k >= params.size(-1) - 1, k - 1, k)

        x_k = params[..., 0, :].gather(-1, k)
        w = params[..., 0, :].gather(-1, k + 1) - x_k

        y_k = params[..., 1, :].gather(-1, k)
        h = params[..., 1, :].gather(-1, k + 1) - y_k

        d_k = params[..., 2, :].gather(-1, k)
        d_kp1 = params[..., 2, :].gather(-1, k + 1)

        s = h / w

        omega = (y - y_k) / h
        omega = torch.clamp(omega, 0., 1.)  # Ensure w is in [0, 1].
        # Compute quadratic coefficients: az^2 + bz + c = 0
        slopes_term = d_kp1 + d_k - 2. * s
        c = - s * omega
        b = d_k - slopes_term * omega
        a = s - b
    
        # Solve quadratic to obtain z and then x.
        xi = _safe_quadratic_root(a, b, c)
        xi = torch.clamp(xi, 0., 1.)  # Ensure z is in [0, 1].
        x = w * xi + x_k
    
        # Compute log det Jacobian.
        sq_xi = xi * xi
        xi1mxi = xi - sq_xi  # ξ(1-ξ)

        denominator = s + slopes_term * xi1mxi
        logdet = - 2. * torch.log(s) - torch.log(
            d_kp1 * sq_xi + 2. * s * xi1mxi +
            d_k * (1. - xi) * (1. - xi)) + 2. * torch.log(denominator)
    
        # If y is outside the spline range, we default to a linear transformation.
        x = torch.where(below_range, (y - params[..., 1, :1]) / params[..., 2, :1] + params[..., 0, :1], x)
        x = torch.where(above_range, (y - params[..., 1, -1:]) / params[..., 2, -1:] + params[..., 0, -1:], x)

        logdet = torch.where(below_range, - torch.log(params[..., 2, :1]), logdet)
        logdet = torch.where(above_range, - torch.log(params[..., 2, -1:]), logdet)
        return x, logdet