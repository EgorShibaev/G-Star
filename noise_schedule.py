import abc

import torch
import torch.nn as nn

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def get_noise(config, dtype=torch.float32):
  if config.noise.type == 'geometric':
    return GeometricNoise(config.noise.sigma_min,
                          config.noise.sigma_max)
  elif config.noise.type == 'loglinear':
    return LogLinearNoise()
  elif config.noise.type == 'cosine':
    return CosineNoise()
  elif config.noise.type == 'cosinesqr':
    return CosineSqrNoise()
  elif config.noise.type == 'linear':
    return Linear(config.noise.sigma_min,
                  config.noise.sigma_max,
                  dtype)
  elif config.noise.type == 'loop':
    return LoopNoise(
      t_off=getattr(config.noise, 't_off', 0.05),
      t_on=getattr(config.noise, 't_on', 0.55),
      alpha_const=getattr(config.noise, 'alpha_const', 0.1),
      eps=getattr(config.noise, 'eps', 1e-6),
    )
  else:
    raise ValueError(f'{config.noise.type} is not a valid noise')


def binary_discretization(z):
  z_hard = torch.sign(z)
  z_soft = z / torch.norm(z, dim=-1, keepdim=True)
  return z_soft + (z_hard - z_soft).detach()


class Noise(abc.ABC, nn.Module):
  """
  Baseline forward method to get the total + rate of noise at a timestep
  """
  def forward(self, t):
    # Assume time goes from 0 to 1
    return self.total_noise(t), self.rate_noise(t)
  
  @abc.abstractmethod
  def rate_noise(self, t):
    """
    Rate of change of noise ie g(t)
    """
    pass

  @abc.abstractmethod
  def total_noise(self, t):
    """
    Total noise ie \int_0^t g(t) dt + g(0)
    """
    pass


class CosineNoise(Noise):
  def __init__(self, eps=1e-3):
    super().__init__()
    self.eps = eps

  def rate_noise(self, t):
    cos = (1 - self.eps) * torch.cos(t * torch.pi / 2)
    sin = (1 - self.eps) * torch.sin(t * torch.pi / 2)
    scale = torch.pi / 2
    return scale * sin / (cos + self.eps)

  def total_noise(self, t):
    cos = torch.cos(t * torch.pi / 2)
    return - torch.log(self.eps + (1 - self.eps) * cos)


class CosineSqrNoise(Noise):
  def __init__(self, eps=1e-3):
    super().__init__()
    self.eps = eps

  def rate_noise(self, t):
    cos = (1 - self.eps) * (
      torch.cos(t * torch.pi / 2) ** 2)
    sin = (1 - self.eps) * torch.sin(t * torch.pi)
    scale = torch.pi / 2
    return scale * sin / (cos + self.eps)

  def total_noise(self, t):
    cos = torch.cos(t * torch.pi / 2) ** 2
    return - torch.log(self.eps + (1 - self.eps) * cos)


class Linear(Noise):
  def __init__(self, sigma_min=0, sigma_max=10, dtype=torch.float32):
    super().__init__()
    self.sigma_min = torch.tensor(sigma_min, dtype=dtype)
    self.sigma_max = torch.tensor(sigma_max, dtype=dtype)

  def rate_noise(self, t):
    return self.sigma_max - self.sigma_min

  def total_noise(self, t):
    return self.sigma_min + t * (self.sigma_max - self.sigma_min)

  def importance_sampling_transformation(self, t):
    f_T = torch.log1p(- torch.exp(- self.sigma_max))
    f_0 = torch.log1p(- torch.exp(- self.sigma_min))
    sigma_t = - torch.log1p(- torch.exp(t * f_T + (1 - t) * f_0))
    return (sigma_t - self.sigma_min) / (
      self.sigma_max - self.sigma_min)


class GeometricNoise(Noise):
  def __init__(self, sigma_min=1e-3, sigma_max=1):
    super().__init__()
    self.sigmas = 1.0 * torch.tensor([sigma_min, sigma_max])

  def rate_noise(self, t):
    return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t * (
      self.sigmas[1].log() - self.sigmas[0].log())

  def total_noise(self, t):
    return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t


class LogLinearNoise(Noise):
  """Log Linear noise schedule.
  
  Built such that 1 - 1/e^(n(t)) interpolates between 0 and
  ~1 when t varies from 0 to 1. Total noise is
  -log(1 - (1 - eps) * t), so the sigma will be
  (1 - eps) * t.
  """
  def __init__(self, eps=1e-3):
    super().__init__()
    self.eps = eps
    self.sigma_max = self.total_noise(torch.tensor(1.0))
    self.sigma_min = self.eps + self.total_noise(torch.tensor(0.0))

  def rate_noise(self, t):
    return (1 - self.eps) / (1 - (1 - self.eps) * t)

  def total_noise(self, t):
    return -torch.log1p(-(1 - self.eps) * t)

  def importance_sampling_transformation(self, t):
    f_T = torch.log1p(- torch.exp(- self.sigma_max))
    f_0 = torch.log1p(- torch.exp(- self.sigma_min))
    sigma_t = - torch.log1p(- torch.exp(t * f_T + (1 - t) * f_0))
    t = - torch.expm1(- sigma_t) / (1 - self.eps)
    return t


class LoopNoise(Noise):
  """Piecewise-constant/linear alpha schedule mapped to sigma.

  Desired alpha(t) = 1 - exp(-sigma(t)) has the following shape:
    - On [0, t_off]: linearly increases 0 -> alpha_const
    - On [t_off, t_on]: stays at alpha_const
    - On [t_on, 1]: linearly increases alpha_const -> 1

  We implement this by defining alpha(t) directly, then mapping back to
  sigma(t) via sigma = -log(1 - alpha). The rate g(t) is d sigma / dt.
  """
  def __init__(self, t_off=0.05, t_on=0.55, alpha_const=0.1, eps=1e-6):
    super().__init__()
    # Validate and store as tensors for correct device/dtype propagation
    if not (0.0 <= t_off <= 1.0 and 0.0 <= t_on <= 1.0):
      raise ValueError('t_off and t_on must be in [0, 1]')
    if t_off > t_on:
      raise ValueError('Expected t_off <= t_on')
    if not (0.0 <= alpha_const < 1.0):
      raise ValueError('alpha_const must be in [0, 1)')

    self.t_off = float(t_off)
    self.t_on = float(t_on)
    self.alpha_const = float(alpha_const)
    self.eps = float(eps)

  def _alpha(self, t):
    # Ensure computations maintain broadcasting and device
    t = t.to(dtype=torch.float32)
    t_off = torch.as_tensor(self.t_off, dtype=t.dtype, device=t.device)
    t_on = torch.as_tensor(self.t_on, dtype=t.dtype, device=t.device)
    alpha_const = torch.as_tensor(self.alpha_const, dtype=t.dtype, device=t.device)

    # Piecewise alpha
    # segment 1: [0, t_off] linear 0 -> alpha_const
    slope1 = torch.where(t_off > 0, alpha_const / torch.clamp_min(t_off, 1e-12), torch.zeros_like(alpha_const))
    alpha1 = slope1 * torch.clamp(t, 0.0, None)

    # segment 2: [t_off, t_on] constant alpha_const
    alpha2 = alpha_const.expand_as(alpha1)

    # segment 3: [t_on, 1] linear alpha_const -> 1
    denom3 = torch.clamp_min(1.0 - t_on, 1e-12)
    slope3 = (1.0 - alpha_const) / denom3
    alpha3 = alpha_const + slope3 * (t - t_on)

    # Select based on t
    alpha = torch.where(t < t_off, alpha1, torch.where(t < t_on, alpha2, alpha3))
    # Clamp to [0, 1 - eps]
    alpha = torch.clamp(alpha, 0.0, 1.0 - self.eps)
    return alpha

  def _dalpha_dt(self, t):
    t = t.to(dtype=torch.float32)
    t_off = torch.as_tensor(self.t_off, dtype=t.dtype, device=t.device)
    t_on = torch.as_tensor(self.t_on, dtype=t.dtype, device=t.device)
    alpha_const = torch.as_tensor(self.alpha_const, dtype=t.dtype, device=t.device)

    slope1 = torch.where(t_off > 0, alpha_const / torch.clamp_min(t_off, 1e-12), torch.zeros_like(alpha_const))
    slope3 = (1.0 - alpha_const) / torch.clamp_min(1.0 - t_on, 1e-12)

    d_alpha = torch.where(t < t_off, slope1, torch.where(t < t_on, torch.zeros_like(slope1), slope3))
    return d_alpha

  def total_noise(self, t):
    alpha = self._alpha(t)
    # sigma = -log(1 - alpha)
    sigma = -torch.log1p(-alpha)
    return sigma

  def rate_noise(self, t):
    # g(t) = d sigma / dt = alpha'(t) / (1 - alpha(t))
    alpha = self._alpha(t)
    dalpha = self._dalpha_dt(t)
    denom = torch.clamp_min(1.0 - alpha, 1e-12)
    return dalpha / denom
