"""Single-step sampler updates for guided star-shaped masked diffusion.

Each function here implements one iteration of a sampler given the current
noisy sequence `x`, the time tensors, and a `model` (Diffusion) object whose
attributes (denoiser, remasker, noise schedule, config, tokenizer) are used as
the sampler context. Dispatch lives in `Diffusion._sample`.
"""

from .remasker import remasker_update
from .conf_star_shape import conf_star_shape_update
from .p2 import p2_update

__all__ = ["remasker_update", "conf_star_shape_update", "p2_update"]
