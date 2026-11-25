import math
import torch


@torch.jit.script
def cosine_cutoff(x: torch.Tensor, r_max: torch.Tensor, r_start_cos_ratio: float = 0.8):
    """A piecewise cosine cutoff starting the cosine decay at r_decay_factor*r_max.

    Broadcasts over r_max.
    """
    r_max, x = torch.broadcast_tensors(r_max.unsqueeze(-1), x.unsqueeze(0))
    r_decay: torch.Tensor = r_start_cos_ratio * r_max
    # for x < r_decay, clamps to 1, for x > r_max, clamps to 0
    x = x.clamp(r_decay, r_max)
    return 0.5 * (torch.cos((math.pi / (r_max - r_decay)) * (x - r_decay)) + 1.0)


@torch.jit.script
def polynomial_cutoff(
    x: torch.Tensor, r_max: torch.Tensor, p: float = 6.0
) -> torch.Tensor:
    """Polynomial cutoff, as proposed in DimeNet: https://arxiv.org/abs/2003.03123


    Parameters
    ----------
    r_max : tensor
        Broadcasts over r_max.

    p : int
        Power used in envelope function
    """
    assert p >= 2.0
    p_int = int(p)
    assert float(p_int) == p
    r_max, x = torch.broadcast_tensors(r_max.unsqueeze(-1), x.unsqueeze(0))
    x = x / r_max
    # The positive-coefficient quotient avoids cancellation at the triple root x=1.
    out = float(p_int * (p_int + 1) // 2) * torch.ones_like(x)
    for k in range(p_int - 2, -1, -1):
        out = out * x + float((k + 1) * (k + 2) // 2)
    tail = 1.0 - x
    return tail * tail * tail * out * (x < 1.0)
