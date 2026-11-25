import torch


@torch.jit.script
def _poly_cutoff(x: torch.Tensor, factor: float, p: float = 6.0) -> torch.Tensor:
    x = x * factor
    assert p >= 2.0
    p_int = int(p)
    assert float(p_int) == p
    out = float(p_int * (p_int + 1) // 2) * torch.ones_like(x)
    for k in range(p_int - 2, -1, -1):
        out = out * x + float((k + 1) * (k + 2) // 2)
    tail = 1.0 - x
    return tail * tail * tail * out * (x < 1.0)


class PolynomialCutoff(torch.nn.Module):
    _factor: float
    p: float

    def __init__(self, r_max: float, p: float = 6):
        r"""Polynomial cutoff, as proposed in DimeNet: https://arxiv.org/abs/2003.03123


        Parameters
        ----------
        r_max : float
            Cutoff radius

        p : int
            Power used in envelope function
        """
        super().__init__()
        assert p >= 2.0
        assert float(int(p)) == p
        self.p = float(p)
        self._factor = 1.0 / float(r_max)

    def forward(self, x):
        """
        Evaluate cutoff function.

        x: torch.Tensor, input distance
        """
        return _poly_cutoff(x, self._factor, p=self.p)
