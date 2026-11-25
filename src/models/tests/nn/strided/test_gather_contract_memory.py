"""Numerical and scripting contracts for the opt-in batched contraction."""

import pytest
import torch

from allegro.nn._strided import GatherContracter

pytestmark = pytest.mark.filterwarnings("ignore:`torch.jit.*:DeprecationWarning")
SH = "0e+1o+2e+3o"
CASES = [
    ("1o", "0e", True, 1),
    ("0e+1o", "0e", True, 2),
    ("1o", "2e", False, 1),
    (SH, SH, False, None),
    (SH, "1o+1o+2o+3o", False, None),
]


@pytest.mark.parametrize("inputs,outputs,diagonal,paths", CASES)
@pytest.mark.parametrize("coupling", [False, True])
@pytest.mark.parametrize("edges", [0, 5])
def test_memory_contraction_parity(inputs, outputs, diagonal, paths, coupling, edges):
    modules = []
    states = []
    for efficient in (False, True):
        torch.manual_seed(73)
        modules.append(
            GatherContracter(
                inputs,
                inputs,
                outputs,
                mul=2,
                path_channel_coupling=coupling,
                memory_efficient=efficient,
            )
        )
        states.append(torch.get_rng_state())
    reference, efficient = modules
    assert torch.equal(*states)
    assert reference.w3j_is_ij_diagonal == diagonal
    if paths is not None:
        assert reference.num_paths == paths
    else:
        assert reference.num_paths > 1
    assert list(reference.state_dict()) == list(efficient.state_dict())
    for key, value in reference.state_dict().items():
        assert torch.equal(value, efficient.state_dict()[key])
    efficient.load_state_dict(reference.state_dict(), strict=True)
    # Transposes make real noncontiguous strided inputs; repeated indices test
    # accumulation into the same gathered row during backward.
    x1 = torch.randn(edges, reference.base_dim1, 2).transpose(1, 2)
    x2 = torch.randn(3, reference.base_dim2, 2).transpose(1, 2)
    # The unchanged uncoupled mm path requires contiguous outer products.
    if not coupling:
        x1, x2 = x1.contiguous(), x2.contiguous()
    indices = torch.tensor([2, 0, 2, 2, 0][:edges], dtype=torch.long)
    outputs_and_grads = []
    probe = torch.randn(edges, 2, reference.base_dim_out)
    for module in modules:
        left = x1.detach().requires_grad_()
        right = x2.detach().requires_grad_()
        output = module(left, right, indices)
        gradients = torch.autograd.grad(
            (output * probe).sum(), (left, right, module.weights)
        )
        outputs_and_grads.append((output, *gradients))
    tolerance = 1e-12 if x1.dtype == torch.float64 else 1e-5
    for expected, actual in zip(*outputs_and_grads):
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    scripted = torch.jit.script(efficient)
    torch.testing.assert_close(
        scripted(x1, x2, indices),
        outputs_and_grads[1][0],
        atol=tolerance,
        rtol=tolerance,
    )
    assert "memory_efficient=True" in efficient.extra_repr()
