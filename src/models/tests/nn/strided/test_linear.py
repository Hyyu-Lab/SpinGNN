import pytest

import torch

import e3nn
from e3nn import o3

from allegro.nn._strided import Linear
from allegro.nn._strided._linear import (
    _instruction_groups_for,
    weights_from_per_instruction,
    weights_per_instruction,
)
from allegro.nn._strided._layout import StridedLayout

from test_contract import _strided_to_cat


@pytest.mark.parametrize(
    "irreps_in", ["0e", "0e + 0o + 1e + 1o", "0e + 0e + 0e + 1o + 1o + 2e"]
)
@pytest.mark.parametrize("irreps_out", ["0e", "0e + 0o + 1e + 1o", "0e + 1o + 2e"])
@pytest.mark.parametrize("mul", [1, 2, 7])
@pytest.mark.parametrize("mulout", [1, 2, 7])
@pytest.mark.parametrize("pad", [1, 4])
@pytest.mark.parametrize("shared_weights", [False, True])
def test_like_linear(
    irreps_in,
    irreps_out,
    mul,
    mulout,
    pad,
    shared_weights,
):
    irreps_in = o3.Irreps(irreps_in)
    irreps_out = o3.Irreps(irreps_out)

    ours = Linear(
        irreps_in=o3.Irreps((mul, ir) for _, ir in irreps_in),
        irreps_out=o3.Irreps((mulout, ir) for _, ir in irreps_out),
        shared_weights=shared_weights,
        pad_to_alignment=pad,
    )
    batchdim = 7
    args_in = (
        torch.randn(batchdim, mul, ours.dim_in),
        torch.ones(
            ((batchdim,) if not shared_weights else tuple()) + (ours.weight_numel,)
        ),
    )

    e3nns = o3.Linear(
        irreps_in=o3.Irreps((mul, ir) for _, ir in irreps_in),
        irreps_out=o3.Irreps((mulout, ir) for _, ir in irreps_out),
        shared_weights=shared_weights,
        internal_weights=False,
    )
    assert e3nns.weight_numel == ours.weight_numel

    args_e3nn = (
        _strided_to_cat(irreps_in, mul, args_in[0]),
        args_in[1],
    )
    ours_out_strided = ours(*args_in)
    assert ours_out_strided.shape[-1] % pad == 0
    ours_out = _strided_to_cat(irreps_out, mulout, ours_out_strided)
    e3nn_out = e3nns(*args_e3nn)
    assert ours_out.shape == e3nn_out.shape
    assert torch.allclose(ours_out, e3nn_out, atol=1e-6)


@pytest.mark.parametrize("pad", [1, 2, 4])
@pytest.mark.parametrize(
    "shared_weights,internal_weights", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("jit_script_fx", [False, True])
@pytest.mark.parametrize(
    "irreps_in,irreps_out,instructions",
    [
        ("0e + 1e + 0e + 1e", "0e + 1e", [(2, 0), (0, 0), (3, 1), (1, 1)]),
        (
            "0e + 0e + 1o + 0e + 3o",
            "0e + 3o + 0e",
            [(3, 2), (1, 0), (4, 1), (0, 0), (3, 0), (0, 2), (1, 2)],
        ),
    ],
)
def test_noncontiguous_instruction_groups_preserve_linear_contract(
    pad,
    shared_weights,
    internal_weights,
    jit_script_fx,
    irreps_in,
    irreps_out,
    instructions,
):
    """Compare disjoint/reordered paths with independent e3nn semantics.

    The second layout combines multi-input runs, repeated output irreps, an
    l=3 channel, and actual input/output padding for alignments 2 and 4.
    """
    irreps_in = o3.Irreps(irreps_in)
    irreps_out = o3.Irreps(irreps_out)
    mul_in, mul_out = 2, 3
    full_in = o3.Irreps((mul_in, ir) for _, ir in irreps_in)
    full_out = o3.Irreps((mul_out, ir) for _, ir in irreps_out)
    old_defaults = e3nn.get_optimization_defaults()
    e3nn.set_optimization_defaults(jit_script_fx=jit_script_fx)
    try:
        ours = Linear(
            irreps_in=full_in,
            irreps_out=full_out,
            instructions=instructions,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
            pad_to_alignment=pad,
        ).double()
        reference = o3.Linear(
            full_in,
            full_out,
            instructions=instructions,
            shared_weights=shared_weights,
            internal_weights=False,
        ).double()
    finally:
        e3nn.set_optimization_defaults(**old_defaults)

    assert (
        isinstance(ours, torch.jit.ScriptModule)
        if jit_script_fx
        else isinstance(ours, torch.fx.GraphModule)
    )
    assert ours.instructions == instructions
    assert (
        ours.weight_numel
        == reference.weight_numel
        == len(instructions) * mul_in * mul_out
    )
    assert sorted(
        i for group in _instruction_groups_for(ours) for i in group[3]
    ) == list(range(len(instructions)))
    assert ours.dim_in % pad == ours.dim_out % pad == 0

    torch.manual_seed(17)
    batchdim = 2
    # Start with independent physical matrices in published instruction order.
    # The strided and e3nn flat weight layouts are different and intentionally
    # converted independently from these matrices, never from group metadata.
    shape = (len(instructions), mul_out, mul_in)
    matrices = torch.randn(
        shape if shared_weights else (batchdim,) + shape,
        dtype=torch.float64,
        requires_grad=True,
    )
    if shared_weights:
        packed = weights_from_per_instruction(ours, matrices)
    else:
        packed = torch.stack(
            [weights_from_per_instruction(ours, row) for row in matrices]
        )
    e3nn_weight = matrices.transpose(-1, -2).reshape(
        (-1,) if shared_weights else (batchdim, -1)
    )
    x = torch.randn(
        batchdim, mul_in, ours.dim_in, dtype=torch.float64, requires_grad=True
    )
    if internal_weights:
        with torch.no_grad():
            ours.w.copy_(packed)
        _, _, unpacked_instructions, unpacked = weights_per_instruction(ours)
        assert unpacked_instructions == instructions
        assert torch.equal(unpacked, matrices.detach())
        assert torch.equal(weights_from_per_instruction(ours, unpacked), ours.w)

    def forward(value):
        result = ours(value) if internal_weights else ours(value, packed)
        assert result.shape == (batchdim, mul_out, ours.dim_out)
        assert torch.count_nonzero(result[..., irreps_out.dim :]) == 0
        return _strided_to_cat(irreps_out, mul_out, result)

    actual = forward(x)
    expected = reference(_strided_to_cat(irreps_in, mul_in, x), e3nn_weight)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    first = torch.autograd.grad(actual.square().sum(), x, create_graph=True)[0]
    expected_first = torch.autograd.grad(expected.square().sum(), x, create_graph=True)[
        0
    ]
    torch.testing.assert_close(first, expected_first, atol=1e-11, rtol=1e-11)
    second = torch.autograd.grad(first.square().sum(), x, retain_graph=True)[0]
    expected_second = torch.autograd.grad(
        expected_first.square().sum(), x, retain_graph=True
    )[0]
    torch.testing.assert_close(second, expected_second, atol=1e-10, rtol=1e-10)
    # Include weights in the derivative check, including mixed weight/input
    # derivatives required by force and stress supervision.
    if not internal_weights:
        weight_grad = torch.autograd.grad(
            first.square().sum(), matrices, retain_graph=True
        )[0]
        expected_weight_grad = torch.autograd.grad(
            expected_first.square().sum(), matrices, retain_graph=True
        )[0]
        torch.testing.assert_close(
            weight_grad, expected_weight_grad, atol=1e-10, rtol=1e-10
        )
    else:
        assert torch.isfinite(
            torch.autograd.grad(first.square().sum(), ours.w)[0]
        ).all()

    input_layout = StridedLayout(full_in, pad_to_multiple=pad)
    for rotation in (
        o3.rand_matrix(dtype=torch.float64),
        -torch.eye(3, dtype=torch.float64),
    ):
        in_transform = full_in.D_from_matrix(rotation)
        out_transform = full_out.D_from_matrix(rotation)
        rotated_cat = _strided_to_cat(irreps_in, mul_in, x.detach()) @ in_transform.T
        rotated = input_layout.to_strided(rotated_cat)
        torch.testing.assert_close(
            forward(rotated), actual @ out_transform.T, atol=1e-10, rtol=1e-10
        )


@pytest.mark.parametrize("jit_script_fx", [False, True])
def test_split_linear_gradcheck_and_capture(jit_script_fx):
    old_defaults = e3nn.get_optimization_defaults()
    e3nn.set_optimization_defaults(jit_script_fx=jit_script_fx)
    try:
        linear = Linear("0e + 1o + 0e", "0e", shared_weights=True).double()
    finally:
        e3nn.set_optimization_defaults(**old_defaults)
    x = torch.randn(2, 1, linear.dim_in, dtype=torch.float64, requires_grad=True)
    w = torch.randn(linear.weight_numel, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(linear, (x, w))
    assert torch.autograd.gradgradcheck(linear, (x, w))
    if not jit_script_fx:
        from torch.fx.experimental.proxy_tensor import make_fx

        captured = make_fx(linear)(x, w)
        torch.testing.assert_close(captured(x, w), linear(x, w))
        grad = torch.autograd.grad(captured(x, w).square().sum(), x, create_graph=True)[
            0
        ]
        assert torch.isfinite(torch.autograd.grad(grad.square().sum(), w)[0]).all()


def test_scripted_split_linear_save_load_preserves_public_metadata(tmp_path):
    """A saved scripted module must rebuild split-group metadata losslessly.

    TorchScript does not serialize the private ``_instruction_groups`` helper
    attribute.  The public instruction/irrep ABI is sufficient for the helper
    to reconstruct it after loading, and internal weights must still unpack in
    the original published instruction order.
    """
    instructions = [(3, 2), (1, 0), (4, 1), (0, 0), (3, 0), (0, 2), (1, 2)]
    old_defaults = e3nn.get_optimization_defaults()
    e3nn.set_optimization_defaults(jit_script_fx=True)
    try:
        linear = Linear(
            "0e + 0e + 1o + 0e + 3o",
            "0e + 3o + 0e",
            instructions=instructions,
            shared_weights=True,
            internal_weights=True,
            pad_to_alignment=4,
        ).double()
    finally:
        e3nn.set_optimization_defaults(**old_defaults)

    torch.manual_seed(29)
    matrices = torch.randn(
        len(instructions), linear.mul_out, linear.mul_in, dtype=torch.float64
    )
    packed = weights_from_per_instruction(linear, matrices)
    with torch.no_grad():
        linear.w.copy_(packed)

    path = tmp_path / "split-linear.pt"
    torch.jit.save(linear, str(path))
    loaded = torch.jit.load(str(path)).double()

    assert loaded.instructions == instructions
    assert loaded.irreps_in == linear.irreps_in
    assert loaded.irreps_out == linear.irreps_out
    assert _instruction_groups_for(loaded) == _instruction_groups_for(linear)
    _, _, unpacked_instructions, unpacked = weights_per_instruction(loaded)
    assert unpacked_instructions == instructions
    torch.testing.assert_close(unpacked, matrices)
    torch.testing.assert_close(
        weights_from_per_instruction(loaded, unpacked), loaded.w.detach()
    )

    x = torch.randn(2, linear.mul_in, linear.dim_in, dtype=torch.float64)
    torch.testing.assert_close(linear(x), loaded(x), atol=0.0, rtol=0.0)


def test_lmax3_split_linear_fx_capture():
    """FX capture must retain l=3 disjoint/reordered instruction semantics."""
    old_defaults = e3nn.get_optimization_defaults()
    e3nn.set_optimization_defaults(jit_script_fx=False)
    try:
        linear = Linear(
            "0e + 0e + 1o + 0e + 3o",
            "0e + 3o + 0e",
            instructions=[
                (3, 2),
                (1, 0),
                (4, 1),
                (0, 0),
                (3, 0),
                (0, 2),
                (1, 2),
            ],
            shared_weights=True,
            pad_to_alignment=4,
        ).double()
    finally:
        e3nn.set_optimization_defaults(**old_defaults)

    from torch.fx.experimental.proxy_tensor import make_fx

    x = torch.randn(
        3, linear.mul_in, linear.dim_in, dtype=torch.float64, requires_grad=True
    )
    w = torch.randn(linear.weight_numel, dtype=torch.float64, requires_grad=True)
    captured = make_fx(linear)(x, w)
    torch.testing.assert_close(captured(x, w), linear(x, w), atol=1e-12, rtol=1e-12)
    grad = torch.autograd.grad(captured(x, w).square().sum(), x, create_graph=True)[0]
    second = torch.autograd.grad(grad.square().sum(), w)[0]
    assert torch.isfinite(second).all()


@pytest.mark.parametrize(
    "instructions,error",
    [
        ([(0, 0)], ValueError),  # scalar cannot feed an odd vector
        ([(2, 0)], IndexError),
        ([(-1, 0)], IndexError),
        ([(1, 1)], IndexError),
        ([(1, -1)], IndexError),
        ([(1, 0), (1, 0)], ValueError),
    ],
)
def test_invalid_linear_instructions_rejected(instructions, error):
    with pytest.raises(error):
        Linear("0e + 1o", "1o", instructions=instructions, shared_weights=True)
