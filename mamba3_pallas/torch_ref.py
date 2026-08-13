"""PyTorch reference for the Mamba-3 SISO layer, for cross-framework parity.

Derived from Mamba (https://github.com/state-spaces/mamba), Apache License 2.0:
Copyright (c) 2026, Dao AI Lab, Goombalab (mamba_ssm/modules/mamba3.py) and
Copyright (c) 2025, Dao AI Lab, Goombalab (tests/ops/triton/test_mamba3_siso.py).
See the NOTICE file at the repo root. Changes: MIMO and varlen paths removed.

Vendored rather than imported: ``mamba_ssm`` requires triton, tilelang and CUDA at
import time, none of which exist on a TPU host or a laptop. This file needs only
``torch`` on CPU.

Two pieces:

* `mamba3_siso_ref` -- the recurrent kernel-boundary reference, transcribed from
  ``mamba3_siso_step_ref`` in ``mamba/tests/ops/triton/test_mamba3_siso.py``. Note
  it works in the **original interleaved** rotary layout, unlike the JAX side.
* `Mamba3SISOTorch` -- the layer around it, transcribed from ``Mamba3.forward`` in
  ``mamba/mamba_ssm/modules/mamba3.py`` with MIMO and varlen removed. Its
  ``state_dict`` keys match upstream, so `convert.torch_to_jax` accepts a real
  checkpoint.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def heavy_tail_activation(x: torch.Tensor) -> torch.Tensor:
    """``1 + x`` for ``x >= 0``, ``1 / (1 - x)`` for ``x < 0``.

    Verbatim from ``mamba/mamba_ssm/modules/mamba3.py``.
    """
    neg = x.clamp_max(0)
    pos = x.clamp_min(0)
    return pos + torch.reciprocal(1 - neg)


def apply_rotary_interleaved(
    tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rotate adjacent channel pairs -- the layout upstream actually uses.

    ``tensor`` is viewed as ``(..., n/2, 2)`` and ``[..., 0]`` mixes with
    ``[..., 1]``. When ``cos`` covers fewer pairs than the tensor has, the remainder
    is padded with ``cos = 1, sin = 0`` so those channels pass through -- that is the
    ``rope_fraction < 1`` case.

    The JAX side instead uses a contiguous half-split, which is why
    `convert.torch_to_jax` applies a permutation; see `layout.deinterleave_perm`.
    """
    reshaped = tensor.view(*tensor.shape[:-1], -1, 2)
    t0, t1 = reshaped[..., 0], reshaped[..., 1]
    if cos.shape[-1] < t0.shape[-1]:
        pad = t0.shape[-1] - cos.shape[-1]
        cos = F.pad(cos, (0, pad), value=1.0)
        sin = F.pad(sin, (0, pad), value=0.0)
    return torch.stack([t0 * cos - t1 * sin, t0 * sin + t1 * cos], dim=-1).view_as(tensor)


@torch.no_grad()
def mamba3_siso_ref(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    ADT: torch.Tensor,
    DT: torch.Tensor,
    Trap: torch.Tensor,
    Q_bias: torch.Tensor,
    K_bias: torch.Tensor,
    Angles: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    Z: Optional[torch.Tensor] = None,
):
    """Recurrent SISO reference at the kernel boundary.

    Args:
      Q, K: ``(B, L, H, N)`` post-norm C and B, interleaved rotary layout.
      V: ``(B, L, H, P)``.
      ADT, DT, Trap: ``(B, H, L)``. ``Trap`` is pre-sigmoid.
      Q_bias, K_bias: ``(H, N)``.
      Angles: ``(B, L, H, Nr)`` pre-tanh.
      D: ``(H,)`` or None. Z: ``(B, L, H, P)`` or None.

    Returns:
      ``(out, (angle_state, ssm_state, k_state, v_state))``; ``out`` is
      ``(B, L, H, P)``.
    """
    batch, seqlen, nheads, headdim_qk = Q.shape
    headdim_v = V.shape[-1]
    n_angles = Angles.shape[-1]
    device = Q.device
    two_pi = 2 * math.pi

    angles = torch.tanh(Angles) * math.pi

    angle_state = torch.zeros((batch, nheads, n_angles), dtype=torch.float32, device=device)
    ssm_state = torch.zeros(
        (batch, nheads, headdim_v, headdim_qk), dtype=torch.float32, device=device
    )
    k_state = torch.zeros((batch, nheads, headdim_qk), dtype=Q.dtype, device=device)
    v_state = torch.zeros((batch, nheads, headdim_v), dtype=V.dtype, device=device)

    outs = []
    for t in range(seqlen):
        q = Q[:, t] + Q_bias.unsqueeze(0)
        k = K[:, t] + K_bias.unsqueeze(0)
        v = V[:, t]
        adt, dt = ADT[:, :, t], DT[:, :, t]
        lam = torch.sigmoid(Trap[:, :, t])

        angle_state = angle_state + angles[:, t] * dt.unsqueeze(-1)
        angle_state = angle_state - two_pi * torch.floor(angle_state / two_pi)
        cos, sin = torch.cos(angle_state), torch.sin(angle_state)
        q_rot = apply_rotary_interleaved(q, cos, sin)
        k_rot = apply_rotary_interleaved(k, cos, sin)

        alpha = torch.exp(adt)
        beta = (1 - lam) * dt * alpha
        gamma = lam * dt

        ssm_state = (
            alpha[..., None, None] * ssm_state
            + beta[..., None, None] * (v_state.unsqueeze(-1) * k_state.unsqueeze(-2))
            + gamma[..., None, None] * (v.unsqueeze(-1) * k_rot.unsqueeze(-2))
        )
        out = torch.einsum("bhdD,bhD->bhd", ssm_state, q_rot.to(ssm_state.dtype))
        if D is not None:
            out = out + D[None, :, None] * v
        if Z is not None:
            z = Z[:, t]
            out = out * z * torch.sigmoid(z)
        outs.append(out)
        k_state, v_state = k_rot, v
    return torch.stack(outs, dim=1), (angle_state, ssm_state, k_state, v_state)


class Mamba3SISOTorch(torch.nn.Module):
    """SISO-only transcription of upstream ``Mamba3``.

    Parameter names and the ``in_proj`` split order match upstream exactly, so
    ``state_dict()`` here is interchangeable with a real Mamba-3 checkpoint's
    SISO subset. MIMO (``mimo_x``/``mimo_z``/``mimo_o``), varlen packing, and the
    optional output-projection norm are omitted.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        headdim: int = 64,
        expand: int = 2,
        rope_fraction: float = 0.5,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        a_floor: float = 1e-4,
        norm_eps: float = 1e-5,
        dtype=torch.float32,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.headdim = headdim
        self.d_inner = expand * d_model
        assert self.d_inner % headdim == 0
        self.nheads = self.d_inner // headdim
        self.a_floor = a_floor
        self.norm_eps = norm_eps

        assert rope_fraction in (0.5, 1.0)
        width = int(d_state * rope_fraction)
        if width % 2:
            width -= 1
        self.n_angles = width // 2

        d_in_proj = 2 * self.d_inner + 2 * d_state + 3 * self.nheads + self.n_angles
        self.in_proj = torch.nn.Linear(d_model, d_in_proj, bias=False, dtype=dtype)
        self.out_proj = torch.nn.Linear(self.d_inner, d_model, bias=False, dtype=dtype)

        dt = torch.exp(
            torch.rand(self.nheads, dtype=torch.float32)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        self.dt_bias = torch.nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        # Upstream stores these as (H, R, N) with R = mimo_rank = 1 for SISO.
        self.B_bias = torch.nn.Parameter(1 + torch.zeros((self.nheads, 1, d_state)))
        self.C_bias = torch.nn.Parameter(1 + torch.zeros((self.nheads, 1, d_state)))
        self.B_norm_weight = torch.nn.Parameter(torch.ones(d_state, dtype=dtype))
        self.C_norm_weight = torch.nn.Parameter(torch.ones(d_state, dtype=dtype))
        self.D = torch.nn.Parameter(torch.ones(self.nheads))

    @torch.no_grad()
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """``u`` is ``(B, L, d_model)``; returns the same shape."""
        batch, seqlen, _ = u.shape
        proj = self.in_proj(u)
        sizes = [
            self.d_inner, self.d_inner, self.d_state, self.d_state,
            self.nheads, self.nheads, self.nheads, self.n_angles,
        ]
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(proj, sizes, dim=-1)

        z = z.view(batch, seqlen, self.nheads, self.headdim)
        x = x.view(batch, seqlen, self.nheads, self.headdim)

        # Negate first, then clamp, same as upstream. Folding these into
        # `-heavy_tail(x).clamp(max=-floor)` parses as `-(heavy_tail(x).clamp(...))`, and
        # since heavy_tail is strictly positive that saturates everything to -floor and
        # then flips it to +floor, giving a growing state instead of a decaying one.
        a = -heavy_tail_activation(dd_A.float())
        a = a.clamp(max=-self.a_floor)
        dt = F.softplus(dd_dt + self.dt_bias)
        ADT = (a * dt).permute(0, 2, 1).contiguous()
        DT = dt.permute(0, 2, 1).contiguous()
        Trap = trap.permute(0, 2, 1).contiguous()

        B = rms_norm_torch(B, self.B_norm_weight, self.norm_eps)
        C = rms_norm_torch(C, self.C_norm_weight, self.norm_eps)
        # B and C are shared across heads (ngroups == 1).
        B = B.unsqueeze(2).expand(-1, -1, self.nheads, -1)
        C = C.unsqueeze(2).expand(-1, -1, self.nheads, -1)
        angles = angles.unsqueeze(2).expand(-1, -1, self.nheads, -1).float()

        # C is the query, B the key.
        y, _ = mamba3_siso_ref(
            Q=C, K=B, V=x, ADT=ADT, DT=DT, Trap=Trap,
            Q_bias=self.C_bias.squeeze(1), K_bias=self.B_bias.squeeze(1),
            Angles=angles, D=self.D, Z=z,
        )
        y = y.reshape(batch, seqlen, self.d_inner)
        return self.out_proj(y.to(u.dtype))


def rms_norm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last axis, f32 internally.

    Matches ``rms_norm_ref`` in ``mamba/mamba_ssm/ops/triton/layernorm_gated.py``:
    no mean subtraction, no bias.
    """
    xf = x.float()
    rstd = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    return (xf * rstd * weight.float()).to(x.dtype)
