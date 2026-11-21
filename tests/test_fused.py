"""The fused in-place carve loop must match the reference (allocate-per-seam) loop exactly."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from reframe.core.energy import backward_energy, forward_costs, to_gray
from reframe.core.seams import carve_width, find_seam, remove_seam


def reference_carve(img, n, mode, extra, hook):
    h, w = img.shape[:2]
    gray = to_gray(img)
    cols = np.broadcast_to(np.arange(w, dtype=np.int64), (h, w)).copy()
    order = np.full((h, w), -1, dtype=np.int64)
    rows = np.arange(h)
    costs_out = []
    for k in range(n):
        costs = forward_costs(gray)
        base = extra if mode == "forward" else backward_energy(gray) + extra
        hook_term = None
        if hook is not None:
            hook_term = hook(k, cols)
            base = base + hook_term
        seam, total = find_seam(base, costs if mode == "forward" else None)
        if hook_term is not None:
            total -= float(hook_term[rows, seam].sum())
        costs_out.append(total)
        order[rows, cols[rows, seam]] = k
        gray = remove_seam(gray, seam)
        extra = remove_seam(extra, seam)
        cols = remove_seam(cols, seam)
    return order, costs_out


@given(
    st.integers(2, 30),
    st.integers(3, 40),
    st.integers(0, 2**32 - 1),
    st.sampled_from(["forward", "backward"]),
    st.booleans(),
)
def test_fused_matches_reference(h, w, seed, mode, with_hook):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    n = int(rng.integers(0, w))
    extra = rng.uniform(0, 50, (h, w))

    def hook(k, cols):
        return np.abs(cols.astype(np.float64) - (k * 3 % w)) * 0.7

    hk = hook if with_hook else None
    ref_order, ref_costs = reference_carve(img, n, mode, extra.copy(), hk)
    res = carve_width(img, n, mode=mode, extra_energy=extra, seam_hook=hk)
    np.testing.assert_array_equal(res.order, ref_order)
    np.testing.assert_allclose(res.seam_costs, ref_costs, rtol=1e-12, atol=1e-9)
