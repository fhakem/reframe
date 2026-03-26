"""Temporally coherent video seam carving (after Grundmann et al., CVPR 2010).

"Discontinuous seam carving for video retargeting" computes seams frame by
frame (not as 2-D surfaces in the space-time volume) and adds a temporal
coherence cost to each frame's DP so that seam ``k`` of frame ``t`` stays close
to seam ``k`` of frame ``t-1``. Seams are free to jump when content demands it
(the cost is soft and clipped), which is what makes them *discontinuous* in time
and lets them follow motion better than connected space-time surfaces.

Per seam ``k`` and pixel ``(i, j)`` of the work image the extra DP term is::

    lam * min(|x_src(i, j) - p_k(i)|, clip)                (position)
    + appearance_weight * |I_t(i, j) - A_k(i)| / 255       (appearance, optional)

* ``x_src = work column + crop_x`` and ``p_k`` is the previous frame's seam ``k``
  in **source-frame columns** at analysis resolution, warped forward by the
  optical flow if given. Keeping seams in source columns is what makes the term
  correct when the crop window (M4) moves between frames.
* ``A_k(i)`` is the luma of the pixel removed by seam ``k`` at row ``i`` in the
  previous frame (Grundmann's appearance coherence, simplified to the removed
  pixel itself rather than the whole re-joined neighbourhood).

Smoothing choice: seam positions are outputs of a discrete DP, so post-hoc
filtering (EMA) would produce invalid seams. Temporal smoothing is done *inside*
the DP through ``lam``: a larger ``lam`` is a stronger low-pass on seam motion;
``clip`` bounds the cost of a jump so that structural changes can still win.
:meth:`TemporalSeamCarver.reset` (on shot cuts) drops all temporal influence.
"""

from __future__ import annotations

import numpy as np

from reframe.core.energy.gradient import to_gray
from reframe.core.flow import FlowArray, warp_columns
from reframe.core.metrics import seam_positions
from reframe.core.seams.carve import CarveResult, EnergyMode, SeamHookFn, carve_width
from reframe.core.types import BoolArray, FloatArray, ImageArray, IntArray

#: Default position weight (forward units per pixel of displacement per row).
DEFAULT_LAM = 3.0
#: Default displacement (px) above which the position cost stops growing.
DEFAULT_CLIP = 10.0


class TemporalSeamCarver:
    """Stateful per-shot carver: call :meth:`carve` on consecutive frames, :meth:`reset` on cuts."""

    def __init__(
        self,
        n_seams: int,
        mode: EnergyMode = "forward",
        *,
        lam: float = DEFAULT_LAM,
        clip: float = DEFAULT_CLIP,
        appearance_weight: float = 0.0,
    ) -> None:
        if n_seams < 0:
            raise ValueError("n_seams must be >= 0")
        self.n_seams = n_seams
        self.mode: EnergyMode = mode
        self.lam = float(lam)
        self.clip = float(clip)
        self.appearance_weight = float(appearance_weight)
        self._prev_src: FloatArray | None = None
        self._prev_luma: FloatArray | None = None

    def reset(self) -> None:
        """Forget the previous frame (next :meth:`carve` equals plain ``carve_width``)."""
        self._prev_src = None
        self._prev_luma = None

    @property
    def prev_seams_src(self) -> FloatArray | None:
        """(n, h) previous seams in source-frame columns (analysis res), or None."""
        return None if self._prev_src is None else self._prev_src.copy()

    def carve(
        self,
        work_img: ImageArray,
        *,
        crop_x: float,
        extra_energy: FloatArray | None = None,
        protect: BoolArray | None = None,
        flow: FlowArray | None = None,
    ) -> CarveResult:
        """Carve ``n_seams`` from the work image (crop window at analysis resolution).

        ``crop_x``: left edge of the window in source-frame analysis columns.
        ``flow``: optional forward flow from the previous to this *full analysis
        frame* (``estimate_flow(prev_analysis_gray, analysis_gray)``), source
        coordinates, same height as ``work_img``.
        """
        img = np.asarray(work_img)
        h = img.shape[0]
        n = self.n_seams
        prev = self._prev_src
        if prev is not None and prev.shape != (n, h):
            self.reset()
            prev = None
        gray = to_gray(img)

        hook: SeamHookFn | None = None
        if prev is not None and (self.lam > 0 or self.appearance_weight > 0):
            target = prev
            if flow is not None:
                if flow.shape[0] != h:
                    raise ValueError(f"flow height {flow.shape[0]} != work height {h}")
                target = warp_columns(prev, flow)
            target_work = target - float(crop_x)
            luma = self._prev_luma
            rows = np.arange(h)[:, None]
            lam, clip, aw = self.lam, self.clip, self.appearance_weight

            def _hook(k: int, cols: IntArray) -> FloatArray:
                term = np.abs(cols - target_work[k][:, None])
                np.minimum(term, clip, out=term)
                term *= lam
                if aw > 0 and luma is not None:
                    term += (aw / 255.0) * np.abs(gray[rows, cols] - luma[k][:, None])
                return term

            hook = _hook

        res = carve_width(
            img, n, mode=self.mode, protect=protect, extra_energy=extra_energy, seam_hook=hook
        )
        pos = seam_positions(res.order)
        idx = pos.astype(np.int64)
        self._prev_luma = gray[np.arange(h)[None, :], idx]
        self._prev_src = pos + float(crop_x)
        return res
