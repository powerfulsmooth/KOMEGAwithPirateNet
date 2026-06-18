"""Geometry for the backward-facing step in the Pioch frame.

Domain (bbox)   : x in [0, 22], y in [0, 6]   (units of step height h)
Solid step notch: x in [0, 3],  y in [0, 1]   -> excluded from the fluid region
Walls:
  - top      y = 6                 : slip   (Pioch)
  - bottom   y = 0, x in [3, 22]   : no-slip
  - step top y = 1, x in [0, 3]    : no-slip
  - step face x = 3, y in [0, 1]   : no-slip
Inlet  x = 0  (y in [1, 6]) and outlet x = 22 (y in [0, 6]) carry DNS U,V
(handled in data_pipeline.py).

Everything here is plain numpy (one-time preprocessing). Arrays are (N, 2) = [x, y].
"""
import numpy as np


def _ranges(cfg):
    (x0, x1) = cfg.geom.x_range
    (y0, y1) = cfg.geom.y_range
    (sx0, sx1) = cfg.geom.step_x
    (sy0, sy1) = cfg.geom.step_y
    return x0, x1, y0, y1, sx0, sx1, sy0, sy1


def in_solid(xy, cfg):
    """True where (x, y) is inside the solid step notch."""
    x, y = xy[:, 0], xy[:, 1]
    _, _, _, _, sx0, sx1, sy0, sy1 = _ranges(cfg)
    return (x >= sx0) & (x <= sx1) & (y >= sy0) & (y <= sy1)


def in_fluid(xy, cfg):
    """True where (x, y) is inside the fluid region (bbox minus notch)."""
    x, y = xy[:, 0], xy[:, 1]
    x0, x1, y0, y1, _, _, _, _ = _ranges(cfg)
    inside_bbox = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    return inside_bbox & (~in_solid(xy, cfg))


def sample_collocation(cfg, rng=None):
    """Return ~n_collocation interior points in the fluid region."""
    x0, x1, y0, y1, _, _, _, _ = _ranges(cfg)
    n = int(cfg.geom.n_collocation)
    area = (x1 - x0) * (y1 - y0)
    notch = (cfg.geom.step_x[1] - cfg.geom.step_x[0]) * (cfg.geom.step_y[1] - cfg.geom.step_y[0])
    fluid_area = area - notch

    if cfg.geom.sampling == "uniform":
        # equispaced grid sized so #fluid points ~= n
        h = np.sqrt(fluid_area / n)
        nx = max(2, int(round((x1 - x0) / h)))
        ny = max(2, int(round((y1 - y0) / h)))
        gx = np.linspace(x0, x1, nx)
        gy = np.linspace(y0, y1, ny)
        gx, gy = np.meshgrid(gx, gy)
        xy = np.stack([gx.ravel(), gy.ravel()], axis=1)
        xy = xy[in_fluid(xy, cfg)]
    else:  # random uniform with rejection of the notch
        rng = np.random.default_rng(0) if rng is None else rng
        pts = []
        while sum(len(p) for p in pts) < n:
            cand = np.stack([rng.uniform(x0, x1, 4 * n), rng.uniform(y0, y1, 4 * n)], axis=1)
            pts.append(cand[in_fluid(cand, cfg)])
        xy = np.concatenate(pts, axis=0)[:n]
    return xy.astype(np.float32)


def wall_points(cfg, density=None):
    """No-slip and slip wall points, grouped by segment.

    density = points per unit length (default chosen for ~a few hundred total).
    Returns dict: {"top","bottom","step_top","step_face"} -> (M,2) arrays.
    """
    x0, x1, y0, y1, sx0, sx1, sy0, sy1 = _ranges(cfg)
    if density is None:
        density = 20.0  # pts per unit length

    def line(p_from, p_to):
        p_from, p_to = np.asarray(p_from, float), np.asarray(p_to, float)
        length = np.linalg.norm(p_to - p_from)
        m = max(2, int(round(length * density)))
        t = np.linspace(0.0, 1.0, m)[:, None]
        return (p_from[None, :] * (1 - t) + p_to[None, :] * t).astype(np.float32)

    return {
        "top": line((x0, y1), (x1, y1)),                 # slip
        "bottom": line((sx1, y0), (x1, y0)),             # no-slip (x in [3,22])
        "step_top": line((x0, sy1), (sx1, sy1)),         # no-slip (y = 1, x in [0,3])
        "step_face": line((sx1, sy0), (sx1, sy1)),       # no-slip (x = 3, y in [0,1])
    }


def _pt_seg_dist(xy, a, b):
    """Distance from points xy (N,2) to segment a-b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    denom = np.dot(ab, ab) + 1e-12
    t = np.clip(((xy - a) @ ab) / denom, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(xy - proj, axis=1)


def wall_distance(xy, cfg):
    """Distance to the nearest no-slip wall (bottom, step_top, step_face).

    Used for the omega wall asymptote / optional input feature.
    """
    x0, x1, y0, y1, sx0, sx1, sy0, sy1 = _ranges(cfg)
    xy = np.asarray(xy, float)
    d_bottom = _pt_seg_dist(xy, (sx1, y0), (x1, y0))
    d_steptop = _pt_seg_dist(xy, (x0, sy1), (sx1, sy1))
    d_stepface = _pt_seg_dist(xy, (sx1, sy0), (sx1, sy1))
    return np.minimum(np.minimum(d_bottom, d_steptop), d_stepface).astype(np.float32)
