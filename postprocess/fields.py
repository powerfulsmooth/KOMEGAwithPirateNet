"""Evaluate a trained model on a regular grid (fluid region only; solid notch = NaN)."""
import numpy as np
from src import geometry


def make_grid(cfg, nx=220, ny=60):
    (x0, x1) = cfg.geom.x_range
    (y0, y1) = cfg.geom.y_range
    gx = np.linspace(x0, x1, nx)
    gy = np.linspace(y0, y1, ny)
    X, Y = np.meshgrid(gx, gy)
    xy = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)
    fluid = geometry.in_fluid(xy, cfg)
    return X, Y, xy, fluid


def evaluate(predict_fn, cfg, nx=220, ny=60):
    """predict_fn: (M,2) -> (M,5) [u,v,p,k,omega]. Returns 2D fields, NaN in solid."""
    X, Y, xy, fluid = make_grid(cfg, nx, ny)
    out = np.full((xy.shape[0], 5), np.nan, dtype=np.float32)
    pred = np.asarray(predict_fn(xy[fluid]))
    out[fluid] = pred
    U = out[:, 0].reshape(X.shape)
    V = out[:, 1].reshape(X.shape)
    P = out[:, 2].reshape(X.shape)
    K = out[:, 3].reshape(X.shape)
    W = out[:, 4].reshape(X.shape)
    speed = np.sqrt(U ** 2 + V ** 2)
    return {"X": X, "Y": Y, "U": U, "V": V, "P": P, "k": K, "omega": W, "speed": speed}


def near_wall_u(predict_fn, cfg, y_eval=0.05, n=200):
    """U just above the bottom wall (x in [step, 22]) for reattachment detection."""
    sx1 = cfg.geom.step_x[1]
    x1 = cfg.geom.x_range[1]
    x = np.linspace(sx1, x1, n)
    xy = np.stack([x, np.full_like(x, y_eval)], axis=1).astype(np.float32)
    u = np.asarray(predict_fn(xy))[:, 0]
    return x, u
