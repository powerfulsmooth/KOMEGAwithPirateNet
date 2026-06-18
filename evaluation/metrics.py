"""Pioch evaluation metrics: NMSE, FAC2, FB, V + reattachment + model selection.

Metrics are computed on the velocity magnitude |U| = sqrt(u^2 + v^2) at the DNS
stations (positive-definite -> avoids sign/zero issues), matching Pioch's
velocity-based comparison. Acceptance limits: NMSE<4, FAC2>0.5, FB<0.2.
"""
import numpy as np

_EPS = 1e-8


def nmse(pred, obs):
    pred, obs = np.asarray(pred), np.asarray(obs)
    return float(np.mean((obs - pred) ** 2) / (np.mean(obs) * np.mean(pred) + _EPS))


def fac2(pred, obs):
    pred, obs = np.asarray(pred), np.asarray(obs)
    ratio = (pred + _EPS) / (obs + _EPS)
    return float(np.mean((ratio >= 0.5) & (ratio <= 2.0)))


def fb(pred, obs):
    pred, obs = np.asarray(pred), np.asarray(obs)
    mo, mp = np.mean(obs), np.mean(pred)
    return float(2.0 * (mo - mp) / (mo + mp + _EPS))


def metric_v(pred, obs):
    """Oberkampf-Trucano-style validation metric in [0,1], higher = better."""
    pred, obs = np.asarray(pred), np.asarray(obs)
    rel = np.abs(pred - obs) / (np.abs(obs) + _EPS)
    return float(1.0 - np.mean(np.tanh(rel)))


def all_metrics(pred_uv, obs_uv):
    """pred_uv, obs_uv: (N,2) [U,V]. Metrics on speed magnitude."""
    sp = np.sqrt(np.sum(np.asarray(pred_uv) ** 2, axis=1))
    so = np.sqrt(np.sum(np.asarray(obs_uv) ** 2, axis=1))
    return {"nmse": nmse(sp, so), "fac2": fac2(sp, so),
            "fb": fb(sp, so), "v": metric_v(sp, so)}


def accepts(m, cfg):
    a = cfg.metrics.accept
    return (m["nmse"] < a.nmse_max) and (m["fac2"] > a.fac2_min) and (abs(m["fb"]) < a.fb_max)


def reattachment_length(x_wall, u_wall, x_step=3.0):
    """First x where near-bottom-wall U crosses 0 (negative->positive).

    Returns reattachment x in the Pioch frame (DNS expects ~9.28). NaN if none.
    """
    x_wall = np.asarray(x_wall)
    u_wall = np.asarray(u_wall)
    order = np.argsort(x_wall)
    x, u = x_wall[order], u_wall[order]
    mask = x >= x_step
    x, u = x[mask], u[mask]
    for i in range(1, len(x)):
        if u[i - 1] < 0.0 <= u[i]:
            t = -u[i - 1] / (u[i] - u[i - 1] + _EPS)
            return float(x[i - 1] + t * (x[i] - x[i - 1]))
    return float("nan")


def select_best(trials, cfg, key="nmse"):
    """trials: list of dicts each with metrics + 'tag'. Drop unphysical, pick lowest key.

    Unphysical = non-finite metrics or outside acceptance limits.
    Returns (best_trial, summary_dict). Falls back to lowest key if none accepted.
    """
    finite = [t for t in trials if np.isfinite(t.get(key, np.nan))]
    accepted = [t for t in finite if accepts(t, cfg)]
    pool = accepted if accepted else finite
    best = min(pool, key=lambda t: t[key]) if pool else None
    vals = {m: np.array([t[m] for t in finite if np.isfinite(t.get(m, np.nan))])
            for m in ("nmse", "fac2", "fb", "v")}
    summary = {m: {"mean": float(v.mean()) if v.size else float("nan"),
                   "std": float(v.std()) if v.size else float("nan"),
                   "min": float(v.min()) if v.size else float("nan")}
               for m, v in vals.items()}
    summary["n_accepted"] = len(accepted)
    summary["n_total"] = len(trials)
    return best, summary
