"""Plots: velocity contour, streamlines, PINN-vs-DNS profiles, convergence curves.

Pioch-style (Fig. 6-8). All functions save a PNG and return the path.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def speed_contour(flds, path, title="PINN |U|"):
    fig, ax = plt.subplots(figsize=(9, 2.8))
    pc = ax.contourf(flds["X"], flds["Y"], flds["speed"], levels=30, cmap="viridis")
    ax.set_aspect("equal"); ax.set_xlabel("x/h"); ax.set_ylabel("y/h"); ax.set_title(title)
    fig.colorbar(pc, ax=ax, shrink=0.9, label="|U|/U0")
    return _save(fig, path)


def streamlines(flds, path, title="PINN streamlines"):
    fig, ax = plt.subplots(figsize=(9, 2.8))
    spd = np.nan_to_num(flds["speed"])
    ax.streamplot(flds["X"], flds["Y"], np.nan_to_num(flds["U"]), np.nan_to_num(flds["V"]),
                  density=1.4, color=spd, cmap="viridis", linewidth=0.7, arrowsize=0.6)
    ax.set_aspect("equal"); ax.set_xlabel("x/h"); ax.set_ylabel("y/h"); ax.set_title(title)
    return _save(fig, path)


def compare_dns_vs_pinn(flds_pinn, flds_dns, path):
    """Two stacked speed contours: PINN (top) vs DNS (bottom)."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True)
    for ax, fl, t in zip(axes, (flds_pinn, flds_dns), ("PINN |U|", "DNS |U|")):
        pc = ax.contourf(fl["X"], fl["Y"], fl["speed"], levels=30, cmap="viridis")
        ax.set_aspect("equal"); ax.set_ylabel("y/h"); ax.set_title(t)
        fig.colorbar(pc, ax=ax, shrink=0.9)
    axes[-1].set_xlabel("x/h")
    return _save(fig, path)


def profiles(stations, path, field="U"):
    """stations: list of dicts {xh, y, pinn, dns} for a velocity component."""
    n = len(stations)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.2), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, s in zip(axes, stations):
        ax.plot(s["dns"], s["y"], "k-", label="DNS")
        ax.plot(s["pinn"], s["y"], "r--", label="PINN")
        ax.set_title(f"x/h={s['xh']}"); ax.set_xlabel(field)
    axes[0].set_ylabel("y/h"); axes[0].legend(fontsize=8)
    return _save(fig, path)


def convergence(history, path, keys=("loss", "loss/r_omega", "loss/r_k")):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    steps = [h["step"] for h in history]
    for k in keys:
        if k in history[0]:
            ax.semilogy(steps, [h[k] for h in history], label=k)
    ax.set_xlabel("step"); ax.set_ylabel("loss (log)"); ax.legend(fontsize=8)
    ax.set_title("convergence")
    return _save(fig, path)
