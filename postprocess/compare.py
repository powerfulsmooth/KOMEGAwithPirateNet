"""Architecture comparison: MLP vs PirateNet — metric distributions, reattachment, contours.

results: dict keyed by arch name, each:
  {"nmse": [..per seed..], "fac2": [...], "fb": [...], "v": [...],
   "reattach": [..per seed..], "best_speed": (X, Y, speed) or None}
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


def metric_distribution(results, path, metric="nmse"):
    """Box/violin of a metric across seeds, per architecture (init sensitivity)."""
    archs = list(results.keys())
    data = [np.asarray(results[a][metric], float) for a in archs]
    data = [d[np.isfinite(d)] for d in data]
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    ax.boxplot(data, labels=archs, showmeans=True)
    ax.set_ylabel(metric.upper()); ax.set_title(f"{metric.upper()} across seeds")
    return _save(fig, path)


def reattachment_bar(results, path, dns_ref=9.28):
    archs = list(results.keys())
    means = [np.nanmean(results[a]["reattach"]) for a in archs]
    stds = [np.nanstd(results[a]["reattach"]) for a in archs]
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    ax.bar(archs, means, yerr=stds, capsize=4)
    ax.axhline(dns_ref, color="k", ls="--", label=f"DNS {dns_ref}")
    ax.set_ylabel("reattachment x/h"); ax.legend(); ax.set_title("reattachment length")
    return _save(fig, path)


def contours_side_by_side(results, path):
    archs = [a for a in results if results[a].get("best_speed") is not None]
    if not archs:
        return None
    fig, axes = plt.subplots(len(archs), 1, figsize=(9, 2.8 * len(archs)), squeeze=False)
    for ax, a in zip(axes[:, 0], archs):
        X, Y, sp = results[a]["best_speed"]
        pc = ax.contourf(X, Y, sp, levels=30, cmap="viridis")
        ax.set_aspect("equal"); ax.set_ylabel("y/h"); ax.set_title(f"{a} |U| (best seed)")
        fig.colorbar(pc, ax=ax, shrink=0.9)
    axes[-1, 0].set_xlabel("x/h")
    return _save(fig, path)


def summary_table(results):
    """Return a printable text table of mean+/-std per arch/metric."""
    lines = ["arch        NMSE          FAC2         FB           V            reattach"]
    for a, r in results.items():
        def ms(key):
            v = np.asarray(r[key], float); v = v[np.isfinite(v)]
            return f"{np.mean(v):.3f}+/-{np.std(v):.3f}" if v.size else "n/a"
        lines.append(f"{a:11s} {ms('nmse'):13s} {ms('fac2'):12s} {ms('fb'):12s} "
                     f"{ms('v'):12s} {ms('reattach')}")
    return "\n".join(lines)
