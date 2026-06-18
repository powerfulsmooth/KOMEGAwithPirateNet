"""End-to-end comparison driver: load checkpoints -> evaluate -> postprocess -> compare.

Run AFTER training both archs (checkpoints in cfg.saving.checkpoint_dir):
    python -m experiments.run_comparison --config=configs/base.py

Expects checkpoints named "<arch>_seed<seed>.pkl". Produces figures in cfg.saving.figs_dir.
"""
import os
import glob
import pickle

import numpy as np
import jax
import jax.numpy as jnp
from absl import app, flags
from ml_collections import config_flags

from src import networks, data_pipeline
from postprocess import fields, plots, compare
from evaluation import metrics

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "configs/base.py", "config")

ARCHS = ("mlp", "pirate")


def make_predict_fn(cfg, params):
    net = networks.build_network(cfg)

    def predict_fn(xy):
        xy = jnp.asarray(xy)
        return np.asarray(jax.vmap(lambda pt: net.apply(params, pt))(xy))
    return predict_fn


def _arch_cfg(cfg, arch):
    # ensure arch block matches the checkpoint (mlp vs pirate)
    c = cfg.copy_and_resolve_references()
    if arch == "pirate":
        import ml_collections
        c.arch.arch_name = "PirateNet"
        c.arch.num_layers = 3
        c.arch.hidden_dim = 256
        c.arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1.0, "embed_dim": 256})
        c.arch.nonlinearity = 0.0
    else:
        c.arch.arch_name = "Mlp"
        c.arch.num_layers = 5
        c.arch.hidden_dim = 128
    return c


def evaluate_arch(cfg, arch, data_pts):
    c = _arch_cfg(cfg, arch)
    ckpts = sorted(glob.glob(os.path.join(cfg.saving.checkpoint_dir, f"{arch}_seed*.pkl")))
    # DNS station points for metrics (inlet + outlet + interior)
    obs_xy, obs_uv = [], []
    for key in ("inlet", "outlet", "interior"):
        if data_pts.get(key) is not None:
            obs_xy.append(data_pts[key][0]); obs_uv.append(data_pts[key][1])
    obs_xy = np.concatenate(obs_xy, 0); obs_uv = np.concatenate(obs_uv, 0)

    trials, reattach, best = [], [], {"nmse": np.inf}
    for path in ckpts:
        with open(path, "rb") as fh:
            params = pickle.load(fh)
        predict_fn = make_predict_fn(c, params)
        pred_uv = predict_fn(obs_xy)[:, :2]
        m = metrics.all_metrics(pred_uv, obs_uv)
        xw, uw = fields.near_wall_u(predict_fn, c)
        m["reattach"] = metrics.reattachment_length(xw, uw, x_step=c.geom.step_x[1])
        m["tag"] = os.path.basename(path)
        trials.append(m); reattach.append(m["reattach"])
        if m["nmse"] < best["nmse"]:
            best = {**m, "params": params}

    res = {k: [t[k] for t in trials] for k in ("nmse", "fac2", "fb", "v")}
    res["reattach"] = reattach
    res["best_speed"] = None
    if "params" in best:
        fl = fields.evaluate(make_predict_fn(c, best["params"]), c)
        res["best_speed"] = (fl["X"], fl["Y"], fl["speed"])
    _, res["summary"] = metrics.select_best(trials, c) if trials else (None, {})
    return res


def main(argv):
    cfg = FLAGS.config
    data_pts = data_pipeline.build_data_points(cfg)
    figdir = cfg.saving.figs_dir
    results = {}
    for arch in ARCHS:
        results[arch] = evaluate_arch(cfg, arch, data_pts)

    plots_done = []
    plots_done.append(compare.metric_distribution(results, os.path.join(figdir, "nmse_dist.png"), "nmse"))
    plots_done.append(compare.reattachment_bar(results, os.path.join(figdir, "reattach.png")))
    plots_done.append(compare.contours_side_by_side(results, os.path.join(figdir, "contours.png")))
    print(compare.summary_table(results))
    print("figures:", [p for p in plots_done if p])


if __name__ == "__main__":
    app.run(main)
