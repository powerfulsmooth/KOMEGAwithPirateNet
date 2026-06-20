# """End-to-end comparison driver: load checkpoints -> evaluate -> postprocess -> compare.

# Run AFTER training both archs (checkpoints in cfg.saving.checkpoint_dir):
#     python -m experiments.run_comparison --config=configs/base.py

# Expects checkpoints named "<arch>_seed<seed>.pkl". Produces figures in cfg.saving.figs_dir.
# """
# import os
# import glob
# import pickle

# import numpy as np
# import jax
# import jax.numpy as jnp
# from absl import app, flags
# from ml_collections import config_flags

# from src import networks, data_pipeline
# from postprocess import fields, plots, compare
# from evaluation import metrics

# FLAGS = flags.FLAGS
# config_flags.DEFINE_config_file("config", "configs/base.py", "config")

# ARCHS = ("mlp", "pirate")


# def make_predict_fn(cfg, params):
#     net = networks.build_network(cfg)

#     def predict_fn(xy):
#         xy = jnp.asarray(xy)
#         return np.asarray(jax.vmap(lambda pt: net.apply(params, pt))(xy))
#     return predict_fn


# def _arch_cfg(cfg, arch):
#     # ensure arch block matches the checkpoint (mlp vs pirate)
#     c = cfg.copy_and_resolve_references()
#     if arch == "pirate":
#         import ml_collections
#         c.arch.arch_name = "PirateNet"
#         c.arch.num_layers = 3
#         c.arch.hidden_dim = 256
#         c.arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1.0, "embed_dim": 256})
#         c.arch.nonlinearity = 0.0
#     else:
#         c.arch.arch_name = "Mlp"
#         c.arch.num_layers = 5
#         c.arch.hidden_dim = 128
#     return c


# def evaluate_arch(cfg, arch, data_pts):
#     c = _arch_cfg(cfg, arch)
#     ckpts = sorted(glob.glob(os.path.join(cfg.saving.checkpoint_dir, f"{arch}_seed*.pkl")))
#     # DNS station points for metrics (inlet + outlet + interior)
#     obs_xy, obs_uv = [], []
#     for key in ("inlet", "outlet", "interior"):
#         if data_pts.get(key) is not None:
#             obs_xy.append(data_pts[key][0]); obs_uv.append(data_pts[key][1])
#     obs_xy = np.concatenate(obs_xy, 0); obs_uv = np.concatenate(obs_uv, 0)

#     trials, reattach, best = [], [], {"nmse": np.inf}
#     for path in ckpts:
#         with open(path, "rb") as fh:
#             params = pickle.load(fh)
#         predict_fn = make_predict_fn(c, params)
#         pred_uv = predict_fn(obs_xy)[:, :2]
#         m = metrics.all_metrics(pred_uv, obs_uv)
#         xw, uw = fields.near_wall_u(predict_fn, c)
#         m["reattach"] = metrics.reattachment_length(xw, uw, x_step=c.geom.step_x[1])
#         m["tag"] = os.path.basename(path)
#         trials.append(m); reattach.append(m["reattach"])
#         if m["nmse"] < best["nmse"]:
#             best = {**m, "params": params}

#     res = {k: [t[k] for t in trials] for k in ("nmse", "fac2", "fb", "v")}
#     res["reattach"] = reattach
#     res["best_speed"] = None
#     if "params" in best:
#         fl = fields.evaluate(make_predict_fn(c, best["params"]), c)
#         res["best_speed"] = (fl["X"], fl["Y"], fl["speed"])
#     _, res["summary"] = metrics.select_best(trials, c) if trials else (None, {})
#     return res


# def main(argv):
#     cfg = FLAGS.config
#     data_pts = data_pipeline.build_data_points(cfg)
#     figdir = cfg.saving.figs_dir
#     results = {}
#     for arch in ARCHS:
#         results[arch] = evaluate_arch(cfg, arch, data_pts)

#     plots_done = []
#     plots_done.append(compare.metric_distribution(results, os.path.join(figdir, "nmse_dist.png"), "nmse"))
#     plots_done.append(compare.reattachment_bar(results, os.path.join(figdir, "reattach.png")))
#     plots_done.append(compare.contours_side_by_side(results, os.path.join(figdir, "contours.png")))
#     print(compare.summary_table(results))
#     print("figures:", [p for p in plots_done if p])


# if __name__ == "__main__":
#     app.run(main)



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
    
    # 💡 [추가] 만약 해당 구조의 체크포인트 파일이 하나도 없으면 에러 방지를 위해 None 반환
    if not ckpts:
        print(f"--> [{arch}] 구조의 체크포인트 파일(.pkl)이 존재하지 않아 평가를 건너뜁니다.")
        return None

    obs_xy, obs_uv = [], []
    for key in ("inlet", "outlet", "interior"):
        if data_pts.get(key) is not None:
            obs_xy.append(data_pts[key][0]); obs_uv.append(data_pts[key][1])
    obs_xy = np.concatenate(obs_xy, 0); obs_uv = np.concatenate(obs_uv, 0)

    trials, reattach, best = [], [], {"nmse": np.inf}
    for path in ckpts:
        print(f"--> [{arch}] 파일 평가 중: {os.path.basename(path)}")
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
    
    # 💡 구글 드라이브 시각화 폴더 생성 보장
    os.makedirs(figdir, exist_ok=True)

    results = {}
    for arch in ARCHS:
        res = evaluate_arch(cfg, arch, data_pts)
        if res is not None:  # 파일이 정상적으로 존재하는 모델군만 취합
            results[arch] = res

    # 💡 [예외 처리 핵심] 수집된 결과가 하나도 없거나, 비교 대상이 부족할 때 시각화 모듈 에러 방지
    if not results:
        print("❌ 평가할 수 있는 체크포인트 데이터가 전혀 없습니다. 학습을 먼저 진행해 주세요.")
        return

    plots_done = []
    
    # 두 모델(`mlp`, `pirate`)이 모두 존재해야만 온전히 작동하는 플롯 함수들을 위한 안전장치
    try:
        plots_done.append(compare.metric_distribution(results, os.path.join(figdir, "nmse_dist.png"), "nmse"))
    except Exception as e:
        print(f"⚠️ 분포도(metric_distribution) 생성 실패 (데이터가 부족할 수 있음): {e}")

    try:
        plots_done.append(compare.reattachment_bar(results, os.path.join(figdir, "reattach.png")))
    except Exception as e:
        print(f"⚠️ 재부착 길이 바 차트(reattachment_bar) 생성 실패: {e}")

    try:
        plots_done.append(compare.contours_side_by_side(results, os.path.join(figdir, "contours.png")))
    except Exception as e:
        print(f"⚠️ 등고선 비교도(contours_side_by_side) 생성 실패: {e}")

    # 요약 테이블과 생성 완료된 그림 경로 출력
    try:
        print("\n" + "="*40 + "\n   [학습 결과 요약 테이블]   \n" + "="*40)
        print(compare.summary_table(results))
    except Exception as e:
        print(f"⚠️ 요약 테이블 출력 실패: {e}")
        
    print("\n[생성된 시각화 리스트]:", [p for p in plots_done if p])


if __name__ == "__main__":
    app.run(main)

