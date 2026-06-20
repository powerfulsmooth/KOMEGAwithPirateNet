"""Base config (ml_collections) for the k-omega BFS PINN, Pioch frame.

All selectable options live here. mlp.py / pirate.py import this and override
only the `arch` block (the comparison variable).
"""
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.mode = "train"  # train | eval

    # ---- problem: 2D steady incompressible RANS + Wilcox(1988) k-omega ----
    config.problem = ml_collections.ConfigDict()
    config.problem.Re = 5100.0
    config.problem.nu = 1.0 / 5100.0          # non-dim molecular viscosity
    config.problem.rho = 1.0
    config.problem.omega_param = "direct"     # "direct" (Pioch 1:1) | "g" (omega = 1/g^2)
    config.problem.omega_floor = 1e-6         # guard for nu_t = k/omega when direct
    # Wilcox(1988) closure constants
    config.problem.beta_star = 9.0 / 100.0
    config.problem.sigma_star = 0.5
    config.problem.sigma = 0.5
    config.problem.alpha = 5.0 / 9.0
    config.problem.beta = 3.0 / 40.0

    # ---- geometry (Pioch frame): bbox [0,22]x[0,6], step notch [0,3]x[0,1] ----
    config.geom = ml_collections.ConfigDict()
    config.geom.x_range = (0.0, 22.0)
    config.geom.y_range = (0.0, 6.0)
    config.geom.step_x = (0.0, 3.0)           # solid notch x-extent
    config.geom.step_y = (0.0, 1.0)           # solid notch y-extent (= step height h=1)
    config.geom.n_collocation = 2000          # Pioch (sensitivity 1000/2000/4000 -> 2000)
    config.geom.n_bc = 2000                   # Pioch (BC + data total)
    config.geom.sampling = "uniform"          # uniform (equispaced) | random

    # ---- DNS data (Pioch frame). mode: none | lines3 | full ----
    config.data = ml_collections.ConfigDict()
    config.data.mode = "full"
    config.data.use_fields = ("U", "V")       # Pioch uses U,V only
    config.data.raw_dir = "data/raw/case031"
    config.data.inlet_xh = 0.0                # Pioch x/h
    config.data.outlet_xh = 22.0
    config.data.lines3 = (7.0, 12.98, 17.98)  # interior lines for mode=lines3
    config.data.full = (7.0, 9.0, 12.98, 17.98)

    # ---- boundary conditions ----
    config.bc = ml_collections.ConfigDict()
    config.bc.top = "slip"                    # Pioch: slip top
    config.bc.outlet = "dirichlet"            # dirichlet (DNS @ x/h=22) | neumann (d/dx=0)

    # ---- network architecture (jaxpi archs.from_config); overridden per-arch ----
    config.arch = ml_collections.ConfigDict()
    config.arch.arch_name = "Mlp"
    config.arch.num_layers = 5
    config.arch.hidden_dim = 128
    config.arch.out_dim = 5                   # u, v, p, k, (omega | g)
    config.arch.activation = "tanh"
    config.arch.periodicity = None
    config.arch.fourier_emb = None
    config.arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 1.0, "stddev": 0.1}
    )

    # ---- optimizer ----
    config.optim = ml_collections.ConfigDict()
    config.optim.optimizer = "Adam"
    config.optim.learning_rate = 1e-3
    config.optim.decay_rate = 0.9
    config.optim.decay_steps = 2000
    config.optim.grad_accum_steps = 0
    config.optim.lbfgs = ml_collections.ConfigDict({"enabled": True, "steps": 8000})

    # ---- training ----
    config.training = ml_collections.ConfigDict()
    config.training.max_steps = 1000
    config.training.batch_size = 2000

    # ---- loss weighting: uniform | grad_norm | ntk  (+ optional RBA) ----
    config.weighting = ml_collections.ConfigDict()
    config.weighting.scheme = "grad_norm"
    config.weighting.init_weights = ml_collections.ConfigDict({
        "r_cont": 1.0, "r_momx": 1.0, "r_momy": 1.0, "r_k": 1.0, "r_omega": 1.0,
        "bc": 1.0, "data": 1.0,
    })
    config.weighting.momentum = 0.9
    config.weighting.update_every_steps = 1000
    config.weighting.rba = ml_collections.ConfigDict(
        {"enabled": False, "gamma": 0.999, "eta": 0.01}
    )

    # ---- run / seeds (init-sensitivity study) ----
    config.run = ml_collections.ConfigDict()
    config.run.n_seeds = 1                   # Pioch reinit count
    config.run.seed = 0
    config.run.select_best_by = "nmse"

    # ---- evaluation metrics (Pioch) ----
    config.metrics = ml_collections.ConfigDict()
    config.metrics.use = ("nmse", "fac2", "fb", "v")
    config.metrics.accept = ml_collections.ConfigDict(
        {"nmse_max": 4.0, "fac2_min": 0.5, "fb_max": 0.2}
    )

    # ---- wandb logging ----
    config.wandb = ml_collections.ConfigDict()
    config.wandb.project = "bfs-komega-pinn"
    config.wandb.entity = None
    config.wandb.mode = "online"              # online | offline | disabled
    config.wandb.name = None
    config.wandb.tags = ()

    # # ---- saving ----
    # config.saving = ml_collections.ConfigDict()
    # config.saving.checkpoint_dir = "checkpoints"
    # config.saving.save_every_steps = 5000
    # config.saving.figs_dir = "figs"

    # return config

    # ---- saving ----
    config.saving = ml_collections.ConfigDict()
    
    # 💡 [수정] 구글 드라이브 내 특정 프로젝트 폴더 지정
    # 개별 Case 구분을 위해 프로젝트 루트 경로를 정의합니다.
    config.saving.base_dir = "/content/drive/MyDrive/bfs-komega-pinn"
    
    # 각 실험 결과물(체크포인트 가중치, 시각화 그림)이 들어갈 상세 경로 설정
    config.saving.checkpoint_dir = f"{config.saving.base_dir}/checkpoints"
    config.saving.figs_dir = f"{config.saving.base_dir}/figs"
    config.saving.save_every_steps = 5000
    config.saving.contour_every_steps = 500   # 학습 중 |U| 등고선 스냅샷 주기(0이면 끔)

    # ⚠️ [주의] train.py는 같은 이름의 .pkl(예: mlp_seed0.pkl)이 이미 있으면 그 seed를 건너뜁니다.
    #   따라서 max_steps(학습 길이)나 seed 수를 바꿔 "다시 제대로" 학습할 때는 반드시
    #   위 checkpoints 폴더의 옛 .pkl을 비우세요. 안 그러면 후처리가 옛 모델을 평가합니다.
    #   여러 학습 길이를 섞어 비교하려면 아래 2줄 주석을 풀어 스텝별로 폴더를 분리하세요:
    # config.saving.checkpoint_dir = f"{config.saving.base_dir}/checkpoints/s{config.training.max_steps}"
    # config.saving.figs_dir = f"{config.saving.base_dir}/figs/s{config.training.max_steps}"

    return config


