# 코드 골격 설계 (구조만 · 코드는 지시 시 작성)

jaxpi 기반. **네트워크만 교체**되고 physics·loss·data·post는 공유. 모든 선택지는 **config**로.

---

## 1. 디렉터리 구조

```
bfs_komega_pinn/
├── data/                         # (구축 완료)
│   ├── download_case031.py       # DNS 다운로드
│   ├── load_dns.py               # 파싱 + to_pioch_frame(x+3, 입구 y+h)  ← 추가 예정
│   ├── raw/case031/              # .dat
│   └── processed/                # 캐시(npz)
├── configs/
│   ├── base.py                   # 공통 기본값
│   ├── plain_mlp.py              # arch=mlp
│   ├── modified_mlp.py           # arch=modified_mlp
│   └── pirate.py                 # arch=pirate
├── src/
│   ├── geometry.py               # 도메인+notch, collocation/경계점 샘플링
│   ├── networks.py               # MLP(=Pioch) / PirateNet (동일 I/O)
│   ├── physics.py                # 잔차 5개·닫힘·ω=1/g²·autodiff
│   ├── boundary.py               # slip/no-slip/inlet/outlet, DNS clamp
│   ├── data_pipeline.py          # load_dns→Pioch frame→BC/data 점 생성
│   ├── losses.py                 # MSE_pde/bc/data + 가중 3종 + RBA
│   ├── model.py                  # KOmegaPINN: net+physics+loss 결합
│   └── train.py                  # 학습 루프(Adam/L-BFGS)·N seed 루프·체크포인트·wandb 로깅
├── evaluation/                   # ← train 직후 결과 검증 (point 2)
│   └── metrics.py                # NMSE·FAC2·FB·V + 수용한계 + select_best + 재부착길이
├── postprocess/                  # ← evaluation 후 시각화 (point 3)
│   ├── fields.py                 # 격자 평가, notch 마스킹, 속도장 복원
│   ├── plots.py                  # contour · streamline · DNS 비교 · 차이장 · 수렴곡선
│   └── compare.py                # 아키텍처별 비교 패널/표
├── experiments/
│   └── run_comparison.py         # 2 arch(MLP·PirateNet) × N seed 오케스트레이션
├── colab/
│   └── run.ipynb                 # Colab 부트스트랩(클론·설치·데이터·학습·검증·후처리)
├── requirements.txt              # jax[cuda12]·flax·optax·ml_collections·wandb·scipy·matplotlib
├── .gitignore                    # data/raw·checkpoints·wandb·__pycache__·*.npz
└── README.md                     # Colab 실행 안내(순서/배지)
```

---

## 2. Config 스키마 (선택 가능한 옵션 전부)

```yaml
arch: mlp | pirate                         # ← 비교 변수(이것만 교체)
arch_params:
  depth, width, activation: tanh
  fourier_scale: s          # pirate/embedding
  n_blocks, alpha_init: 0   # pirate
  pi_init: true             # pirate: 최종층 최소제곱

loss:
  weighting: uniform | grad_norm | ntk      # ← 3종 모두 구현, config로 선택
  lambda_init: {pde: 1, bc: 1, data: 1}     # uniform일 때 수동 가중
  rba: {enabled: true|false, gamma: 0.99, eta: 0.1}   # 점별 가중 on/off

data:
  mode: none | lines3 | full                # Pioch §5.1/5.2/5.3 재현
  data_stations: [7.0, 9.0, 12.98, 17.98]   # mode에 따라 자동 선택(Pioch x/h)
  use_fields: [U, V]                         # Pioch 동일(U,V만)

bc:
  top: slip                                  # Pioch 동일
  outlet: dirichlet | neumann                # 기본 dirichlet(x/h=22), 전환 가능
  omega_wall: g_transform                    # ω=1/g²

geometry:
  bbox: [[0,22],[0,6]]                       # Pioch 프레임
  step_notch: {x: [0,3], y: [0,1]}           # solid 제외 + no-slip
  n_collocation: 2000                        # Pioch (민감도분석 1000/2000/4000 → 2000)
  n_bc: 2000                                 # Pioch (BC+data 합 2000)
  sampling: uniform                          # 등간격, 비교 통제

optim:
  adam: {lr: [1e-3→1e-4], steps: 30000}
  lbfgs: {enabled: true, steps}

run:
  n_seeds: 10                                # Pioch 동일(10회 재초기화) → 성능 분포 비교
  select_best_by: nmse                       # 비물리 샘플 제거 후 최저 NMSE 선택(Pioch)

metrics:                                     # Pioch 평가지표 (point 3·4)
  use: [nmse, fac2, fb, v]                   # 수용한계: NMSE<4, FAC2>0.5, FB<0.2
  vars: [U, V]                               # DNS 스테이션에서 비교

wandb:                                       # 학습 로깅 (point 5)
  project, entity
  mode: online | offline
  group: <arch>                              # 아키텍처별 그룹
  tags: [arch, weighting, data_mode, seed]
  log: 손실(총/pde/bc/data)·잔차별·lr·λ가중·grad-norm / run요약(NMSE,FAC2,FB,V,재부착)

postprocess:
  contour: true, streamline: true            # point 3 (Pioch Fig.6–8)
  compare_dns: true, reattachment: true
  convergence_curves: true
  trial_distribution: true                   # seed별 NMSE 분포(box/violin) — point 4
```

---

## 3. 데이터 흐름 (호출 그래프)

```
config
  ├─ geometry.py     → collocation 점 + 경계 점 (notch 제외, BC 그룹별)
  ├─ data_pipeline.py→ load_dns → to_pioch_frame(x+3, 입구 y+h)
  │                    → 입구/출구/내부선 (U,V) 값
  ├─ networks.py     → net = build_network(config.arch)    # ← 교체점
  └─ losses.py       → 가중기(uniform|grad_norm|ntk) + RBA

model.py: KOmegaPINN(net, physics, losses, bc, data)
  └─ loss(params) = Σ λ_r·MSE_pde_r + λ_bc·MSE_bc + λ_data·MSE_data

train.py → optimize(Adam→LBFGS), 시드 루프, 체크포인트 저장
evaluation → trial별 NMSE/FAC2/FB/V·재부착 계산 → select_best (비물리 제거)
postprocess → 검증 통과 모델 로드 → 격자 평가 → contour/streamline/DNS비교/compare
```

핵심 인터페이스(설계 수준, 시그니처만):
- `networks.build_network(cfg) → net`  (net.apply(params, xy) → (u,v,p,k,g))
- `physics.residuals(net, params, xy) → (R_cont,R_momx,R_momy,R_k,R_ω)`  (autodiff 2차)
- `boundary.bc_terms(...) → MSE_bc` / `data_pipeline.data_terms(...) → MSE_data`
- `losses.aggregate(residuals, bc, data, scheme, rba) → scalar`
- `model.loss(params, batch) → scalar`

> 비교 통제: physics·boundary·data_pipeline·losses·geometry·optim·evaluation·postprocess는 **두 아키텍처 공통**.
> `config.arch` 한 줄로만 네트워크가 바뀜.

---

## 4. 검증 (evaluation) — train 직후 (point 2)

`evaluation/metrics.py` 가 각 학습 결과를 **Pioch 지표로 검증**하고 모델을 선별 →
**이 단계를 통과한 모델만 후처리로 전달**:
- **NMSE**(정규화 평균제곱오차), **FAC2**(관측의 ½~2배 내 비율),
  **FB**(분율 편향), **V**(Oberkampf–Trucano 검증지표). DNS 스테이션에서 U,V 비교.
  수용한계 **NMSE<4 · FAC2>0.5 · FB<0.2**.
- **재부착 길이**: 하단벽 근처 U 부호 전환점(또는 Cf=0) → 9.28h와 비교.
- `select_best(trials)`: **비물리 샘플 제거 → 최저 NMSE 모델 선택** (Pioch 방식).
- **trial 분포**: seed별 NMSE/FAC2/FB/V 평균±σ·최저 (초기화 민감성, point 4).
- 출력: 검증표(지표 × seed) + **선별된 대표 모델** → postprocess로 전달.

## 5. 후처리 (postprocess) — 검증 후 (point 3)

`postprocess/` 가 **검증 통과 모델**만 시각 비교:
1. **fields.py**: 유체영역(bbox−notch) 정규격자 → net 평가 → notch 마스킹 →
   속도장 (u,v), |U|, p, k, ω 복원.
2. **plots.py**:
   - **속도 크기 contour** — Pioch Fig.6–8 형식.
   - **streamline** — 재순환 버블·재부착 시각화.
   - **PINN vs DNS 나란히 + 차이장(deviation)**.
   - **수렴곡선** (손실/잔차 vs step).
3. **compare.py**: 2 아키텍처(MLP·PirateNet, × N seeds)의 contour·streamline·
   지표·재부착·**분포(box/violin)**를 한 패널/표로 → 최종 비교 결과물.

---

## 6. 비교 설계 확정 사항
- **아키텍처 집합**: **MLP(Pioch 재현) vs PirateNet** 2종만.
- **초기화 민감성(point 4)**: 아키텍처마다 **n_seeds=10**(Pioch 동일) 학습 →
  비물리 제거 → 최저 NMSE 선택, 동시에 **지표 분포(평균±σ, box/violin)** 보고.
  "PirateNet이 분산↓" 이 핵심 결과.
- **collocation**: PDE 2000 + BC/data 2000, 등간격(Pioch). 두 아키텍처 동일.

## 7. 실행 환경 · repo/Colab 워크플로우

작업 분담: **노트북**(편집·git push, 학습 X) → **GitHub** → **Colab GPU**(학습·검증·후처리).

### 7.1 repo 추가 파일
- `requirements.txt`: jax[cuda12]·flax·optax·ml_collections·wandb·numpy·scipy·matplotlib
  (+ jaxpi는 PyPI 미배포 → GitHub에서 설치, commit hash 고정)
- `colab/run.ipynb`: 부트스트랩 노트북(아래 순서)
- `.gitignore`: `data/raw/`, `checkpoints/`, `wandb/`, `__pycache__/`, `*.npz`, `figs/`
- `README.md`: Colab 실행 순서 + (선택) "Open in Colab" 배지

### 7.2 Colab 부트스트랩 순서 (run.ipynb)
1. (영속성) **Google Drive 마운트** → checkpoints/figs를 Drive에 저장 (Colab 세션 휘발성)
2. `git clone <repo>` (또는 `git pull`)
3. `pip install -r requirements.txt` + jaxpi
4. `python data/download_case031.py` (Colab 인터넷 가능 → DNS 받기)
5. `wandb login` (API key) — 또는 `WANDB_MODE=offline`
6. 학습: `python -m src.train --config configs/pirate.py` (mlp도 동일)
7. 검증: `python -m evaluation.metrics …` → select_best
8. 후처리: `python -m postprocess.compare …` → contour/streamline/지표표

### 7.3 영속성·로깅
- checkpoints·figures → **Google Drive** (또는 repo push)
- 지표·수렴곡선·run 메타 → **wandb** (arch별 group, seed별 run)
- 런타임 끊김 대비: 주기적 체크포인트 + wandb resume

### 7.4 주의
- **JAX-GPU**: Colab 런타임=GPU 선택, `jax[cuda12]`와 CUDA 버전 매칭.
- **jaxpi**: GitHub clone/pip, 버전(commit) 고정.
- **DNS 데이터**: 작아도(~210KB) raw는 gitignore + Colab에서 다운로드 권장(라이선스 CC BY-NC-SA).

## 8. 다음 단계
- 본 골격 승인 시, `geometry.py`부터 순서대로 실제 코드 작성(지시 대기).
- 우선순위(제안): geometry → data_pipeline(load_dns 보강) → physics → networks →
  losses → model → train(+wandb) → **evaluation(검증)** → postprocess.
- repo 부속(requirements·colab/run.ipynb·.gitignore)은 코드 착수 시 함께 생성.
