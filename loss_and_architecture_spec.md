# 손실 함수 + PirateNet 구조 사양 · Pioch 비교

본 연구의 학습 구조를 (1) 손실 함수, (2) PirateNet 아키텍처로 정리하고 Pioch(2023)와 대조한다.
**비교 통제 원칙**: 입력·출력·손실 항·손실 가중·collocation·옵티마이저·스텝·데이터는 두 아키텍처
(Plain MLP / PirateNet)에서 **모두 동일**. 오직 *네트워크* 만 교체.

---

## 1. 손실 함수 구조

### 1.1 전체 손실
$$
\mathcal{L}(\theta)=
\underbrace{\sum_{r}\lambda_r\,\mathcal{L}^{\text{pde}}_r}_{\text{물리}}
+\lambda_{\text{bc}}\,\mathcal{L}_{\text{bc}}
+\lambda_{\text{data}}\,\mathcal{L}_{\text{data}},
\qquad r\in\{\text{cont},\text{mom}x,\text{mom}y,k,\omega\}
$$

### 1.2 각 항
PDE 잔차 (점별 RBA 가중 $w_i$ 포함)
$$
\mathcal{L}^{\text{pde}}_r=\frac{1}{N_f}\sum_{i=1}^{N_f} w_i\,\big|\mathcal{R}_r(x_i,y_i)\big|^2
$$
경계조건
$$
\mathcal{L}_{\text{bc}}=\frac{1}{N_b}\sum_{b}\big|\,\mathcal{B}(\hat{q}(x_b,y_b))-\text{BC}_b\,\big|^2
$$
데이터 (내부 라벨선; Pioch의 0/3/5선 = config)
$$
\mathcal{L}_{\text{data}}=\frac{1}{N_d}\sum_{d}\big|\,\hat{q}(x_d,y_d)-q^{\text{DNS}}_d\,\big|^2,
\quad q\in\{U,V\}\ \text{(Pioch 동일)}
$$

### 1.3 경계조건 $\mathcal{B}$ 의 구체형
| 경계 | u, v | k | ω (= 1/g²) |
|---|---|---|---|
| 하단벽·step (no-slip) | u=v=0 | k=0 | g→ g_wall (ω=6ν/βy²) |
| 상단벽 (slip) | v=0, ∂u/∂y=0 | ∂k/∂y=0 | ∂ω/∂y=0 |
| inlet (x/h=0) | DNS U,V | (Pioch: 미부과) | (Pioch: 미부과) |
| outlet (x/h=22) | DNS U,V 또는 ∂/∂x=0 | ∂k/∂x=0 | ∂ω/∂x=0 |

### 1.4 손실 가중 (jaxpi 기능 — 세 아키텍처 공통 적용)
- **전역 가중 $\lambda_r$**: grad-norm 균형 또는 NTK 기반 (Wang et al. 2022/23).
  목적: 운동량과 스케일이 다른 k·ω 잔차의 기울기 크기를 자동 균형.
  $$
  \lambda_r \leftarrow \frac{\sum_j \lVert\nabla_\theta \mathcal{L}_j\rVert}{\lVert\nabla_\theta \mathcal{L}_r\rVert}\ \ (\text{grad-norm})
  $$
- **점별 RBA 가중 $w_i$** (Anagnostopoulos et al. 2024): 잔차 큰 점에 주의 집중.
  $$
  w_i \leftarrow \gamma\,w_i + \eta\,\frac{|\mathcal{R}(x_i)|}{\max_j|\mathcal{R}(x_j)|}
  $$
- 정상 문제 → causal weighting 불필요.

> 공정 비교를 위해 **가중 스킴도 세 아키텍처에 동일** 적용 (가중은 학습 하니스 소속, 아키텍처 무관).

---

## 2. PirateNet 아키텍처 구조

입력 $\mathbf{x}=(x,y)$, 출력 $\hat q=(u,v,p,k,g)$, 활성화 $\sigma=\tanh$.

### 2.1 랜덤 푸리에 임베딩
$$
\Phi(\mathbf{x})=\big[\cos(\mathbf{B}\mathbf{x}),\ \sin(\mathbf{B}\mathbf{x})\big],
\qquad \mathbf{B}\in\mathbb{R}^{m\times 2},\ B_{ij}\sim\mathcal{N}(0,s^2)
$$
($s$ = 주파수 스케일 하이퍼파라미터; 다중 스케일 가능.)

### 2.2 게이트 인코더 (1회 계산)
$$
\mathbf{U}=\sigma(\mathbf{W}_U\Phi+\mathbf{b}_U),\qquad
\mathbf{V}=\sigma(\mathbf{W}_V\Phi+\mathbf{b}_V)
$$

### 2.3 적응형 residual 블록 ($l=1,\dots,L$), 입력 $\mathbf{x}_1=\sigma(\mathbf{W}_0\Phi+\mathbf{b}_0)$
$$
\begin{aligned}
\mathbf{f}&=\sigma(\mathbf{W}_1^{(l)}\mathbf{x}_l+\mathbf{b}_1^{(l)}), &
\mathbf{z}_1&=\mathbf{f}\odot\mathbf{U}+(1-\mathbf{f})\odot\mathbf{V},\\
\mathbf{g}&=\sigma(\mathbf{W}_2^{(l)}\mathbf{z}_1+\mathbf{b}_2^{(l)}), &
\mathbf{z}_2&=\mathbf{g}\odot\mathbf{U}+(1-\mathbf{g})\odot\mathbf{V},\\
\mathbf{h}&=\sigma(\mathbf{W}_3^{(l)}\mathbf{z}_2+\mathbf{b}_3^{(l)}), &
\mathbf{x}_{l+1}&=\boxed{\ \alpha^{(l)}\,\mathbf{h}+(1-\alpha^{(l)})\,\mathbf{x}_l\ }
\end{aligned}
$$
$\alpha^{(l)}\in\mathbb{R}$ 학습가능, **초기값 0**.

### 2.4 출력층 + 물리정보 초기화
$$
\hat q=\mathbf{W}_{\text{out}}\,\mathbf{x}_{L+1}
$$
- $\alpha^{(l)}=0$ (초기) ⇒ 모든 블록이 항등 ⇒ $\mathbf{x}_{L+1}=\mathbf{x}_1$ ⇒ $\hat q=\mathbf{W}_{\text{out}}\sigma(\mathbf{W}_0\Phi)$
  = **푸리에 특징의 (얕은) 선형사상**으로 출발.
- $\mathbf{W}_{\text{out}}$ 을 **가용 데이터에 대한 최소제곱**으로 초기화:
  $\min_{\mathbf{W}_{\text{out}}}\lVert \mathbf{W}_{\text{out}}\Psi-Y_{\text{data}}\rVert$.
  → 학습이 **데이터 일관 해 근처**에서 시작 (Pioch의 초기화 민감성 정면 완화 = 핵심 가설).
- 학습이 진행되며 $\alpha^{(l)}$ 가 0에서 증가 → "필요한 만큼 깊이를 키우는" 효과.

---

## 3. Pioch(2023) vs 본 연구 — 종합 비교

| 항목 | Pioch (2023) | 본 연구 (PirateNet) |
|---|---|---|
| 프레임워크 | TensorFlow / DeepXDE | JAX / jaxpi |
| 입력 | (x, y) | (x, y) **[동일]** |
| 입력 임베딩 | 없음 (좌표 직접) | 랜덤 푸리에 특징 |
| 은닉 구조 | plain MLP 5층×128 | 게이트 적응형 residual 블록 (+U,V 인코더) |
| skip 연결 | 없음 | **adaptive α-skip** (α init 0) |
| 활성화 | tanh | tanh |
| 초기화 | 표준 랜덤 → **민감** | α=0 + 최종층 최소제곱 (physics-informed) |
| 출력 | u, v, p, k, ω | u, v, p, k, g (ω=1/g²) |
| 손실 항 | MSE_pde+bc+data (균일 합) | **동일 항** |
| 손실 가중 | 없음(균일) | grad-norm/NTK + RBA (세 아키텍처 공통) |
| ω 벽 조건 | 미기재 | g 변환으로 부과 |
| 옵티마이저 | Adam(30k)+L-BFGS-B | Adam (+ L-BFGS) **[동일]** |
| collocation | 균일 2000 | 균일 (동일·통제) |
| 재초기화 | 10회, 분산 37–70% | 다중 시드 (분산 비교) |
| 가중 민감성 | 큼 | 완화 목표(검증 대상) |

### 비교 통제 요약
- **동일(공유)**: 입력, 출력, 손실 항, 손실 가중 스킴, collocation, 옵티마이저, 스텝, 데이터, 시드 집합.
- **차이(=비교 변수)**: 네트워크 아키텍처 — 임베딩 / 게이트 / α-skip / 초기화.
  (Plain MLP: 임베딩·게이트·skip·pi-init 없음 → Pioch 재현. PirateNet: 전부.)

### 핵심 가설
PirateNet의 **푸리에 임베딩 + α-skip + physics-informed init** 이, k·ω의 stiff한 잔차 수렴과
초기화 민감성(Pioch 약점)을 개선한다 → NMSE 평균↓·분산↓·재부착 정확도↑로 검증.
