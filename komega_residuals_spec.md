# k–ω PINN — 반영 수식 (residual) 사양 · 교과서 형태

2D 정상·비압축 RANS + 표준 Wilcox(1988) k–ω. 무차원화: 길이 h, 속도 U₀, ρ=1, ν=1/Re=1/5100.
PINN 출력: u, v, p, k, ω  (**직접 출력 = Pioch 1:1**; 선택적으로 g=1/√ω 전환 가능).  좌표 미분은 자동미분.

---

## 1. 닫힘(closure) 관계

평균 변형률
$$
S_{xx}=\frac{\partial u}{\partial x},\quad
S_{yy}=\frac{\partial v}{\partial y},\quad
S_{xy}=S_{yx}=\tfrac12\!\left(\frac{\partial u}{\partial y}+\frac{\partial v}{\partial x}\right)
$$

와점성 (기본: ω 직접 출력; `omega_param=g`이면 ω=1/g²)
$$
\nu_t=\frac{k}{\omega}
$$

부시네스크 레이놀즈 응력 (2D 성분)
$$
\tau'_{xx}=2\nu_t\frac{\partial u}{\partial x}-\frac{2}{3}k,\qquad
\tau'_{yy}=2\nu_t\frac{\partial v}{\partial y}-\frac{2}{3}k,\qquad
\tau'_{xy}=\tau'_{yx}=\nu_t\!\left(\frac{\partial u}{\partial y}+\frac{\partial v}{\partial x}\right)
$$

난류 생성항
$$
P=\tau'_{xx}\frac{\partial u}{\partial x}
 +\tau'_{xy}\!\left(\frac{\partial u}{\partial y}+\frac{\partial v}{\partial x}\right)
 +\tau'_{yy}\frac{\partial v}{\partial y}
\;=\;\tau'_{ij}\frac{\partial u_i}{\partial x_j}
$$

---

## 2. 잔차 (각각 = 0)

연속
$$
\mathcal{R}_{\text{cont}}=\frac{\partial u}{\partial x}+\frac{\partial v}{\partial y}
$$

x-운동량
$$
\mathcal{R}_{\text{mom}x}=
u\frac{\partial u}{\partial x}+v\frac{\partial u}{\partial y}
+\frac{\partial p}{\partial x}
-\nu\!\left(\frac{\partial^2u}{\partial x^2}+\frac{\partial^2u}{\partial y^2}\right)
-\left(\frac{\partial \tau'_{xx}}{\partial x}+\frac{\partial \tau'_{xy}}{\partial y}\right)
$$

y-운동량
$$
\mathcal{R}_{\text{mom}y}=
u\frac{\partial v}{\partial x}+v\frac{\partial v}{\partial y}
+\frac{\partial p}{\partial y}
-\nu\!\left(\frac{\partial^2v}{\partial x^2}+\frac{\partial^2v}{\partial y^2}\right)
-\left(\frac{\partial \tau'_{yx}}{\partial x}+\frac{\partial \tau'_{yy}}{\partial y}\right)
$$

k 수송
$$
\mathcal{R}_{k}=
u\frac{\partial k}{\partial x}+v\frac{\partial k}{\partial y}
-P+\beta^{*}k\omega
-\frac{\partial}{\partial x}\!\left[(\nu+\sigma^{*}\nu_t)\frac{\partial k}{\partial x}\right]
-\frac{\partial}{\partial y}\!\left[(\nu+\sigma^{*}\nu_t)\frac{\partial k}{\partial y}\right]
$$

ω 수송
$$
\mathcal{R}_{\omega}=
u\frac{\partial \omega}{\partial x}+v\frac{\partial \omega}{\partial y}
-\alpha\frac{\omega}{k}P+\beta\omega^{2}
-\frac{\partial}{\partial x}\!\left[(\nu+\sigma\nu_t)\frac{\partial \omega}{\partial x}\right]
-\frac{\partial}{\partial y}\!\left[(\nu+\sigma\nu_t)\frac{\partial \omega}{\partial y}\right]
$$

상수 (Wilcox 1988)
$$
\beta^{*}=\tfrac{9}{100},\quad \sigma^{*}=\sigma=\tfrac12,\quad
\alpha=\tfrac59,\quad \beta=\tfrac{3}{40}
$$

PDE 손실
$$
\text{MSE}_{\text{pde}}=\big\langle
\mathcal{R}_{\text{cont}}^2+\mathcal{R}_{\text{mom}x}^2+\mathcal{R}_{\text{mom}y}^2
+\mathcal{R}_{k}^2+\mathcal{R}_{\omega}^2\big\rangle
$$

---

## 3. Pioch(2023)와의 대응 및 보정

| 본 사양 | Pioch 식 | 비고 |
|---|---|---|
| 𝓡_cont | (3) | 동일 |
| 𝓡_momx, 𝓡_momy | (4),(5) + (6),(7) | 레이놀즈 힘 = ∇·τ′ (식 6,7로 정의) |
| τ′ (Boussinesq) | (8) | 동일 |
| ν_t = k/ω | (9) | 동일 |
| 𝓡_k | (10) | 동일 |
| 𝓡_ω | (11) | **생성항 보정**: 아래 참조 |

**보정 1 — ω 생성항.** Pioch 식(11)은 `α (k/ω) P`로 *인쇄*되어 있으나 차원이 맞지 않음.
표준 Wilcox는 `α (ω/k) P` ( = α P/ν_t ). 본 사양은 **α (ω/k) P** 채택.

**보정 2 — 레이놀즈 힘 부호/규약.** 운동량의 난류항은 물리적으로 응력 텐서의 발산
∂τ′_ij/∂x_j (τ′=2ν_t S−⅔kδ). 본 사양은 이 **표준·일관 규약**으로 작성(Pioch 식 4–6의
표기를 명확화한 것).

**보정 3 — ω 벽 특이성.** 네트워크가 g를 출력하고 ω=1/g². 벽에서 g→0(유한),
ω의 1/y² 발산을 자동 처리. (Pioch는 벽 ω 조건 미기재 → 본 사양에서 보완.)

---

## 4. 자동미분 차수 (구현 점검표)

| 항 | 필요한 도함수 |
|---|---|
| 𝓡_cont | u_x, v_y (1차) |
| 𝓡_momx/y 대류·압력 | 1차 |
| 𝓡_momx/y 점성 | u_xx,u_yy,v_xx,v_yy (2차) |
| 𝓡_momx/y 레이놀즈 힘 ∇·τ′ | τ′ 안에 u,v 1차 + k,ω; ∇·τ′ 는 **u,v 2차 + k,ω 1차** |
| 생성항 P | u,v 1차 (+ν_t) |
| 𝓡_k, 𝓡_ω 확산 | k,ω 2차 + ν_t 1차 |

→ 최대 **2차 도함수**. JAX: `jacfwd(jacrev(...))` 또는 jvp/vjp 중첩 (jaxpi 제공).
ω=1/g², ν_t=k/ω 는 정의만 하면 연쇄법칙은 자동미분이 처리.
