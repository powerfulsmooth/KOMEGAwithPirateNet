# 기하 · 경계조건 설계 (Pioch 좌표계, 1:1 비교용)

> 본 문서는 메인 사양서(PirateNet_kOmega_BFS_PINN_Design_Spec.md)의 §1·§3(기하·BC)을
> **사용자 결정에 따라 대체**한다. 목표: Pioch(2023)와 **1:1 비교** → Pioch 좌표계·BC를 그대로 사용.

## 0. 확정된 결정
1. 도메인 형상: **Pioch식 사각형(bounding box) + step은 내부 no-slip 경계로 표현**.
2. 상단벽: **slip** (Pioch와 동일; DNS는 no-slip이나 단순화 채택).
3. 입구: **U, V만** 주입 (Pioch와 동일; k, ω는 미부과).
4. 출구: Pioch와 동일하게 적용 → **x/h=22 DNS 데이터(Dirichlet)** (§3 참조; Pioch가 명시 안 함).
5. 좌표: **Pioch 프레임 사용**, DNS↔Pioch x-shift 및 DNS 데이터 적용 시 y-shift를 설계에 반영(§2).

---

## 1. Pioch 좌표계 & 기하

무차원: 길이 ← h(step height), 속도 ← U₀, ν = 1/Re = 1/5100.

**도메인 (Pioch 프레임)**: bounding box $x/h\in[0,22],\ y/h\in[0,6]$.
- **입구** $x/h=0$ (= DNS/ERCOFTAC $x/h=-3$, step 3h 상류)
- **step** $x/h=3$ (= DNS $x/h=0$), 높이 $h=1$
- **재부착** $x/h\approx 9.28$ (= DNS $6.28$)
- **출구** $x/h=22$ (= DNS $x/h=19$)

**고체 notch (collocation 제외)**: $\{\,0\le x/h\le 3,\ 0\le y/h\le 1\,\}$ (step 블록).
유체 영역 = bounding box − notch (L자형). 유입부 높이 5h(y∈[1,6]), 확장부 높이 6h(y∈[0,6]).

```
 y/h
  6 ┌────────────────────────────────────────────┐  ← slip wall (top)
    │ 유입(DNS U,V)                                │
    │ y∈[1,6]                                      │  유출(x/h=22)
  1 ├───────┐                                      │
    │ notch │   재순환  ↺   재부착≈9.28            │
  0 │(solid)└──────────────────────────────────────┘  ← no-slip (bottom)
    0       3                                      22   x/h  (Pioch)
   inlet   step                                  outlet
```

---

## 2. 좌표 & 데이터 시프트 (point 5 — 핵심)

DNS 데이터(ERCOFTAC Case031)를 Pioch 프레임에 넣을 때의 변환.

### 2.1 x-시프트 (모든 스테이션 공통)
$$
x_{\text{Pioch}} = x_{\text{ERCOFTAC}} + 3
$$

### 2.2 y-시프트 (입구 스테이션만)
ERCOFTAC 상류 프로파일(x-181)은 y가 **유입부 바닥 기준 local 좌표**(0→~5).
Pioch 프레임에선 유입 유동이 $y\in[1,6]$ → **입구 스테이션만 $y_{\text{Pioch}} = y_{\text{data}} + h\,(=1)$**.
하류 스테이션(x-360…744)은 이미 전역 좌표($y\in[0,6]$) → **y-시프트 없음**.

### 2.3 스테이션 매핑표
| 파일 | ERCOFTAC x/h | Pioch x/h | y-shift | 역할 |
|---|---:|---:|---|---|
| x-181 | −2.99 | **0.00** | **+h** | 입구 (BC: U,V) |
| x-360 | 4.00 | 7.00 | 0 | 내부 데이터선 |
| x-411 | 6.00 | 9.00 | 0 | 내부 데이터선 |
| x-513 | 9.98 | 12.98 | 0 | 내부 데이터선 |
| x-641 | 14.98 | 17.98 | 0 | 내부 데이터선 |
| x-744 | 19.00 | 22.00 | 0 | 출구 (BC: U,V) |

### 2.4 변환 함수 (코드 단계에서 load_dns에 반영)
```
to_pioch_frame(station):
    x_pioch = x_ercoftac + 3
    if station is inlet (idx 181):  y_pioch = y_data + 1.0
    else:                           y_pioch = y_data
    (U, V, k, … 값은 불변; 좌표만 변환)
```

---

## 3. 경계조건 (Pioch 프레임)

| 경계 | 위치 (Pioch) | u, v | k, ω |
|---|---|---|---|
| 상단벽 (slip) | y=6 | v=0, ∂u/∂y=0 | (미부과; Pioch 동일) |
| 하단벽 (no-slip) | y=0, x∈[3,22] | u=v=0 | (미부과) |
| step 윗면 (no-slip) | y=1, x∈[0,3] | u=v=0 | (미부과) |
| step 면 (no-slip) | x=3, y∈[0,1] | u=v=0 | (미부과) |
| 입구 | x=0, y∈[1,6] | **DNS U,V** (x-181, y+h) | 미부과 |
| 출구 | x=22, y∈[0,6] | **DNS U,V** (x-744) | 미부과 |
| 내부 데이터선 (config) | x=7,9,12.98,17.98 | DNS U,V (선택) | — |

- k, ω는 어떤 경계에도 직접 부과하지 않음(Pioch 동일) → 내부 잔차로만 결정.
  (이는 Pioch의 약점이나, 1:1 비교 위해 동일 적용.)
- 손실: $\mathcal{L}=\text{MSE}_{\text{pde}}+\text{MSE}_{\text{bc}}+\text{MSE}_{\text{data}}$,
  데이터 모드 = none / 3선 / full (Pioch의 §5.1–5.3 재현).

### 출구 BC — 확인 결과 (point 4)
- Pioch 논문은 PINN의 **출구 조건을 명시하지 않음**. ("convective BC at outlet"은 **DNS** 설명.)
- 확실한 것: x/h=22에 DNS U,V를 공급(6개 선 중 하나).
- **1:1 채택**: 출구 = x/h=22 DNS Dirichlet (U,V). 
- 대안(논문의 "Neumann for u,v" 문구 반영 시): 출구 ∂/∂x=0. → config로 전환 가능하게 둠.

---

## 4. 확인 필요 (open)
- [ ] step notch 표현: 본 설계는 x∈[0,3]에 유입부(y∈[1,6])+solid notch. Pioch가 평평한
      rectangle을 썼을 가능성도 있으나(논문 미상), BFS 물리(재순환·코너와류) 위해 notch 권장.
- [ ] 출구: Dirichlet(x/h=22) vs Neumann(∂/∂x=0) — 기본 Dirichlet, config 전환.
