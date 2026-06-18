# DNS data — ERCOFTAC Case031 (Backward-Facing Step, Le-Moin-Kim, Re_h = 5100)

## Source & license
- **Dataset**: ERCOFTAC Classic Collection, **Case031** — "Backward-Facing Step, DNS by Le & Moin".
- **Page**: http://cfd.mace.manchester.ac.uk/ercoftac/doku.php?id=cases:case031
- **Primary paper**: Le, H.; Moin, P.; Kim, J. (1997) *Direct numerical simulation of turbulent flow over a backward-facing step.* J. Fluid Mech. 330, 349–374. https://doi.org/10.1017/S0022112096003941
- **License**: CC BY-NC-SA 4.0 (attribution, **non-commercial**, share-alike). Cite Le-Moin-Kim and ERCOFTAC.

## Flow parameters
- Re_h = U₀·h/ν = **5100** (h = step height, U₀ = inlet free-stream velocity).
- Expansion ratio **1.2** (inlet height 5h, outlet height 6h).
- Mean reattachment **x_R ≈ 6.28h** (page states ~6h; paper 6.28h).
- Domain: 10h inlet section + 20h post-expansion; spanwise 4h (periodic). Grid 770×194×66.

## What Pioch et al. (2023) used from this dataset
Only the **mean velocities U, V** at six wall-normal lines (as boundary + interior
data loss). They did **not** prescribe k or ω. They also **shifted the x-coordinate**:

> **x_Pioch = x_ERCOFTAC + 3**

So Pioch's six lines map 1:1 onto the six Case031 stations:

| ERCOFTAC x/h | Pioch x/h | file | role in PINN |
|---:|---:|---|---|
| −2.99 | 0.00 | `x-181.dat` | inlet |
| 4.00 | 7.00 | `x-360.dat` | interior line |
| 6.00 | 9.00 | `x-411.dat` | interior line |
| 9.98 | 12.98 | `x-513.dat` | interior line |
| 14.98 | 17.98 | `x-641.dat` | interior line |
| 19.00 | 22.00 | `x-744.dat` | outlet |

## File format (`x-*.dat`)
Header lines start with `#`. Data columns (all non-dimensional, scaled by U₀, h):
```
y/h    U/Uo    V/Uo    u'/Uo    v'/Uo    w'/Uo    u'v'/Uo^2
```
Note `u', v', w'` are **RMS** values → `<u'u'> = u'^2`, etc. Therefore the full
turbulent kinetic energy is recoverable:
```
k = 0.5 (u'^2 + v'^2 + w'^2)
```
(All three normal stresses are present, so k needs no approximation.)

`stat-inf.dat` holds integral/wall quantities per station: δ₉₉, δ*, θ, H, G,
U_e, **U_tau**, **Cf**, Cp, Re_δ*, Re_θ, Re_τ. (Cf < 0 marks the recirculation
zone → reattachment is where Cf crosses 0, between x/h=6 and 9.98.)

## Files in this folder
- `download_case031.py` — fetches the full dataset into `raw/case031/`
  (`python download_case031.py`; add `--full` for the 30 budget files).
- `load_dns.py` — parser: reads `x-*.dat`, derives k, ν_t, ω; maps ERCOFTAC↔Pioch x/h.
- `raw/case031/` — downloaded `.dat` files (6 profiles + stat-inf provided directly;
  budgets via `--full`).
- `processed/` — destination for derived/cached arrays (npz) built from the raw files.

## Deriving k, ω boundary values (for the k-ω PINN)
- **k**: directly, `k = 0.5(u'^2+v'^2+w'^2)`.
- **ω** (no ε in the profile files): `ω = k/ν_t` with `ν_t = −u'v' / (dU/dy)`
  (implemented in `load_dns.py`). For a more accurate ε-based `ω = ε/(β*·k)`,
  download the k-budget files `rskk-*.dat` (`--full`) which contain the dissipation term.

## Reproduce
```
python download_case031.py        # populate raw/case031
python load_dns.py                # sanity-print all 6 stations
```
