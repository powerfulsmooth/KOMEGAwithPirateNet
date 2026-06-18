#!/usr/bin/env python3
"""
Parser for ERCOFTAC Case031 (Le-Moin-Kim BFS, Re_h=5100) profile files.

Each x-<idx>.dat file holds a wall-normal profile at one station:
    columns: y/h  U/Uo  V/Uo  u'/Uo  v'/Uo  w'/Uo  u'v'/Uo^2
where u',v',w' are RMS values (so <u'u'> = u'^2, etc.).

Derived quantities (non-dimensional, scaled by Uo and h):
    uu = u'^2 ,  vv = v'^2 ,  ww = w'^2 ,  uv = u'v'  (given)
    k  = 0.5 (uu + vv + ww)                              # turbulent KE
    nu_t = -uv / (dU/dy)                                  # eddy visc. estimate
    omega = k / nu_t                                      # specific dissipation

Coordinate note: Pioch et al. (2023) shifted x so that  x_Pioch = x_ERCOFTAC + 3.
This module reports both.
"""
import os
import glob
import numpy as np

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw", "case031")

# ERCOFTAC streamwise index -> (ERCOFTAC x/h, Pioch x/h)
STATION_XH = {
    181: (-2.99, 0.00),
    360: (4.00, 7.00),
    411: (6.00, 9.00),
    513: (9.98, 12.98),
    641: (14.98, 17.98),
    744: (19.00, 22.00),
}


def _read_profile(path):
    """Return dict with raw + derived fields for one station file."""
    y, U, V, up, vp, wp, uv = [], [], [], [], [], [], []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            c = [float(t) for t in s.split()]
            if len(c) < 7:
                continue
            y.append(c[0]); U.append(c[1]); V.append(c[2])
            up.append(c[3]); vp.append(c[4]); wp.append(c[5]); uv.append(c[6])
    y = np.array(y); U = np.array(U); V = np.array(V)
    up = np.array(up); vp = np.array(vp); wp = np.array(wp); uv = np.array(uv)
    uu, vv, ww = up**2, vp**2, wp**2
    k = 0.5 * (uu + vv + ww)
    # eddy viscosity from shear stress / mean shear (guard div-by-zero)
    dUdy = np.gradient(U, y)
    with np.errstate(divide="ignore", invalid="ignore"):
        nu_t = np.where(np.abs(dUdy) > 1e-8, -uv / dUdy, np.nan)
        omega = np.where((nu_t > 1e-12) & np.isfinite(nu_t), k / nu_t, np.nan)
    return dict(y=y, U=U, V=V, uu=uu, vv=vv, ww=ww, uv=uv,
                k=k, nu_t=nu_t, omega=omega)


def load_dns(raw_dir=RAW):
    """Load all 6 stations. Returns {pioch_xh: {...fields...}}."""
    out = {}
    for path in sorted(glob.glob(os.path.join(raw_dir, "x-*.dat"))):
        idx = int(os.path.basename(path).split("-")[1].split(".")[0])
        if idx not in STATION_XH:
            continue
        erc_xh, pioch_xh = STATION_XH[idx]
        d = _read_profile(path)
        d["x_ercoftac"] = erc_xh
        d["x_pioch"] = pioch_xh
        out[pioch_xh] = d
    return out


if __name__ == "__main__":
    data = load_dns()
    print(f"loaded {len(data)} stations (Pioch x/h):", sorted(data))
    for xh in sorted(data):
        d = data[xh]
        print(f"  x/h(Pioch)={xh:6.2f}  npts={len(d['y']):3d}  "
              f"k_max={d['k'].max():.4e}  U_min={d['U'].min():+.3f}")
