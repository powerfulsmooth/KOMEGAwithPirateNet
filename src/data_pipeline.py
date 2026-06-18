"""DNS data -> Pioch frame -> boundary/data points.

Uses data/load_dns.py (ERCOFTAC Case031). Frame transform:
  x_pioch  = x_ercoftac + 3   (already applied in load_dns STATION_XH)
  y_pioch  = y_data           (downstream stations)
  y_pioch  = y_data + h       (INLET station only; h = step height = 1)

Pioch uses U, V only. Returns coordinate+value arrays for inlet / outlet / interior lines.
"""
import importlib.util
import os
import numpy as np


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_dns_module():
    path = os.path.join(_repo_root(), "data", "load_dns.py")
    spec = importlib.util.spec_from_file_location("load_dns", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _station_points(station, xh, y_shift=0.0):
    y = np.asarray(station["y"], np.float32) + np.float32(y_shift)
    x = np.full_like(y, np.float32(xh))
    xy = np.stack([x, y], axis=1)
    uv = np.stack([np.asarray(station["U"], np.float32),
                   np.asarray(station["V"], np.float32)], axis=1)
    return xy.astype(np.float32), uv.astype(np.float32)


def build_data_points(cfg):
    """Return dict with inlet/outlet (always) and interior (per data.mode).

    Each entry is (xy, uv) with xy=(N,2), uv=(N,2) = [U, V].
    Interior is None when mode == "none".
    """
    m = _load_dns_module()
    raw_dir = cfg.data.raw_dir
    if not os.path.isabs(raw_dir):
        raw_dir = os.path.join(_repo_root(), raw_dir)
    stations = m.load_dns(raw_dir)            # keyed by Pioch x/h
    h = float(cfg.geom.step_y[1] - cfg.geom.step_y[0])   # step height = 1

    inlet_xh = float(cfg.data.inlet_xh)
    outlet_xh = float(cfg.data.outlet_xh)

    inlet = _station_points(stations[inlet_xh], inlet_xh, y_shift=h)      # +h shift
    outlet = _station_points(stations[outlet_xh], outlet_xh, y_shift=0.0)

    mode = cfg.data.mode
    if mode == "none":
        lines = []
    elif mode == "lines3":
        lines = list(cfg.data.lines3)
    else:  # "full"
        lines = list(cfg.data.full)

    interior = None
    if lines:
        xs, vs = [], []
        for xh in lines:
            xy, uv = _station_points(stations[float(xh)], float(xh), y_shift=0.0)
            xs.append(xy)
            vs.append(uv)
        interior = (np.concatenate(xs, 0), np.concatenate(vs, 0))

    return {"inlet": inlet, "outlet": outlet, "interior": interior}
