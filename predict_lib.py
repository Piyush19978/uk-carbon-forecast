"""Live inference: fetch -> clean -> features -> forecast. Same rules as training."""
import json
import joblib
import numpy as np
import pandas as pd
import requests

BASE = "https://api.carbonintensity.org.uk"
MIX = ["wind", "gas", "solar", "imports", "nuclear", "biomass", "coal", "hydro"]
MAX_FILL = 6
BUCKETS = [(1, 4), (5, 12), (13, 24), (25, 36), (37, 48)]


def bucket(h):
    return next(i for i, (a, b) in enumerate(BUCKETS) if a <= h <= b)


def load_artifacts(folder="deploy"):
    models = {k: joblib.load(f"{folder}/model_{k}.joblib") for k in ("lo", "med", "hi")}
    meta = json.load(open(f"{folder}/meta.json"))
    offsets = {int(k): v for k, v in meta["cqr_offset_by_bucket"].items()}
    return models, meta["features"], offsets


def fill_short_gaps(s, max_steps):
    isna = s.isna()
    run_len = isna.groupby((isna != isna.shift()).cumsum()).transform("size")
    return s.interpolate(limit_area="inside").mask(isna & (run_len > max_steps))


def fetch_recent(days=8):
    end = pd.Timestamp.now(tz="UTC").floor("30min")
    start = end - pd.Timedelta(days=days)
    f = lambda t: t.strftime("%Y-%m-%dT%H:%MZ")
    ci = requests.get(f"{BASE}/intensity/{f(start)}/{f(end)}", timeout=60).json()["data"]
    gm = requests.get(f"{BASE}/generation/{f(start)}/{f(end)}", timeout=60).json()["data"]
    act = pd.Series({pd.Timestamp(x["from"]): x["intensity"]["actual"] for x in ci}, dtype=float).sort_index()
    mix = pd.DataFrame({pd.Timestamp(x["from"]): {m["fuel"]: m["perc"] for m in x["generationmix"]}
                        for x in gm}).T.sort_index()
    return act, mix


def build_hist(act, mix):
    grid = pd.date_range(act.index.min(), act.index.max(), freq="30min")
    raw = act.reindex(grid)
    raw = raw.mask(raw == 0)
    hist = pd.DataFrame(index=grid)
    hist["actual_filled"] = fill_short_gaps(raw, MAX_FILL)
    m = mix.reindex(grid).apply(lambda c: fill_short_gaps(c, MAX_FILL))
    for f in MIX:
        hist[f"mix_{f}"] = m[f]
    return hist


def make_features(hist, origin):
    idx = pd.date_range(end=origin, periods=337, freq="30min")
    s = hist["actual_filled"].reindex(idx)
    now = hist.loc[origin]
    w24, w7 = s.iloc[-48:], s.iloc[-336:]
    base = {
        "y_now": s.iloc[-1], "y_lag_1h": s.iloc[-3], "y_lag_3h": s.iloc[-7],
        "y_lag_6h": s.iloc[-13], "y_lag_24h": s.iloc[-49],
        "roll_mean_24h": w24.mean() if w24.notna().sum() >= 40 else np.nan,
        "roll_std_24h": w24.std() if w24.notna().sum() >= 40 else np.nan,
        "roll_mean_7d": w7.mean() if w7.notna().sum() >= 280 else np.nan,
    }
    for f in MIX:
        base[f"mix_{f}_now"] = now[f"mix_{f}"]
    rows = []
    for h in range(1, 49):
        t = origin + pd.Timedelta(minutes=30 * h)
        lt = t.tz_convert("Europe/London")
        rows.append(dict(base, h=h, target_ts=t,
                         y_target_minus_1d=s.iloc[288 + h], y_target_minus_7d=s.iloc[h],
                         hod=lt.hour + lt.minute / 60, dow=lt.dayofweek, is_weekend=int(lt.dayofweek >= 5),
                         doy_sin=np.sin(2 * np.pi * lt.dayofyear / 365.25),
                         doy_cos=np.cos(2 * np.pi * lt.dayofyear / 365.25)))
    return pd.DataFrame(rows)


def predict(hist, origin, models, feats, offsets):
    X = make_features(hist, origin)
    if pd.isna(X.y_now.iloc[0]):
        raise ValueError("no recent actual intensity: cannot forecast")
    out = X[["target_ts", "h"]].copy()
    for k in ("lo", "med", "hi"):
        out[k] = models[k].predict(X[feats]) + X["y_now"]
    out["lo"], out["hi"] = np.minimum(out.lo, out.med), np.maximum(out.hi, out.med)
    off = out.h.map(bucket).map(offsets)
    out["lo_cqr"] = np.minimum(out.lo - off, out.med)
    out["hi_cqr"] = np.maximum(out.hi + off, out.med)
    return out


def latest_forecast(models, feats, offsets):
    act, mix = fetch_recent()
    hist = build_hist(act, mix)
    valid = hist.actual_filled.notna() & hist.mix_gas.notna()
    origin = valid[valid].index.max()
    return predict(hist, origin, models, feats, offsets), origin


def best_window(fc, hours, deadline_hours):
    """Cheapest contiguous block (by median forecast) that ends within the deadline."""
    f = fc.iloc[: int(deadline_hours * 2)].reset_index(drop=True)
    L = int(hours * 2)
    if len(f) < L:
        return None
    m = f.med.to_numpy()
    cs = np.r_[0, np.cumsum(m)]
    means = (cs[L:] - cs[:-L]) / L
    s = int(means.argmin())
    return dict(start=f.target_ts.iloc[s], end=f.target_ts.iloc[s + L - 1] + pd.Timedelta(minutes=30),
                avg=float(means[s]), now_avg=float(means[0]))
