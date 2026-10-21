import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import predict_lib as P

st.set_page_config(page_title="UK carbon forecast", layout="wide")
st.title("UK grid carbon intensity: next 24 hours")
st.caption("Forecast in gCO2/kWh with a calibrated 80% interval. "
           "Quantile gradient boosting + conformal calibration. Data: Carbon Intensity API.")


@st.cache_resource
def load():
    return P.load_artifacts("deploy")


@st.cache_data(ttl=600)
def get_forecast():
    models, feats, offsets = load()
    return P.latest_forecast(models, feats, offsets)


try:
    fc, origin = get_forecast()
except Exception as e:
    st.error(f"Could not get live data right now ({e}). Please try again in a few minutes.")
    st.stop()

now = pd.Timestamp.now(tz="UTC").floor("30min")
fc = fc[fc.target_ts >= now].reset_index(drop=True)          # only future half-hours

hours = st.sidebar.slider("Charging time needed (hours)", 1, 8, 3)
deadline = st.sidebar.slider("Must finish within (hours from now)", 4, 24, 13)
w = P.best_window(fc, hours, deadline)

if w is None:
    st.warning("Window too short for that charging time.")
else:
    loc = lambda t: t.tz_convert("Europe/London")
    c1, c2, c3 = st.columns(3)
    c1.metric("Best start (UK time)", loc(w["start"]).strftime("%a %H:%M"))
    c2.metric("Forecast average in that window", f"{w['avg']:.0f} gCO2/kWh")
    c3.metric("If you start now", f"{w['now_avg']:.0f} gCO2/kWh",
              delta=f"{w['avg'] - w['now_avg']:.0f}", delta_color="inverse")

    fig, ax = plt.subplots(figsize=(10, 4))
    x = fc.target_ts.dt.tz_convert("Europe/London")
    ax.fill_between(x, fc.lo_cqr, fc.hi_cqr, alpha=0.25, label="80% calibrated interval")
    ax.plot(x, fc.med, label="forecast")
    ax.axvspan(loc(w["start"]), loc(w["end"]), alpha=0.2, color="green", label="best window")
    ax.set_ylabel("gCO2/kWh"); ax.legend(); fig.autofmt_xdate()
    st.pyplot(fig)

with st.expander("How reliable is this?"):
    st.markdown("""
Tested on Jul 2025 - Oct 2026 data the model never saw:
- the 80% intervals contained the true value **80.0%** of the time;
- for an evening plug-in (7pm to 8am window), picking the charging window with this forecast beat a "same time yesterday" forecast by about **4 gCO2/kWh** (3.8 to 4.8 depending on charging time);
- within about the next hour, simply repeating the latest value is more accurate than the model;
- uses *average* grid intensity, not *marginal* (what your extra load actually causes).
""")
st.caption(f"Latest real observation used: {origin:%Y-%m-%d %H:%M} UTC")
