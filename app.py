"""BreathBreak — personal-baseline fatigue check (web app).
Run:  python3 -m streamlit run app.py
  ① Input baseline   : upload a rested recording once per person -> saved as their baseline
  ② Input recording  : upload a new recording -> compared with the baseline -> FATIGUE / NORMAL
  History            : past checks per person
  HR reports         : fatigue alerts sent to HR
  Model              : how the model was trained, honest held-out results, retrain button
"""
import hashlib
import json
from datetime import datetime
import base64
from contextlib import contextmanager
import gzip
from pathlib import PurePosixPath
import re
import shlex
import threading
import time
import uuid

import serial

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fatigue_pipeline import (FS_ECG, LABELS, MODEL_PATH, PRE, ROOT, DeltaModel, analyze,
                              beat_similarity, load_recording, quality_warnings)
from hr_report import REPORTS, SEVERITY_BANDS, build_report, load_reports, report_text, send_report
USERS = ROOT / 'user_data'
CONTRIB = ROOT / 'data' / 'contributed'
MANIFEST = ROOT / 'data_manifest.csv'
st.set_page_config(page_title='BreathBreak — Fatigue Check', page_icon='🫀', layout='wide')
# Palette shared with the slides and the report (report/breathbreak_report.tex)
NAVY, TEAL, CORAL, AMBER, LIGHT, GREY, GREEN, PURPLE = (
    '#12294A', '#0E8A8F', '#F2685B', '#F5A623', '#E8F4F4', '#5B6573', '#3DDC97', '#7C5CFC')
# ---------------- look & feel ----------------
def ecg_points(width=1400, height=70, beats=7):
    """Polyline for the decorative ECG trace in the header (same motif as the title slide)."""
    mid, pts, step = height * .62, [], width / beats
    for i in range(beats):
        x = i * step
        for dx, dy in ((0, 0), (.30, 0), (.34, -4), (.38, 0), (.44, 0), (.46, 5), (.49, -mid + 4),
                       (.52, height - mid - 2), (.55, 0), (.63, 0), (.68, -7), (.74, 0)):
            pts.append(f'{x + dx * step:.0f},{mid + dy:.0f}')
    pts.append(f'{width},{mid:.0f}')
    return ' '.join(pts)
CSS = f"""
<style>
:root {{ --navy:{NAVY}; --teal:{TEAL}; --coral:{CORAL}; --amber:{AMBER}; --light:{LIGHT}; --grey:{GREY}; --purple:{PURPLE}; }}
.block-container {{ padding-top: 3.6rem; max-width: 1240px; }}
h1, h2, h3 {{ color: var(--navy); }}
/* header (title slide) */
.bb-hero {{ position: relative; overflow: hidden; border-radius: 18px; padding: 24px 32px 70px;
  background: linear-gradient(120deg, #12294A 0%, #0F5468 55%, #0E8A8F 100%); color: #fff; margin-bottom: 8px; }}
.bb-hero .kicker {{ color: var(--amber); font-weight: 700; letter-spacing: .07em; font-size: .78rem; }}
.bb-hero .title {{ color: #fff; font-size: 2.5rem; font-weight: 800; line-height: 1.1; margin: 2px 0 6px; }}
.bb-hero .sub {{ color: #d5eaec; font-size: 1.02rem; max-width: 600px; }}
.bb-hero .orb {{ position: absolute; border-radius: 50%; background: rgba(255,255,255,.06); }}
.bb-hero svg {{ position: absolute; left: 0; bottom: 6px; width: 100%; height: 70px; }}
.bb-hero .page {{ position: absolute; right: 30px; top: 26px; background: rgba(255,255,255,.12);
  border: 1px solid rgba(255,255,255,.25); border-radius: 999px; padding: 6px 16px; font-weight: 700; font-size: .9rem; }}
/* section title bar (slide titles) */
.bb-sec {{ position: relative; display: flex; align-items: center; gap: 12px; margin: 26px 0 14px; padding: 10px 18px;
  border-radius: 10px; background: linear-gradient(90deg, #12294A, #0E8A8F); color: #fff; }}
.bb-sec:after {{ content: ''; position: absolute; left: 0; bottom: -5px; width: 150px; height: 5px; border-radius: 3px;
  background: linear-gradient(90deg, var(--amber) 0 60%, var(--coral) 60% 100%); }}
.bb-sec .num {{ background: #fff; color: var(--teal); border-radius: 50%; min-width: 30px; height: 30px; display: inline-flex;
  align-items: center; justify-content: center; font-weight: 800; }}
.bb-sec .t {{ font-weight: 800; font-size: 1.2rem; }}
.bb-sec .s {{ color: #cfe7e9; font-size: .92rem; margin-left: auto; }}
/* cards (slide boxes with coloured top border) */
.bb-card {{ background: #fff; border: 1px solid #e2eaee; border-top: 4px solid var(--teal); border-radius: 12px;
  padding: 14px 18px; box-shadow: 0 2px 10px rgba(18,41,74,.06); margin-bottom: 10px; color: var(--navy); }}
.bb-card h4 {{ margin: 0 0 6px; padding: 0; color: var(--teal); font-size: 1.05rem; font-weight: 800; }}
.bb-card ol, .bb-card ul {{ margin: 0; padding-left: 1.2rem; }} .bb-card li {{ margin: 3px 0; }}
.bb-card.coral {{ border-top-color: var(--coral); background: #fff5f4; }} .bb-card.coral h4 {{ color: var(--coral); }}
.bb-card.amber {{ border-top-color: var(--amber); }} .bb-card.amber h4 {{ color: #c98200; }}
.bb-card.purple {{ border-top-color: var(--purple); }} .bb-card.purple h4 {{ color: var(--purple); }}
.bb-card.navy {{ border-top-color: var(--navy); }} .bb-card.navy h4 {{ color: var(--navy); }}
.bb-muted {{ color: var(--grey); font-size: .88rem; }}
/* stat tiles */
.bb-stat {{ background: var(--light); border-radius: 12px; padding: 12px 16px; height: 100%; }}
.bb-stat .lbl {{ color: var(--grey); font-size: .74rem; text-transform: uppercase; letter-spacing: .06em; font-weight: 700; }}
.bb-stat .val {{ color: var(--navy); font-size: 1.65rem; font-weight: 800; line-height: 1.25; }}
.bb-stat .dlt {{ font-size: .84rem; color: var(--grey); }}
/* verdict (watch face on the slides) */
.bb-verdict {{ background: var(--navy); border-radius: 22px; border: 7px solid #dfe9ec; padding: 16px 14px 18px;
  text-align: center; color: #fff; box-shadow: 0 6px 18px rgba(18,41,74,.18); }}
.bb-verdict .lv {{ font-size: 1.9rem; font-weight: 800; letter-spacing: .04em; }}
.bb-verdict svg {{ width: 100%; height: 38px; margin: 4px 0 6px; }}
.bb-verdict .pill {{ display: inline-block; border-radius: 999px; padding: 6px 18px; font-weight: 800; color: var(--navy); }}
.bb-verdict .p {{ color: #b9cfd6; font-size: .84rem; margin-top: 8px; }}
/* sidebar */
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #12294A 0%, #0F4D63 100%); }}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {{ color: #e6f2f3; }}
.bb-side-brand {{ font-size: 1.6rem; font-weight: 800; color: #fff; }}
.bb-side-kicker {{ color: var(--amber) !important; font-weight: 700; font-size: .72rem; letter-spacing: .07em; }}
.bb-side-box {{ background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.15); border-radius: 10px;
  padding: 10px 12px; font-size: .86rem; margin-top: 10px; }}
.bb-side-box b {{ color: #fff; }}
/* buttons */
[data-testid="stSidebar"] :is(.st-key-edge_connect button, button[kind="primary"]) {{
  background: #08767A !important; border: 1px solid #68C1C2 !important;
  color: #FFFFFF !important; font-weight: 750; width: 100%;
}}
[data-testid="stSidebar"] :is(.st-key-edge_connect button, button[kind="primary"]) :is(p,span,div) {{ color: #FFFFFF !important; }}
[data-testid="stSidebar"] :is(.st-key-edge_connect button, button[kind="primary"]):hover {{ background: #065B61 !important; }}
[data-testid="stSidebar"] :is(.st-key-edge_connect button, button[kind="primary"]):focus-visible {{
  outline: 3px solid #FFC75D !important; outline-offset: 2px;
}}
.stButton > button[kind="primary"], .stDownloadButton > button {{ border-radius: 999px; font-weight: 700; }}
.bb-foot {{ margin-top: 40px; padding-top: 10px; border-top: 1px solid #dde6ea; color: var(--grey); font-size: .82rem;
  display: flex; justify-content: space-between; }}
.bb-foot b {{ color: var(--teal); }}
</style>
"""
def html(s):
    st.markdown(s, unsafe_allow_html=True)
def hero(page_label):
    html(f"<div class='bb-hero'><div class='orb' style='width:340px;height:340px;right:-80px;top:-150px'></div>"
         f"<div class='orb' style='width:220px;height:220px;right:180px;bottom:-140px'></div>"
         f"<div class='page'>{page_label}</div>"
         f"<div class='kicker'>UTD HACKEDGE 2026 &nbsp;•&nbsp; EDGE AI TRACK</div>"
         f"<div class='title'>BreathBreak</div>"
         f"<div class='sub'>Edge fatigue detection from wearable ECG, compared with each worker's own rested baseline.</div>"
         f"<svg viewBox='0 0 1400 70' preserveAspectRatio='none'><polyline points='{ecg_points()}' fill='none' "
         f"stroke='{GREEN}' stroke-width='2.5' stroke-linejoin='round'/></svg></div>")
def section(title, sub='', num=''):
    n = f"<span class='num'>{num}</span>" if num else ''
    html(f"<div class='bb-sec'>{n}<span class='t'>{title}</span><span class='s'>{sub}</span></div>")
def card(title, body, kind=''):
    html(f"<div class='bb-card {kind}'><h4>{title}</h4>{body}</div>")
def stat(col, label, value, delta=''):
    col.markdown(f"<div class='bb-stat'><div class='lbl'>{label}</div><div class='val'>{value}</div>"
                 f"<div class='dlt'>{delta}</div></div>", unsafe_allow_html=True)
def verdict_card(col, fatigued, p, threshold):
    color = CORAL if fatigued else GREEN
    pts = '0,20 60,20 70,14 78,20 92,20 98,26 106,2 114,34 120,20 140,20 150,12 160,20 220,20'
    score = f'P(fatigue) = {p:.2f}' if p is not None else 'Probability not reported'
    if threshold is not None:
        score += f' · fatigue at ≥ {threshold:.2f}'
    col.markdown(
        f"<div class='bb-verdict'><div class='lv' style='color:{'#fff' if fatigued else GREEN}'>"
        f"{'FATIGUE' if fatigued else 'NORMAL'}</div>"
        f"<svg viewBox='0 0 220 38' preserveAspectRatio='none'><polyline points='{pts}' fill='none' stroke='{color}' "
        f"stroke-width='2.5'/></svg>"
        f"<span class='pill' style='background:{AMBER if fatigued else GREEN}'>{'Take a breath' if fatigued else 'Keep going'}</span>"
        f"<div class='p'>{score}</div></div>", unsafe_allow_html=True)
def style_fig(fig, height=300, title=None):
    fig.update_layout(height=height, title=dict(text=title, font=dict(size=15, color=NAVY)) if title else None,
                      font=dict(family='Source Sans Pro, sans-serif', color=GREY, size=12),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#ffffff',
                      margin=dict(l=10, r=10, t=40 if title else 10, b=10),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.update_xaxes(gridcolor='#edf2f4', zeroline=False, linecolor='#d5dfe3')
    fig.update_yaxes(gridcolor='#edf2f4', zeroline=False, linecolor='#d5dfe3')
    return fig
def footer():
    html("<div class='bb-foot'><span><b>BreathBreak</b> · UTD HackEdge 2026 · Track 1</span>"
         "<span>Nhat Truong · Dung Anh Nguyen · Anthony Tran</span></div>")
# ---------------- helpers ----------------
@st.cache_data(show_spinner='Preprocessing ECG (filter → artifact mask → R-peaks → features)…')
def run_analysis(data: bytes):
    ecg, opt = load_recording(data)
    return analyze(ecg, opt)
def clear_history():
    for h in USERS.glob('*/history.csv'):
        h.unlink(missing_ok=True)
def get_model():
    return DeltaModel(MODEL_PATH) if MODEL_PATH.exists() else None
def safe_id(s):
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in s.strip()) or 'user'
def users():
    return sorted(p.name for p in USERS.glob('*') if (p / 'baseline.json').exists()) if USERS.exists() else []
def load_baseline(uid):
    return json.loads((USERS / uid / 'baseline.json').read_text())
def samples():
    out = {}
    for folder in ('normal', 'fatigue'):
        for p in sorted((ROOT / 'data' / folder).glob('*.csv')):
            out[f'{folder}/{p.name}'] = p
    return out
def recording_input(key, action='▶ Use this recording'):
    """Choose a recording (upload or sample). Nothing is analyzed or saved until the button is clicked."""
    up = st.file_uploader('TI AFE4950 export (.csv)', type='csv', key=f'up_{key}')
    opts = samples()
    pick = st.selectbox('…or use a sample recording', ['—'] + list(opts), key=f'sample_{key}',
                        disabled=up is not None)
    if up is not None:
        data, name = up.getvalue(), up.name
    elif pick != '—':
        data, name = opts[pick].read_bytes(), pick
    else:
        st.session_state.pop(f'use_{key}', None)
        return None, None
    sig = (name, len(data), hashlib.md5(data).hexdigest())
    if st.session_state.get(f'use_{key}') != sig:
        st.caption(f'Selected: **{name}**')
        if st.button(action, type='primary', key=f'btn_{key}'):
            st.session_state[f'use_{key}'] = sig
            st.rerun()
        return None, None
    c1, c2 = st.columns([4, 1])
    c1.caption(f'Using: **{name}**')
    if c2.button('✖ Clear', key=f'clear_{key}'):
        st.session_state.pop(f'use_{key}', None)
        st.rerun()
    return data, name
def show_warnings(res, role):
    for level, msg in quality_warnings(res, role):
        (st.error if level == 'error' else st.warning)(msg)
def ecg_figure(res, title, seconds=10):
    t, x, rp = res['t'], res['ecg_clean'], res['rpeaks']
    a = res['keep'][0]; w = (t >= a) & (t <= a + seconds)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t[w], y=x[w], line=dict(width=1.3, color=TEAL), name='clean ECG'))
    r = rp[(t[rp] >= a) & (t[rp] <= a + seconds)]
    fig.add_trace(go.Scatter(x=t[r], y=x[r], mode='markers', marker=dict(color=AMBER, size=8), name='R-peak'))
    fig.update_layout(xaxis_title='time (s)', yaxis_title='mV', showlegend=False)
    return style_fig(fig, 230, title)
def overview_figure(res, title):
    t, x = res['t'], res['ecg_clean']
    step = 5
    fig = go.Figure(go.Scattergl(x=t[::step], y=x[::step], line=dict(width=.7, color=TEAL)))
    a, b = res['keep']
    fig.add_vrect(x0=0, x1=a, fillcolor=GREY, opacity=.15, line_width=0)
    fig.add_vrect(x0=b, x1=t[-1], fillcolor=GREY, opacity=.15, line_width=0)
    fig.update_layout(xaxis_title='time (s)', yaxis_title='mV')
    return style_fig(fig, 190, title)
def quality_stats(res):
    q, f = res['quality'], res['features'] or {}
    c = st.columns(5)
    stat(c[0], 'Recording', f"{q['duration_s']:.0f} s", f"stable part {q['stable_start']:.0f}–{q['stable_end']:.0f} s")
    stat(c[1], 'Clean beats', str(q['clean_beats']), f"{q['masked_pct']:.1f} % masked (motion)")
    stat(c[2], 'Heart rate', f"{f['HR']:.0f} bpm" if f else '–')
    stat(c[3], 'SDNN', f"{f['SDNN']:.0f} ms" if f else '–', 'overall HRV')
    stat(c[4], 'RMSSD', f"{f['RMSSD']:.0f} ms" if f else '–', 'beat-to-beat HRV')
# Model A (ours) vs. Model B (transfer learning), leave-one-recording-out on our 11 recordings.
# From report/breathbreak_report.tex, Table "tab:modelb" (produced by transfer_model_b.py; WESAD / DROZY not in the repo).
MODEL_COMPARISON = pd.DataFrame([
    ('Model A: ours only (logistic regression)', 'A', 11, 0.025, 0.06),
    ('Zero-shot: WESAD', 'B', 10, 0.101, -0.38),
    ('Zero-shot: DROZY', 'B', 9, 0.159, -0.46),
    ('Zero-shot: WESAD → DROZY', 'B', 9, 0.147, -0.47),
    ('Model B: WESAD → DROZY → ours', 'B', 10, 0.035, -0.04),
    ('WESAD → ours', 'B', 11, 0.028, 0.03),
    ('NN 3-8-1: WESAD → DROZY → ours (frozen)', 'B', 11, 0.017, 0.11),
    ('NN 3-8-1: ours only', 'A', 11, 0.020, 0.15),
], columns=['Model', 'Family', 'Correct', 'Brier', 'Worst margin'])
def model_comparison():
    section('Compare models', 'Model A (ours) vs. Model B (transfer learning)')
    c = st.columns(2, gap='large')
    with c[0]:
        card('Model A: personal shift (deployed)',
             '<ul><li>Logistic regression on the change in <b>HR, SDNN, RMSSD</b> from your own baseline</li>'
             '<li>4 parameters, trains on the SK-AM62B in ~10 s</li>'
             '<li><b>11/11 correct</b>, Brier 0.025</li></ul>')
    with c[1]:
        card('Model B: transfer learning',
             '<ul><li>Pre-trained on <b>WESAD</b> (stress), fine-tuned on <b>DROZY</b> (sleep deprivation), then on our data</li>'
             '<li>Needs public datasets and more compute</li>'
             '<li><b>10/11 correct</b>, Brier 0.035 (one rested recording flagged)</li></ul>', 'purple')
    d = MODEL_COMPARISON.sort_values('Brier', ascending=False)
    colors = [NAVY if m.startswith('Model A') else PURPLE if m.startswith('Model B') else (TEAL if n == 11 else CORAL)
              for m, n in zip(d.Model, d.Correct)]
    fig = go.Figure(go.Bar(x=d.Brier, y=d.Model, orientation='h', marker_color=colors,
                           text=[f'{n}/11 · {b:.3f}' for n, b in zip(d.Correct, d.Brier)], textposition='outside'))
    fig.update_traces(cliponaxis=False)
    fig.update_layout(xaxis=dict(title='Brier score (lower is better)', range=[0, .22]))
    g1, g2 = st.columns([1.4, 1], gap='large')
    g1.plotly_chart(style_fig(fig, 340, 'Leave-one-recording-out on our 11 recordings'), width='stretch')
    with g2:
        st.dataframe(MODEL_COMPARISON.drop(columns='Family').assign(Correct=lambda x: x.Correct.astype(str) + '/11'),
                     hide_index=True, width='stretch', height=320)
    st.caption('Teal = all 11 correct, coral = at least one mistake. Worst margin < 0 means a recording was misclassified. '
               'Transfer learning did not beat Model A on our data, so Model A runs on the board. '
               'Source: project report, "Does transfer learning help?" (transfer_model_b.py).')
def how_model_a_works(model):
    """Explain the logistic regression and the leave-one-out test with the numbers stored in the trained model."""
    if model is None:
        return
    m, ev = model.meta, model.meta.get('evaluation', {})
    w = dict(zip(m['features'], m['coef']))
    section('How Model A works', 'logistic regression + leave-one-out test')
    c = st.columns(2, gap='large')
    with c[0]:
        card('① Logistic regression',
             "For a new recording and the person's baseline, it measures the <b>percent change</b> (log ratio) of "
             "three features:<ul>"
             f"<li><b>Heart rate</b>: weight <b style='color:{CORAL}'>{w['HR']:+.2f}</b> (up → fatigue)</li>"
             f"<li><b>SDNN</b> (overall HRV): weight <b style='color:{TEAL}'>{w['SDNN']:+.2f}</b> (down → fatigue)</li>"
             f"<li><b>RMSSD</b> (beat-to-beat HRV): weight <b style='color:{TEAL}'>{w['RMSSD']:+.2f}</b> (down → fatigue)</li></ul>"
             "Each change is standardised, multiplied by its weight and summed. The sigmoid turns the sum into "
             f"P(fatigue) between 0 and 1. <b>FATIGUE when P ≥ {model.threshold:.2f}</b> (moderate scores count as normal, to avoid false alarms). "
             "4 parameters in total, so it trains on the board in seconds.")
        st.latex(r'P(\text{fatigue}) = \frac{1}{1 + e^{-(b + \sum_i w_i x_i)}}, \quad x_i = \log\frac{\text{new}_i}{\text{baseline}_i}')
    with c[1]:
        card('② Leave-one-out test',
             '<ol><li><b>Hide</b> one recording.</li>'
             '<li><b>Train</b> the model on all the others.</li>'
             "<li><b>Test</b> the hidden one against that person's rested baselines → P(fatigue).</li>"
             '<li><b>Repeat</b> until every recording has been hidden once.</li></ol>'
             'Every recording is scored by a model that <b>never saw it</b>, which is a fair test for a small dataset. '
             '<span class="bb-muted">Three sessions of A labelled normal but recorded at 105–109 bpm (not at rest) '
             'are excluded.</span>', 'navy')
    if not ev or 'per_recording' not in ev:
        return
    person = {d['new_recording']: d['person'] for d in ev['details']}
    r = pd.DataFrame(ev['per_recording'])
    r['Person'] = r.new_recording.map(person)
    r = r.sort_values(['true', 'Person', 'new_recording'], ascending=[False, True, True]).reset_index(drop=True)
    r['Recording'] = r.new_recording.str.replace('data/', '', regex=False)
    r['level'] = np.where(r.p_fatigue >= model.threshold, 'fatigue', 'normal')   # current threshold (0.8 floor)
    r['ok'] = r.level == r.true
    r['p_fatigue'] = r.p_fatigue.round(2)                # same rounding in the chart and the table
    k = st.columns(4)
    n_ok = int(r.ok.sum())
    stat(k[0], 'Accuracy', f'{n_ok}/{len(r)}', f'{n_ok / len(r):.0%} of held-out recordings')
    stat(k[1], 'Fatigue caught', f"{int(((r.true == 'fatigue') & r.ok).sum())}/{int((r.true == 'fatigue').sum())}")
    stat(k[2], 'Normal left alone', f"{int(((r.true == 'normal') & r.ok).sum())}/{int((r.true == 'normal').sum())}")
    stat(k[3], 'False alarms', int(((r.true == 'normal') & ~r.ok).sum()))
    st.write('')
    g1, g2 = st.columns([1.3, 1], gap='large')
    fig = go.Figure()
    for lab, colr in (('normal', TEAL), ('fatigue', CORAL)):
        s = r[r.true == lab]
        fig.add_trace(go.Bar(x=s.p_fatigue, y=s.Recording, orientation='h', name=f'true {lab}', marker_color=colr,
                             text=[f'{p:.2f}' for p in s.p_fatigue], textposition='outside'))
    fig.add_vline(x=model.threshold, line_dash='dash', line_color=NAVY,
                  annotation_text=f'FATIGUE if ≥ {model.threshold:.2f}', annotation_position='top')
    fig.update_traces(cliponaxis=False)
    fig.update_layout(xaxis=dict(range=[0, 1.08], title='P(fatigue) when the recording was hidden'),
                      yaxis=dict(autorange='reversed'), bargap=.35)
    style_fig(fig, 90 + 32 * len(r), 'Each recording, scored while hidden from training')
    fig.update_layout(legend=dict(orientation='h', yanchor='top', y=-0.2, xanchor='left', x=0), margin=dict(b=95))
    g1.plotly_chart(fig, width='stretch')
    t = pd.DataFrame({'#': range(1, len(r) + 1), 'Hidden recording': r.Recording, 'True': r.true,
                      'P': r.p_fatigue.round(2), 'OK': np.where(r.ok, '✅', '❌')})
    g2.dataframe(t, hide_index=True, width='stretch', height=40 + 35 * len(r))
def fatigue_types():
    """What kind of fatigue the personal-shift model separates well, from our own recordings."""
    section('What fatigue can we detect?', 'findings from our recordings')
    c = st.columns(2, gap='large')
    with c[0]:
        card('✅ Tired from work: clearly different from baseline (subject B)',
             '<ul><li>Healthy baseline: HR <b>69–88 bpm</b>, high HRV</li>'
             '<li>Tired from work or exertion: HR <b>87–106 bpm</b>, RMSSD down 50–75 %</li>'
             '<li>A big shift: fatigue scores <b>0.71–0.84</b> vs. rest <b>≤ 0.26</b>. <b>No false alarms</b>; '
             'an alert needs P ≥ 0.8, so the strongest sessions alert</li></ul>')
    with c[1]:
        card('⚠️ Sick + tired ≈ sick alone (subject A)',
             '<ul><li>Illness alone already raises HR and lowers HRV: healthy rest 65–69 bpm, sick "normal" '
             '<b>105–109 bpm</b></li>'
             '<li>Sick "normal" vs. sick and tired: HR <b>105 vs 100</b>, RMSSD <b>8 vs 12 ms</b>, '
             '<b>almost identical</b></li>'
             '<li>Against a sick baseline, being tired on top of the illness barely shows</li></ul>', 'coral')
    card('Takeaway',
         'Fatigue from work shows up <b>clearly</b> against a healthy baseline. When someone is sick, the illness '
         'already looks like fatigue, so being tired on top of it barely changes HR and HRV. '
         '<b>Record the baseline when healthy.</b> Next step: add breathing and PPG signals to separate illness '
         'from work fatigue.', 'amber')
def contribute(data, name, uid, condition, kss, note):
    d = CONTRIB / uid; d.mkdir(parents=True, exist_ok=True)
    fname = f"{datetime.now():%Y%m%d_%H%M%S}_{condition}.csv"
    (d / fname).write_bytes(data)
    row = pd.DataFrame([dict(file=str((d / fname).relative_to(ROOT)), person=uid, condition=condition,
                             kss=kss, notes=f'contributed via app; source {name}; {note}'.strip('; '))])
    row.to_csv(MANIFEST, mode='a', header=False, index=False)
    return d / fname

# ---------------- USB serial backend ----------------
BOARD_ROOT = "/root/hackedge"
BOARD_MODEL = "models/shift_model.json"
BOARD_RECORDING_FOLDERS = {"baseline": "data/baseline", "check": "data/test"}
PORT = "COM12"
BAUD = 115200
MAX_CSV_BYTES = 12 * 1024 * 1024
ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}")

LIST_SCRIPT = """
import json
from pathlib import Path
data = Path('data')
root = data.resolve()
files = sorted(str(p) for p in data.rglob('*')
               if p.is_file() and not p.is_symlink() and p.suffix.lower() == '.csv'
               and p.resolve().is_relative_to(root))
print(json.dumps(files))
""".strip()

UPLOAD_SCRIPT = """
import base64
import gzip
import hashlib
from pathlib import Path
import sys
wire, dest = map(Path, sys.argv[1:3])
content = gzip.decompress(base64.b64decode(wire.read_bytes(), validate=True))
if len(content) != int(sys.argv[3]) or hashlib.sha256(content).hexdigest() != sys.argv[4]:
    raise SystemExit('Upload checksum or size mismatch')
with dest.open('xb') as output:
    try:
        output.write(content)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
wire.unlink()
print('Upload verified on SK-AM62B:', dest)
""".strip()


class BoardError(RuntimeError):
    pass


def quote(value):
    """Quote an argument for the Linux shell behind the local serial console."""
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ValueError("Control characters are not supported in board arguments.")
    return shlex.quote(value)


@st.cache_resource
def port_lock():
    return threading.Lock()


@contextmanager
def board_session():
    # One caller at a time, even if the dashboard is open in several browser tabs.
    lock = port_lock()
    if not lock.acquire(blocking=False):
        raise BoardError("Another dashboard request is using COM12. Wait for it to finish.")
    try:
        with serial.Serial(PORT, BAUD, timeout=0.2, write_timeout=5) as connection:
            connection.reset_input_buffer()
            yield connection
    except serial.SerialException as exc:
        raise BoardError(f"{PORT} could not be used. Close Miniterm and check the debug cable.\n{exc}") from exc
    finally:
        lock.release()


def board_command(connection, command, timeout=25):
    """Run a predefined app operation and collect only its framed output."""
    token = uuid.uuid4().hex
    start, end = f"BB_{token}_START", f"BB_{token}_END"
    wrapped = (
        f"printf '\\n%s\\n' {start}; "
        f"if cd {quote(BOARD_ROOT)}; then ( {command} ); rc=$?; else rc=$?; fi; "
        f"printf '\\n%s:%s\\n' {end} \"$rc\"\n"
    ).encode("utf-8")
    if len(wrapped) > 3500:
        raise BoardError("The requested operation exceeds the serial-console line length.")
    connection.write(wrapped)
    deadline = time.monotonic() + timeout
    pending = bytearray()
    lines = []
    started = False
    tail = ""
    while time.monotonic() < deadline:
        chunk = connection.read(connection.in_waiting or 1)
        if not chunk:
            continue
        pending.extend(chunk)
        while (i := pending.find(b"\n")) >= 0:
            line = bytes(pending[:i]).rstrip(b"\r").decode("utf-8", errors="replace")
            del pending[:i + 1]
            if line.strip() == start:
                started = True
                continue
            if not started:
                tail = (tail + "\n" + line)[-500:]
                continue
            match = re.fullmatch(re.escape(end) + r":(\d+)", line.strip())
            if match:
                output = "\n".join(lines).strip()
                if match[1] != "0":
                    raise BoardError(f"Board command exited with code {match[1]}.\n{output}")
                return output
            lines.append(line)
    if not started:
        context = (tail + pending.decode("utf-8", errors="replace"))[-500:].strip()
        raise BoardError(
            "No board command response. Open Miniterm, log into Linux, leave it "
            "at the shell prompt, exit Miniterm with Ctrl+], then try again."
            + (f"\nConsole output:\n{context}" if context else "")
        )
    raise BoardError(
        "The board did not finish in time and may still be working. "
        "Check the console before trying again.\n" + "\n".join(lines[-15:])
    )


def python_command(source, *args):
    # Encode the fixed multiline script so the serial console receives one line.
    payload = base64.b64encode(source.encode()).decode("ascii")
    inline = f"import base64;exec(base64.b64decode('{payload}'))"
    return " ".join(["python3", "-c", quote(inline), *(quote(arg) for arg in args)])


def inventory():
    with board_session() as conn:
        device = board_command(conn, "hostname && uname -m && test -f board_shift.py")
        model_info = board_command(
            conn, f"if test -s {BOARD_MODEL}; then ls -lh {BOARD_MODEL} && sha256sum {BOARD_MODEL}; "
            "else printf '%s\\n' 'No saved board model. Train on the board first.'; fi"
        )
        people = board_command(conn, "python3 -u board_shift.py people")
        raw = board_command(conn, python_command(LIST_SCRIPT))
    paths = json.loads(raw.splitlines()[-1])
    paths = [
        path for path in paths
        if not any(ord(ch) < 32 or ord(ch) == 127 for ch in path)
    ]
    return {"device": device, "model": model_info, "people": people, "paths": paths}


def shift_operation(action, uid, path):
    if action not in ("enroll", "check") or not ID_PATTERN.fullmatch(uid):
        raise ValueError("Use an ID that starts with a letter or digit and has no more than 40 characters (letters, digits, underscores, hyphens).")
    recording = PurePosixPath(path)
    role = "baseline" if action == "enroll" else "check"
    folder = PurePosixPath(BOARD_RECORDING_FOLDERS[role])
    if (not recording.is_relative_to(folder) or ".." in recording.parts
            or recording.suffix.lower() != ".csv"):
        raise ValueError(f"Select a board CSV under {folder}/.")
    args = f"{action} {quote(uid)} {quote(path)}"
    verify = f"test -f {quote(path)} && "
    if action == "check":
        verify += f"sha256sum {BOARD_MODEL} {quote(path)} && "
    with board_session() as conn:
        return board_command(conn, verify + f"python3 -u board_shift.py {args}", timeout=180)


def train_on_board():
    with board_session() as conn:
        return board_command(
            conn,
            f"python3 -u board_shift.py train && test -s {BOARD_MODEL} && "
            f"ls -lh {BOARD_MODEL} && sha256sum {BOARD_MODEL}",
            timeout=900,
        )


def send_csv_bytes(conn, content, target, progress):
    wire = target + '.part'
    folder = str(PurePosixPath(target).parent)
    payload = base64.b64encode(gzip.compress(content, compresslevel=3)).decode("ascii")
    digest = hashlib.sha256(content).hexdigest()
    block = 2048
    board_command(conn, f"mkdir -p {quote(folder)} && : > {quote(wire)}")
    for offset in range(0, len(payload), block):
        piece = payload[offset:offset + block]
        board_command(conn, f"printf %s '{piece}' >> {quote(wire)}")
        if offset % (block * 8) == 0:
            progress.progress(min(offset + len(piece), len(payload)) / len(payload))
    board_command(conn, python_command(UPLOAD_SCRIPT, wire, target, str(len(content)), digest), timeout=60)
    progress.progress(1.0)


def upload_csv(name, content, progress, *, role):
    if not name.lower().endswith(".csv") or not 0 < len(content) <= MAX_CSV_BYTES:
        raise ValueError("Choose a nonempty CSV no larger than 12 MiB.")
    tag = uuid.uuid4().hex
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    safe_stem = re.sub(r"[^A-Za-z0-9_-]", "_", name[:-4])[:48] or "recording"
    folder = f"{BOARD_RECORDING_FOLDERS[role]}/uploads"
    target = f"{folder}/{safe_stem}_{tag[:12]}.csv"
    with board_session() as conn:
        send_csv_bytes(conn, content, target, progress)
    return target


# ---------------- board data and pages ----------------
BOARD_PREVIEW_SCRIPT = """
import csv
import json
import math
import sys

channels = {
    'ECG': ('ECG_Time', 'ECG_Value', 12),
    'PPG': ('TIA1-3_Time', 'TIA1-3_Value', 2),
}
points = {name: [] for name in channels}
starts, counts, previous = {}, {}, {}
with open(sys.argv[1], encoding='utf-8-sig', newline='') as source:
    rows = csv.DictReader(source)
    for i, row in enumerate(rows):
        if i >= 15000:
            break
        for name, (time_key, signal_key, stride) in channels.items():
            try:
                t = float(row.get(time_key) or '')
                y = float(row.get(signal_key) or '')
            except ValueError:
                continue
            if not math.isfinite(t) or not math.isfinite(y) or t <= previous.get(name, -math.inf):
                continue
            previous[name] = t
            t -= starts.setdefault(name, t)
            counts[name] = counts.get(name, 0) + 1
            if t <= 10 and counts[name] % stride == 1:
                points[name].append([round(t, 5), round(y, 6)])
        if previous and all(previous[name] - starts[name] > 10 for name in channels if name in starts):
            if len(starts) == len(channels):
                break
print(json.dumps({name: data[:500] for name, data in points.items()}, separators=(',', ':')))
""".strip()


def fetch_board_preview(path):
    with board_session() as conn:
        raw = board_command(conn, python_command(BOARD_PREVIEW_SCRIPT, path), timeout=120)
    return json.loads(raw.splitlines()[-1])


def board_upload_widget(key):
    kind = 'baseline' if key == 'baseline' else 'test'
    folder = BOARD_RECORDING_FOLDERS[key]
    with st.expander(f'Upload a new {kind} CSV to the board'):
        recording = st.file_uploader('AFE4950 export (.csv)', type='csv', key=f'edge_upload_{key}')
        st.caption(f'The board verifies the upload and saves it in {folder}/uploads/. Uploading does not add it to training.')
        if st.button('Send CSV to SK-AM62B', key=f'edge_send_{key}', disabled=recording is None):
            progress = st.progress(0.0)
            try:
                with st.spinner('Sending and verifying over COM12…'):
                    path = upload_csv(recording.name, recording.getvalue(), progress, role=key)
                    info = st.session_state.board_inventory
                    info['paths'] = sorted(set(info['paths']) | {path})
                    st.session_state[f'edge_notice_{key}'] = f'Verified on board · {path}'
                st.rerun()
            except (BoardError, ValueError) as exc:
                st.error(str(exc))
            finally:
                progress.empty()
        if notice := st.session_state.pop(f'edge_notice_{key}', None):
            st.success(notice)


def board_preview_widget(path, key):
    if not path:
        return
    if st.button('Load ECG / PPG preview from board', key=f'edge_preview_{key}'):
        st.session_state.pop(f'edge_wave_{key}', None)
        try:
            with st.spinner('Reading waveform samples on the board…'):
                st.session_state[f'edge_wave_{key}'] = (path, fetch_board_preview(path))
        except (BoardError, ValueError, IndexError) as exc:
            st.error(str(exc))
    saved = st.session_state.get(f'edge_wave_{key}')
    if not saved or saved[0] != path:
        return
    section('Signal preview', 'raw samples returned by the SK-AM62B')
    columns = st.columns(2, gap='large')
    found = False
    for col, name, color in zip(columns, ('ECG', 'PPG'), (TEAL, CORAL)):
        trace = saved[1].get(name) or []
        if not trace:
            col.caption(f'{name} channel not found in this export.')
            continue
        found = True
        t, y = zip(*trace)
        fig = go.Figure(go.Scatter(x=t, y=y, mode='lines', line=dict(color=color, width=1.4)))
        fig.update_layout(xaxis_title='time (s)', yaxis_title='exported units', showlegend=False)
        col.plotly_chart(style_fig(fig, 230, name), width='stretch')
    if found:
        st.caption('Downsampled preview of the first 10 seconds. The board classifier uses the full recording, not these plotted points.')


def board_result_label(output):
    # Only read a verdict from a prediction line, not from hashes or directory names.
    candidates = []
    for line in output.splitlines():
        line = line.strip()
        if ': ' in line:
            prefix, value = line.split(': ', 1)
            if prefix.lower().endswith('.csv'):
                line = value
        match = re.match(
            r'^(?:(?:result|verdict|prediction|classification|decision)\s*[:=]?\s*)?'
            r'(NORMAL|FATIGUE)\b', line, re.I
        )
        if match:
            candidates.append(match[1].upper())
    return candidates[0] if len(set(candidates)) == 1 else None


def board_result_details(output):
    # Read only fields the SK-AM62B printed. Never run the local model to fill gaps.
    label = board_result_label(output)
    details = {'label': label, 'probability': None, 'threshold': None, 'measures': {}}
    if not label:
        return details
    number = r'(?:\d+(?:\.\d+)?|\.\d+)'
    for key, field in (
        ('probability', r'P\(fatigue\)\s*[:=]?'),
        ('threshold', r'(?:threshold\s*[:=]?|fatigue at\s*(?:≥|>=))'),
    ):
        match = re.search(rf'{field}\s*({number})(?![\d.%])', output, re.I)
        if match and 0 <= float(match[1]) <= 1:
            details[key] = float(match[1])
    for match in re.finditer(
        rf'\b(HR|SDNN|RMSSD)\s+({number})\s*(?:->|→)\s*({number})'
        rf'\s*(?:\(\s*([+\-−]?{number})\s*%\s*\))?', output, re.I
    ):
        name, before, after, pct = match.groups()
        details['measures'][name.upper()] = (
            float(before), float(after), pct.replace('−', '-') + '%' if pct is not None else None
        )
    return details


def render_board_result(result):
    section('Result', 'computed on the SK-AM62B')
    details = board_result_details(result['output'])
    label = details['label']
    if label:
        fatigued = label == 'FATIGUE'
        cols = st.columns([1.25, 1, 1, 1], gap='medium')
        verdict_card(cols[0], fatigued, details['probability'], details['threshold'])
        rows = []
        for col, (name, key, unit) in zip(cols[1:], (
            ('Heart rate', 'HR', 'bpm'), ('SDNN', 'SDNN', 'ms'), ('RMSSD', 'RMSSD', 'ms'),
        )):
            measure = details['measures'].get(key)
            if measure is None:
                stat(col, name, '—', 'Not reported by board')
                continue
            before, after, pct = measure
            change = after - before
            delta = f'{change:+g} {unit} vs baseline'
            if pct is not None:
                delta += f' ({pct})'
            stat(col, name, f'{after:g} {unit}', delta)
            rows.append({
                'Measure': f'{name} ({unit})', 'Baseline': before, 'Current': after,
                'Change': round(change, 3), 'Reported change (%)': pct or '—',
            })
        if result['expected'] != 'Not recorded':
            comparison = 'matches' if label == result['expected'] else 'does not match'
            st.caption(f"The prediction {comparison} the recorded label ({result['expected']}).")
        if fatigued:
            card('Take a breathing break',
                 'The board flagged this recording compared with the rested baseline. Pause and re-check when ready.',
                 'coral')
        if rows:
            section('Why: change from the baseline')
            st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        st.caption('The raw board result does not include decision weights. Use the detailed charts to compare heartbeat shape with a selected baseline.')
    else:
        st.info('No NORMAL or FATIGUE verdict was recognized. Check the original board response below.')
    hashes = re.findall(r'^([0-9a-fA-F]{64})\s+\*?(.+)$', result['output'], re.M)
    with st.expander('Board output and SHA-256 verification', expanded=not bool(label)):
        if hashes:
            st.caption('SHA-256 hashes identify the model and recording used by the SK-AM62B.')
        st.code(result['output'], language=None)


def capture_board_check(participant, path, expected):
    baseline = st.session_state.get('edge_session_baselines', {}).get(participant)
    if baseline is None:
        raise ValueError('Choose a session baseline before checking this participant.')
    if not path.startswith('data/test/'):
        raise ValueError('Choose a test CSV from the board.')
    result = run_session_board_check(participant, path, baseline)
    label = board_result_label(result)
    entry = {
        'time': datetime.now().isoformat(timespec='seconds'), 'participant': participant,
        'recording': path, 'board_label': label or 'See board output',
        'expected': expected, 'output': result,
    }
    st.session_state.setdefault('edge_history', []).append(entry)
    st.session_state.edge_last_check = entry


BOARD_READ_SCRIPT = """
import base64,gzip,hashlib,sys
from pathlib import Path
p=Path(sys.argv[1])
root=Path('data').resolve()
if not p.resolve().is_relative_to(root) or p.suffix.lower()!='.csv':
    raise SystemExit('Not a recording in board data')
with p.open('rb') as f:
    data=f.read(int(sys.argv[2])+1)
if len(data)>int(sys.argv[2]):
    raise SystemExit('This recording exceeds the detailed-chart transfer limit')
print('BB_CSV_SHA256 '+hashlib.sha256(data).hexdigest())
wire=base64.b64encode(gzip.compress(data,compresslevel=3)).decode()
for i in range(0,len(wire),192):
    print('BB_CSV_BLOCK '+wire[i:i+192])
""".strip()


def read_board_recording(path, expected_hash=None):
    # Read original recorded samples. The Windows dashboard processes them for charts only.
    with board_session() as conn:
        output = board_command(conn, python_command(BOARD_READ_SCRIPT, path, str(MAX_CSV_BYTES)),
                               timeout=2100)
    lines = output.splitlines()
    header = [line[14:] for line in lines if line.startswith('BB_CSV_SHA256 ')]
    blocks = [line[13:] for line in lines if line.startswith('BB_CSV_BLOCK ')]
    if len(header) != 1 or not blocks:
        raise BoardError('The board did not return a complete recording.')
    try:
        data = gzip.decompress(base64.b64decode(''.join(blocks), validate=True))
    except (ValueError, EOFError, OSError) as exc:
        raise BoardError('The chart recording transfer was incomplete or corrupt.') from exc
    if hashlib.sha256(data).hexdigest() != header[0] or (expected_hash and header[0] != expected_hash):
        raise BoardError('Recording checksum did not match. Run the board check again before plotting.')
    return data


def board_signal_analysis(path, expected_hash=None):
    cache = st.session_state.setdefault('edge_signal_cache', {})
    cache_key = (path, expected_hash)
    if cache_key not in cache:
        recording = read_board_recording(path, expected_hash)
        analysis = run_analysis(recording)
        if len(cache) >= 4:
            cache.pop(next(iter(cache)))
        cache[cache_key] = analysis
    return cache[cache_key]


def render_board_signal(res, role, matched=None):
    section('Signal check', 'dashboard analysis of board recordings')
    show_warnings(res, role)
    quality_stats(res)
    base = matched
    c1, c2 = st.columns(2, gap='large')
    c1.plotly_chart(overview_figure(res, 'Whole recording (grey = trimmed unstable sections)'), width='stretch')
    c2.plotly_chart(ecg_figure(res, 'First 10 seconds of the stable part'), width='stretch')
    if base is None:
        return
    st.caption('These plots compare the two board CSVs. They do not change the board verdict or its feature values.')
    c1, c2 = st.columns(2, gap='large')
    for column, chart in ((c1, 'Beat-to-beat heart rate'), (c2, 'Average heartbeat shape')):
        fig = go.Figure()
        for label, signal, color in (('baseline reference', base, TEAL), ('test recording', res, NAVY)):
            if chart == 'Beat-to-beat heart rate':
                t = np.asarray(signal['t_rr'])
                rr = np.asarray(signal['rr'])
                if len(t) and len(t) == len(rr):
                    fig.add_trace(go.Scatter(x=t - t[0], y=60 / rr, name=label, line=dict(color=color)))
                fig.update_layout(xaxis_title='seconds into stable part', yaxis_title='bpm')
            else:
                beat = np.asarray(signal['median_beat'])
                if len(beat):
                    t = (np.arange(len(beat)) - int(PRE * FS_ECG)) / FS_ECG * 1000
                    fig.add_trace(go.Scatter(x=t, y=beat, name=label, line=dict(color=color)))
                fig.update_layout(xaxis_title='ms from R-peak', yaxis_title='mV')
        column.plotly_chart(style_fig(fig, 300, chart), width='stretch')


def board_detailed_charts(path, role, info, result=None):
    if not path:
        return
    if role == 'check' and result is None:
        return
    ref_path = None
    expected_hash = None
    if result:
        expected_hash = next((digest for digest, name in re.findall(
            r'^([0-9a-fA-F]{64})\s+\*?(.+)$', result['output'], re.M) if name == path), None)
        options = [p for p in info['paths'] if p.startswith('data/baseline/')]
        baseline = st.session_state.get('edge_session_baselines', {}).get(result['participant'], {})
        registered = baseline.get('path')
        if baseline.get('data') is not None:
            registered = 'Session upload: ' + baseline['name']
        if registered:
            if registered in options:
                options.remove(registered)
            options.insert(0, registered)
        ref_path = st.selectbox('Baseline reference for detailed charts (optional)', [None, *options],
                               index=int(registered in options), key='edge_chart_reference',
                               format_func=lambda p: p or 'No reference selected')
        st.caption('Choose the matching participant’s baseline for charts. This selection does not change board inference.')
    st.caption('Detailed charts download the full recording from the board. Use the quick ECG / PPG preview if you only need a short trace.')
    state_key = f'edge_detailed_{role}'
    selection = (path, ref_path, expected_hash)
    if st.button('Load detailed dashboard charts', key=f'edge_chart_load_{role}'):
        st.session_state.pop(state_key, None)
        try:
            with st.spinner('Reading board recordings and computing dashboard plots…'):
                current = board_signal_analysis(path, expected_hash)
                if result and ref_path and ref_path == registered and baseline.get('data') is not None:
                    reference = analyze(*load_recording(baseline['data']))
                else:
                    reference = board_signal_analysis(ref_path) if ref_path else None
            st.session_state[state_key] = (selection, current, reference)
        except Exception as exc:
            st.error(f'Charts could not be loaded. The board result is unchanged. {exc}')
    saved = st.session_state.get(state_key)
    if not saved or saved[0] != selection:
        return
    current, reference = saved[1:]
    st.caption('Chart and signal-quality calculations run in the dashboard from board CSVs. Classification runs on the SK-AM62B.')
    if reference is not None and reference.get('usable') and current.get('usable'):
        match = beat_similarity(np.asarray(reference['median_beat']), np.asarray(current['median_beat']))
        if np.isfinite(match):
            stat(st, 'Beat-shape match', f'{match:.3f}', 'Dashboard comparison with the selected reference, not a board model input')
    render_board_signal(current, role, reference)


def show_board_history(history):
    if not history:
        st.info('No board checks yet. Go to ② Input recording to run a prediction.')
        return
    frame = pd.DataFrame(history)
    people = sorted(frame.participant.unique())
    person = st.selectbox('Person', people, key='edge_history_person')
    selected = frame[frame.participant == person]
    c = st.columns(3)
    stat(c[0], 'Checks', len(selected))
    stat(c[1], 'Fatigue', int((selected.board_label == 'FATIGUE').sum()))
    stat(c[2], 'Last check', selected.time.iloc[-1][:16].replace('T', ' '), selected.board_label.iloc[-1])
    scores = [board_result_details(output) for output in selected.output]
    if any(item['probability'] is not None for item in scores):
        fig = go.Figure(go.Scatter(x=selected.time, y=[item['probability'] for item in scores],
                                   mode='lines+markers', line=dict(color='#b9cdd3'),
                                   marker=dict(color=np.where(selected.board_label == 'FATIGUE', CORAL, TEAL), size=11)))
        thresholds = {item['threshold'] for item in scores}
        if len(thresholds) == 1 and None not in thresholds:
            fig.add_hline(y=next(iter(thresholds)), line_dash='dash', line_color=CORAL, annotation_text='board threshold')
        fig.update_layout(yaxis=dict(range=[0, 1], title='Board P(fatigue)'))
        st.plotly_chart(style_fig(fig, 300), width='stretch')
    visible = selected.drop(columns='output').assign(probability=[item['probability'] for item in scores])
    st.dataframe(visible.iloc[::-1], hide_index=True, width='stretch')
    st.download_button('⬇️ Download board check history', selected.to_csv(index=False), 'board_history.csv', 'text/csv')
    with st.expander('Latest board response'):
        st.code(selected.output.iloc[-1], language=None)
    if st.button('🗑 Clear board history (all people)'):
        st.session_state.pop('edge_history', None)
        st.rerun()


def board_hr_page():
    section('Workload reports for HR', 'board-generated fatigue flags')
    st.caption('Reports contain recording summaries, not raw ECG. This tab does not send messages automatically.')
    history = [item for item in st.session_state.get('edge_history', []) if item['board_label'] == 'FATIGUE']
    if not history:
        st.info('No fatigue flags from the SK-AM62B in this session.')
        return
    people = sorted({item['participant'] for item in history})
    who = st.multiselect('Person', people, default=people, key='edge_hr_people')
    rows = []
    for item in reversed(history):
        if item['participant'] not in who:
            continue
        details = board_result_details(item['output'])
        row = {k: item[k] for k in ('time', 'participant', 'recording')}
        row['probability'] = details['probability']
        row['threshold'] = details['threshold']
        for key in ('HR', 'SDNN', 'RMSSD'):
            measure = details['measures'].get(key)
            row[key + ' baseline'] = measure[0] if measure else None
            row[key + ' current'] = measure[1] if measure else None
        rows.append(row)
    if not rows:
        return
    report = pd.DataFrame(rows)
    c1, c2 = st.columns(2)
    stat(c1, 'Fatigue flags', len(report))
    stat(c2, 'People flagged', report.participant.nunique())
    if report.probability.notna().any():
        fig = go.Figure(go.Scatter(x=report.time, y=report.probability, mode='markers',
                                   text=report.participant, marker=dict(color=CORAL, size=11)))
        fig.update_layout(yaxis=dict(range=[0, 1], title='Board P(fatigue)'))
        st.plotly_chart(style_fig(fig, 280), width='stretch')
    st.dataframe(report, hide_index=True, width='stretch')
    st.download_button('⬇️ Download board HR report', report.to_csv(index=False), 'board_hr_report.csv', 'text/csv')
    with st.expander('Message text of the latest flag'):
        entry = rows[0]
        st.code('\n'.join(f'{key}: {value}' for key, value in entry.items() if value is not None), language=None)


def board_model_panels(connected):
    section('Model diagnostics', 'read from the SK-AM62B')
    if st.button('Read board model diagnostics', disabled=not connected):
        st.session_state.pop('edge_model_diagnostics', None)
        try:
            with board_session() as conn:
                raw = board_command(conn, f'cat {BOARD_MODEL}')
            st.session_state.edge_model_diagnostics = json.loads(raw)
        except (BoardError, ValueError) as exc:
            st.error(str(exc))
    raw = st.session_state.get('edge_model_diagnostics')
    if not isinstance(raw, dict):
        return
    model = raw.get('model', raw)
    if not isinstance(model, dict):
        model = raw
    threshold = model.get('threshold', raw.get('threshold'))
    training = model.get('training', raw.get('training'))
    ev = model.get('evaluation', raw.get('evaluation'))
    if isinstance(training, dict):
        st.dataframe(pd.DataFrame([training]), hide_index=True, width='stretch')
    weights = model.get('coef', model.get('weights'))
    features = model.get('features', raw.get('features'))
    if isinstance(weights, list) and len(weights) == 1 and isinstance(weights[0], list):
        weights = weights[0]
    if isinstance(weights, list) and isinstance(features, list) and len(weights) == len(features):
        weights = dict(zip(map(str, features), weights))
    if isinstance(weights, dict):
        values = [(str(k), v) for k, v in weights.items() if isinstance(v, (int, float))]
        if values:
            fig = go.Figure(go.Bar(x=[v for k, v in values], y=[k for k, v in values], orientation='h',
                                   marker_color=[CORAL if v > 0 else TEAL for k, v in values]))
            st.plotly_chart(style_fig(fig, 260, 'What the board model uses'), width='stretch')
    if isinstance(ev, dict):
        confusion = ev.get('confusion')
        if isinstance(confusion, list) and len(confusion) == 2 and all(
            isinstance(row, list) and len(row) == 2 for row in confusion
        ):
            st.dataframe(pd.DataFrame(confusion, index=['true normal', 'true fatigue'],
                                      columns=['predicted normal', 'predicted fatigue']), width='stretch')
        records = ev.get('per_recording')
        if isinstance(records, list) and records and all(isinstance(row, dict) for row in records):
            frame = pd.DataFrame(records)
            if {'new_recording', 'p_fatigue'} <= set(frame):
                fig = go.Figure(go.Bar(x=frame.p_fatigue, y=frame.new_recording, orientation='h'))
                if isinstance(threshold, (int, float)):
                    fig.add_vline(x=threshold, line_dash='dash', line_color=NAVY)
                st.plotly_chart(style_fig(fig, 100 + 30 * len(frame), 'Held-out predictions reported by the board'), width='stretch')
            st.dataframe(frame, hide_index=True, width='stretch')
        if isinstance(ev.get('details'), list):
            with st.expander('All held-out comparisons'):
                st.dataframe(pd.DataFrame(ev['details']), hide_index=True, width='stretch')
    else:
        st.caption('This board model file does not contain structured held-out results. Check the board training output and the separate research comparison.')
    with st.expander('Board model data'):
        st.json(raw)


BOARD_CONTRIBUTE_SCRIPT = """
import csv,os,shutil,sys,uuid
from pathlib import Path
src,person,label,kss,note=sys.argv[1:]
src=Path(src)
if not src.resolve().is_relative_to(Path('data').resolve()) or not src.is_file():
    raise SystemExit('Choose an existing board recording')
manifest=Path('board_manifest.csv')
with manifest.open(encoding='utf-8-sig',newline='') as f:
    reader=csv.DictReader(f)
    fields,rows=reader.fieldnames,list(reader)
lookup={name.strip().lower():name for name in fields or []}
def field(*names):
    return next((lookup[n] for n in names if n in lookup),None)
path=field('file','path','recording','csv','csv_path','filename')
uid=field('person','participant','subject','user','participant_id','subject_id')
kind=field('condition','label','state','class','status')
if not all((path,uid,kind)) or any(row.get(None) for row in rows):
    raise SystemExit('Board manifest format is not recognized. No files were changed.')
labels={(row.get(kind) or '').strip().lower() for row in rows}
if not labels or not labels <= {'normal','fatigue'}:
    raise SystemExit('Board manifest does not use normal/fatigue labels. No files were changed.')
if any((row.get(path) or '').strip()==str(src) for row in rows):
    raise SystemExit('This recording is already listed in the board training manifest.')
row=dict.fromkeys(fields,'')
row.update({path:str(src),uid:person,kind:label})
for key,val in ((field('kss'),kss),(field('notes','note'),note)):
    if key:
        row[key]=val
tmp=manifest.with_name('.board_manifest.'+uuid.uuid4().hex+'.tmp')
try:
    with tmp.open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fields)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(row)
    shutil.copymode(manifest,tmp)
    os.replace(tmp,manifest)
finally:
    tmp.unlink(missing_ok=True)
print('Added to board training manifest:',src)
""".strip()


def contribute_board_recording(path, participant, label, kss, note):
    if not ID_PATTERN.fullmatch(participant or '') or label not in ('normal', 'fatigue'):
        raise ValueError('Enter a valid participant ID and recording label.')
    if not path.startswith(('data/baseline/', 'data/test/')):
        raise ValueError('Choose a baseline or test recording on the board.')
    with board_session() as conn:
        return board_command(conn, python_command(BOARD_CONTRIBUTE_SCRIPT, path, participant, label,
                                                   str(kss), note), timeout=30)


def board_contribution_widget(path, participant, role):
    if role == 'check':
        title = '➕ Add this recording to the board training data'
    else:
        title = '➕ Also add this baseline to board training as normal'
    with st.expander(title):
        st.caption('Once added to training, this recording should no longer count as an independent test for a retrained model.')
        label = 'normal'
        kss = ''
        if role == 'check':
            label = st.radio('How did you actually feel?', ['fatigue', 'normal'], horizontal=True,
                             key='edge_contrib_label',
                             format_func=lambda x: {'fatigue': 'Fatigued / tired', 'normal': 'Normal / rested'}[x])
            kss = st.slider('Sleepiness (Karolinska Scale)', 1, 9, 7 if label == 'fatigue' else 3,
                            key='edge_contrib_kss')
        note = st.text_input('Notes (optional)', max_chars=120, key=f'edge_contrib_note_{role}').strip()
        ok = st.checkbox('I agree this recording can be used to train the model', key=f'edge_contrib_ok_{role}')
        if st.button('Add to board training data', key=f'edge_contrib_{role}', disabled=not (ok and participant)):
            try:
                message = contribute_board_recording(path, participant, label, kss, note)
                st.session_state.pop('edge_manifest', None)
                st.success(message + ' · Retrain under Model when ready.')
            except (BoardError, ValueError) as exc:
                st.error(str(exc))


BOARD_SESSION_CHECK_SCRIPT = """
import hashlib,os,shutil,subprocess,sys
from pathlib import Path
root=Path.cwd().resolve()
temp=Path(sys.argv[1]).resolve()
baseline,test,person=sys.argv[2:]
if temp.parent!=Path('/tmp').resolve() or not temp.name.startswith('bb-check.'):
    raise SystemExit('Invalid temporary workspace')
def recording(path,folder):
    p=(root/path).resolve()
    if p.suffix.lower()!='.csv' or not p.is_file() or not p.is_relative_to(folder):
        raise SystemExit('Invalid recording path: '+path)
    return p
base=recording(baseline,temp if Path(baseline).is_absolute() else root/'data')
test_csv=recording(test,root/'data'/'test')
if base==test_csv:
    raise SystemExit('Choose a test recording different from the session baseline.')
for path in root.glob('*.py'):
    shutil.copy2(path,temp/path.name)
for rel in ('models/shift_model.json','board_manifest.csv'):
    src=root/rel
    if src.is_file():
        dst=temp/rel
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
model=temp/'models/shift_model.json'
if not model.is_file():
    raise SystemExit('Board model not found. Train it under Model first.')
if hashlib.sha256(base.read_bytes()).digest()==hashlib.sha256(test_csv.read_bytes()).digest():
    raise SystemExit('The baseline and test contain identical data. Choose a different recording.')
base_rel='data/baseline/session_baseline.csv'
test_rel='data/test/'+test_csv.name
for src,rel in ((base,base_rel),(test_csv,test_rel)):
    dest=temp/rel
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dest)
runner='''
import os,runpy,sys
from pathlib import Path
root=Path.cwd().resolve()
def ensure(path):
    if not isinstance(path,int) and not Path(os.fsdecode(path)).resolve().is_relative_to(root):
        raise PermissionError("Session operation cannot write outside its temporary workspace")
def audit(event,args):
    if event=='open':
        path,mode,flags=args
        if flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
            ensure(path)
    elif event in ('os.remove','os.rmdir','os.mkdir','os.chmod','os.utime','os.truncate','os.chown'):
        ensure(args[0])
    elif event in ('os.rename','os.link','os.symlink'):
        ensure(args[0]); ensure(args[1])
    elif event in ('os.system','subprocess.Popen'):
        raise PermissionError("Session classifier subprocesses are not supported")
sys.addaudithook(audit)
sys.argv=['board_shift.py',*sys.argv[1:]]
runpy.run_path('board_shift.py',run_name='__main__')
'''
def run(*args):
    return subprocess.run([sys.executable,'-B','-u','-c',runner,*args],cwd=temp,
                          env={**os.environ,'HOME':str(temp),'XDG_DATA_HOME':str(temp)},
                          stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
before=hashlib.sha256(model.read_bytes()).hexdigest()
for label,source in (('models/shift_model.json',model),(test,temp/test_rel)):
    print(hashlib.sha256(source.read_bytes()).hexdigest()+'  '+label,flush=True)
enrollment=run('enroll',person,base_rel)
if enrollment.returncode:
    print(enrollment.stdout,end='')
    raise SystemExit(enrollment.returncode)
people=run('people')
if people.returncode or not any(line.split()[:1]==[person] and base_rel in line
                              for line in people.stdout.splitlines()):
    print(enrollment.stdout,end='')
    raise SystemExit('Board did not confirm enrollment of the temporary baseline')
result=run('check',person,test_rel)
print(result.stdout.replace(test_rel,test).rstrip())
if hashlib.sha256((root/'models/shift_model.json').read_bytes()).hexdigest()!=before:
    raise SystemExit('The persistent board model changed during this check. Refresh and try again.')
raise SystemExit(result.returncode)
""".strip()


def compressed_python_command(source, *args):
    payload = base64.b64encode(gzip.compress(source.encode())).decode('ascii')
    launcher = f"import base64,gzip;exec(gzip.decompress(base64.b64decode('{payload}')))"
    return ' '.join(['python3', '-c', quote(launcher), *(quote(arg) for arg in args)])


def run_session_board_check(participant, path, baseline):
    if not ID_PATTERN.fullmatch(participant):
        raise ValueError('Choose a valid participant ID.')
    with board_session() as conn:
        temp = board_command(conn, 'mktemp -d /tmp/bb-check.XXXXXXXX').splitlines()[-1]
        if not re.fullmatch(r'/tmp/bb-check\.[A-Za-z0-9]{8}', temp):
            raise BoardError('The board did not return a valid temporary workspace.')
        try:
            baseline_path = baseline['path']
            if baseline['data'] is not None:
                baseline_path = temp + '/upload.csv'
                bar = st.progress(0.0)
                try:
                    send_csv_bytes(conn, baseline['data'], baseline_path, bar)
                finally:
                    bar.empty()
            # A new ID prevents the model copy from falling back to pre-existing participant references.
            run_id = 'session_' + uuid.uuid4().hex
            return board_command(conn, compressed_python_command(BOARD_SESSION_CHECK_SCRIPT,
                                 temp, baseline_path, path, run_id), timeout=300)
        finally:
            board_command(conn, f'rm -rf -- {quote(temp)}', timeout=15)


def clear_session_baselines():
    for key in ('edge_session_baselines', 'edge_last_check', 'edge_chart_reference', 'edge_check_participant',
                'edge_enroll_path', 'edge_enroll_uid', 'edge_detailed_baseline',
                'edge_detailed_check', 'edge_signal_cache', 'edge_wave_baseline'):
        st.session_state.pop(key, None)
    st.session_state['edge_baseline_revision'] = st.session_state.get('edge_baseline_revision', 0) + 1


def board_enrolled_ids(people_output):
    # board_shift.py people prints one case-sensitive ID and baseline list per line.
    people = {}
    for line in people_output.splitlines():
        record = re.fullmatch(r'\s*(\S+)\s+\[\s*(.+?)\s*\]\s*', line)
        if record and ID_PATTERN.fullmatch(record[1]) and record[2].strip():
            people[record[1]] = None
    return list(people)


def board_page(page):
    info = st.session_state.get('board_inventory')
    if page == 'History':
        section('History', 'board checks from this browser session')
        show_board_history(st.session_state.get('edge_history', []))
        return
    if page == 'HR reports':
        board_hr_page()
        return
    connected = info is not None
    if not connected:
        st.info('Click Connect / refresh board in the sidebar to load your board model, participants and recordings.')
        info = {'paths': [], 'people': '', 'model': 'Board not connected.'}

    if page == PAGES[0]:
        section('Input baseline', 'one rested recording per person', '1')
        left, right = st.columns([1, 1.45], gap='large')
        with left:
            card('How to record a baseline',
                 '<ol><li>Use a clean, rested AFE4950 ECG recording.</li>'
                 '<li>Use the same participant ID for later recordings.</li>'
                 '<li>Keep the rested baseline separate from the test recording.</li></ol>')
            card('Why a personal baseline?',
                 'BreathBreak compares each recording with the same person’s rested state.', 'amber')
        with right:
            with st.container(border=True):
                participant = st.text_input('Name or ID', placeholder='e.g. D', key='edge_enroll_uid').strip()
                paths = sorted(info['paths'], key=lambda p: (not p.startswith('data/baseline/'), p))
                path = st.selectbox('Rested recording on board (all CSVs)', paths, index=None, key='edge_enroll_path')
                with st.expander('Use a CSV from this computer for this session'):
                    revision = st.session_state.get('edge_baseline_revision', 0)
                    uploaded = st.file_uploader('AFE4950 export (.csv)', type='csv',
                                                key=f'edge_session_upload_{revision}', disabled=not connected)
                    st.caption('This upload takes priority over the board dropdown and is not saved to the board dataset.')
                st.caption('Choose a rested recording. The baseline assignment is cleared on browser refresh or Connect / refresh board.')
                if st.button('Use baseline for this session', type='primary', disabled=not (participant and (path or uploaded is not None))):
                    try:
                        if not ID_PATTERN.fullmatch(participant):
                            raise ValueError('Use an ID starting with a letter or digit, with at most 40 letters, digits, underscores, or hyphens.')
                        data = uploaded.getvalue() if uploaded is not None else None
                        if data is not None and (not uploaded.name.lower().endswith('.csv')
                                                 or not 0 < len(data) <= MAX_CSV_BYTES):
                            raise ValueError('Choose a nonempty CSV no larger than 12 MiB.')
                        entry = {'path': path if data is None else None, 'data': data,
                                 'name': path if data is None else uploaded.name}
                        st.session_state.setdefault('edge_session_baselines', {})[participant] = entry
                        st.session_state.pop('edge_last_check', None)
                        st.success(f'Baseline selected for {participant} this session.')
                    except ValueError as exc:
                        st.error(str(exc))
        session_baselines = st.session_state.get('edge_session_baselines', {})
        if session_baselines:
            st.dataframe(pd.DataFrame([{'Participant': person, 'Session baseline': entry['name']}
                                      for person, entry in session_baselines.items()]), hide_index=True, width='stretch')
        if uploaded is None:
            board_preview_widget(path, 'baseline')
            board_detailed_charts(path, 'baseline', info)
        elif st.button('Preview the uploaded session baseline', key='edge_preview_session_upload'):
            try:
                data = uploaded.getvalue()
                if not 0 < len(data) <= MAX_CSV_BYTES:
                    raise ValueError('Choose a nonempty CSV no larger than 12 MiB.')
                render_board_signal(analyze(*load_recording(data)), 'baseline')
            except Exception as exc:
                st.error(f'Could not preview this CSV: {exc}')
        with st.expander('Research comparison and limitations'):
            model_comparison()
            fatigue_types()
        return

    if page == PAGES[1]:
        section('Input recording', 'compared with the person’s own baseline', '2')
        left, right = st.columns([1, 1.45], gap='large')
        with left:
            card('Participant baseline', 'Choose the ID used to enroll the rested recording.', 'navy')
            enrolled = list(st.session_state.get('edge_session_baselines', {}))
            participant = st.selectbox('Enrolled participant ID', enrolled, index=None,
                                       placeholder='Choose a participant with a session baseline',
                                       key='edge_check_participant', disabled=not enrolled)
            if connected and not enrolled:
                st.info('No session baselines yet. Go to ① Input baseline and choose a rested recording.')
            if participant:
                entry = st.session_state['edge_session_baselines'][participant]
                st.caption(f"Session baseline: {entry['name']}")
        with right:
            with st.container(border=True):
                if connected:
                    board_upload_widget('check')
                prefix = BOARD_RECORDING_FOLDERS['check'] + '/'
                paths = [p for p in info['paths'] if p.startswith(prefix)]
                path = st.selectbox('Recording on board', paths, index=None, key='edge_check_path')
                expected = st.selectbox('Independent label (optional)', ['Not recorded', 'NORMAL', 'FATIGUE'])
                if st.button('▶ Check this recording on SK-AM62B', type='primary', disabled=not (participant and path)):
                    st.session_state.pop('edge_last_check', None)
                    try:
                        with st.spinner('Preprocessing and scoring on the SK-AM62B…'):
                            capture_board_check(participant, path, expected)
                    except (BoardError, ValueError) as exc:
                        st.error(str(exc))
        result = st.session_state.get('edge_last_check')
        if result and (result['participant'], result['recording']) == (participant, path):
            render_board_result(result)
            board_detailed_charts(path, 'check', info, result)
            if result['board_label'] in ('NORMAL', 'FATIGUE'):
                board_contribution_widget(path, participant, 'check')
        board_preview_widget(path, 'check')
        return

    section('Model & retraining', 'SK-AM62B personal-shift model')
    left, right = st.columns(2, gap='large')
    with left:
        card('What it learns', 'The board compares ECG-derived heart rate, SDNN and RMSSD '
             'with the same participant’s rested baseline.')
    with right:
        card('Training data', 'The SK-AM62B uses its board_manifest.csv and saves the fitted model '
             'to models/shift_model.json. Windows does not train or load a fallback model.', 'navy')
    st.code(info['model'], language=None)
    with st.expander('Training manifest on SK-AM62B'):
        if st.button('Read board manifest', disabled=not connected):
            try:
                with board_session() as conn:
                    st.session_state.edge_manifest = board_command(conn, 'cat board_manifest.csv')
            except BoardError as exc:
                st.error(str(exc))
        if manifest := st.session_state.get('edge_manifest'):
            try:
                import io
                st.dataframe(pd.read_csv(io.StringIO(manifest)), hide_index=True, width='stretch')
            except (ValueError, pd.errors.ParserError):
                pass
            st.code(manifest, language='csv')
    if st.button('🔁 Train model on SK-AM62B', type='primary', disabled=not connected):
        st.session_state.pop('edge_training_output', None)
        st.session_state.pop('edge_last_check', None)
        st.session_state.pop('edge_model_diagnostics', None)
        try:
            with st.spinner('Training and checking the saved model on the board…'):
                st.session_state.edge_training_output = train_on_board()
                st.session_state.board_inventory = inventory()
            st.rerun()
        except (BoardError, ValueError, IndexError) as exc:
            st.error(str(exc))
    if report := st.session_state.get('edge_training_output'):
        st.code(report, language=None)
    board_model_panels(connected)
    with st.expander('Original research comparison and limitations'):
        st.caption('Reference results from the project report, separate from the most recent board run.')
        model_comparison()
        fatigue_types()

# ---------------- sidebar ----------------
html(CSS)
PAGES = ['① Input baseline', '② Input recording', 'History', 'HR reports', 'Model']
with st.sidebar:
    html("<div class='bb-side-kicker'>UTD HACKEDGE 2026</div><div class='bb-side-brand'>🫀 BreathBreak</div>")
    mode = st.radio('Processing device', ['SK-AM62B (COM12)', 'Local (original)'], key='processing_mode')
    board_mode = mode == 'SK-AM62B (COM12)'
    page = st.radio('Go to', PAGES, label_visibility='collapsed')
    model = None
    if board_mode:
        html("<div class='bb-side-box'><b>SK-AM62B</b><br>Board data, training and inference. Windows displays the dashboard.</div>")
        if st.button('Connect / refresh board', key='edge_connect', type='primary'):
            st.session_state.pop('board_inventory', None)
            clear_session_baselines()
            for key in ('edge_signal_cache', 'edge_detailed_baseline', 'edge_detailed_check',
                        'edge_model_diagnostics', 'edge_manifest'):
                st.session_state.pop(key, None)
            try:
                with st.spinner('Querying the board via COM12…'):
                    st.session_state.board_inventory = inventory()
            except (BoardError, ValueError, IndexError) as exc:
                st.error(str(exc))
        snapshot = st.session_state.get('board_inventory')
        if snapshot:
            st.success('Board responded at last refresh')
            st.caption('Refreshing the board connection clears session baselines.')
            st.caption(' · '.join(snapshot['device'].splitlines()))
        else:
            st.caption('Keep the board powered on and logged into Linux. Close Miniterm to free COM12.')
    else:
        model = get_model()
        if model:
            ev = model.meta.get('evaluation', {})
            html(f"<div class='bb-side-box'><b>Local model</b><br>Trained {model.meta['trained_at'][:16].replace('T', ' ')} on "
                 f"{model.meta['training']['recordings']} recordings from {model.meta['training']['people']} people."
                 + (f"<br>Held-out: <b>{ev['correct']}/{ev['comparisons']}</b> correct." if ev else '') + "</div>")
        else:
            st.error('No local model yet. Switch to SK-AM62B or run `python train_delta_model.py`.')
    html("<div class='bb-side-box'><b>How it works</b><br>① Everyone records a <b>rested baseline</b> once.<br>"
         "② New recordings are compared with <b>their own</b> baseline.<br>HR and HRV changes inform the model.</div>")

hero(page)

if board_mode:
    board_page(page)
    footer()
    st.stop()

# ---------------- ① baseline ----------------
if page == PAGES[0]:
    section('Input baseline', 'one rested recording per person', '1')
    left, right = st.columns([1, 1.45], gap='large')
    with left:
        card('How to record a baseline',
             "<ol><li>Be <b>rested and healthy</b>; sit quietly for 5 minutes.</li>"
             "<li>Record <b>3–5 minutes</b> without talking or moving.</li>"
             "<li>Check that the ECG wave looks clean in the TI GUI.</li>"
             "<li>Export <code>AFE4950_CAPTURED_DATA.csv</code> (manual §6.3) and upload it.</li>"
             "<li><b>Mark the electrode spots</b> so later recordings use the same placement.</li></ol>")
        card('Why a personal baseline?',
             "<span class='bb-muted'>Resting heart rate and HRV differ a lot between people. BreathBreak looks at how far "
             "<b>you</b> have moved from <b>your own</b> rested state.</span>", 'amber')
    with right:
        with st.container(border=True):
            uid = safe_id(st.text_input('Name or ID', '', placeholder='e.g. Nhat'))
            data, name = recording_input('baseline', '▶ Analyze this baseline')
    if data:
        res = run_analysis(data)
        section('Signal check', name)
        show_warnings(res, 'baseline')
        quality_stats(res)
        g1, g2 = st.columns([1, 1])
        g1.plotly_chart(overview_figure(res, 'Whole recording (grey = trimmed unstable start / end)'), width='stretch')
        g2.plotly_chart(ecg_figure(res, 'First 10 s of the stable part'), width='stretch')
        if res['usable']:
            section('Save baseline', uid if uid != 'user' else '')
            if uid in users():
                st.warning(f'**{uid}** already has a baseline. Saving will replace it.')
            add = st.checkbox('Also add this recording to the training data as **normal** (helps improve the model)')
            if st.button('💾 Save as baseline', type='primary', disabled=not uid or uid == 'user'):
                d = USERS / uid; d.mkdir(parents=True, exist_ok=True)
                (d / 'baseline.csv').write_bytes(data)
                (d / 'baseline.json').write_text(json.dumps(dict(
                    user=uid, source=name, saved_at=datetime.now().isoformat(timespec='seconds'),
                    features=res['features'], quality={k: v for k, v in res['quality'].items()},
                    median_beat=res['median_beat'].tolist(),
                    hr_t=res['t_rr'].tolist(), hr=(60 / res['rr']).tolist()), indent=1, default=float))
                if add:
                    contribute(data, name, uid, 'normal', None, 'baseline')
                st.success(f'Baseline saved for **{uid}**. Next, go to **② Input recording**.')
            if not uid or uid == 'user':
                st.caption('Enter a name or ID above to save.')
    model_comparison()
    how_model_a_works(model)
    fatigue_types()
# ---------------- ② check ----------------
elif page == PAGES[1]:
    section('Input recording', "compared with the person's own baseline", '2')
    if not users():
        st.info('No baselines yet. Start with **① Input baseline**.'); footer(); st.stop()
    if model is None:
        st.error('No trained model. Run `python train_delta_model.py`.'); footer(); st.stop()
    left, right = st.columns([1, 1.45], gap='large')
    with left:
        uid = st.selectbox('Person', users())
        base = load_baseline(uid)
        bf = base['features']
        card(f'Baseline of {uid}',
             f"<div style='display:flex;gap:18px;margin:4px 0 6px'>"
             f"<div><div class='bb-muted'>Heart rate</div><b style='font-size:1.3rem'>{bf['HR']:.0f}</b> bpm</div>"
             f"<div><div class='bb-muted'>SDNN</div><b style='font-size:1.3rem'>{bf['SDNN']:.0f}</b> ms</div>"
             f"<div><div class='bb-muted'>RMSSD</div><b style='font-size:1.3rem'>{bf['RMSSD']:.0f}</b> ms</div></div>"
             f"<span class='bb-muted'>{base['source']} · saved {base['saved_at'][:16].replace('T', ' ')}</span>")
    with right:
        with st.container(border=True):
            data, name = recording_input('check', '▶ Check this recording')
    if not data:
        footer(); st.stop()
    res = run_analysis(data)
    show_warnings(res, 'check')
    if not res['usable']:
        footer(); st.stop()
    sim = beat_similarity(np.array(base['median_beat']), res['median_beat'])
    p, contrib = model.predict(res['features'], base['features'])
    fatigued = model.level(p) == 'fatigue'
    col = CORAL if fatigued else TEAL
    verdict = 'FATIGUE' if fatigued else 'NORMAL'
    section('Result', name)
    f = res['features']
    c = st.columns([1.25, 1, 1, 1], gap='medium')
    verdict_card(c[0], fatigued, p, model.threshold)
    stat(c[1], 'Heart rate', f"{f['HR']:.0f} bpm", f"{f['HR'] - bf['HR']:+.0f} bpm vs baseline ({f['HR'] / bf['HR'] - 1:+.0%})")
    stat(c[2], 'RMSSD', f"{f['RMSSD']:.0f} ms", f"{f['RMSSD'] - bf['RMSSD']:+.0f} ms vs baseline ({f['RMSSD'] / bf['RMSSD'] - 1:+.0%})")
    stat(c[3], 'Beat-shape match', f'{sim:.3f}', '1.000 = same heartbeat shape as baseline')
    st.write('')
    if fatigued:
        card('Take a breathing break',
             'Clear fatigue signs compared with your baseline. Take <b>2 minutes of slow breathing</b> '
             '(about 6 breaths per minute), then re-check.', 'coral')
    if sim < 0.98:
        st.warning(f'The heartbeat shape differs from the baseline (match {sim:.3f}). Check that the right person is '
                   'selected; otherwise the electrodes were placed differently. The model uses heart rate and HRV, '
                   'which are not affected by electrode placement.')
    section('Why: change from the baseline')
    t = contrib.copy()
    t['Measure'] = [f"{LABELS[k][0]} ({LABELS[k][1]})" if LABELS[k][1] else LABELS[k][0] for k in t.feature]
    t['Direction'] = np.where(t.push_to_fatigue > 0, '→ fatigue', '→ normal')
    st.dataframe(t[['Measure', 'baseline', 'new', 'change', 'pct_change', 'push_to_fatigue', 'Direction']].rename(columns={
        'baseline': 'Baseline', 'new': 'Now', 'change': 'Change', 'pct_change': 'Change %', 'push_to_fatigue': 'Effect on decision'}).round(2),
        hide_index=True, width='stretch')
    g1, g2 = st.columns(2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.array(base['hr_t']) - base['hr_t'][0], y=base['hr'], name='baseline', line=dict(color=TEAL, width=2)))
    fig.add_trace(go.Scatter(x=res['t_rr'] - res['t_rr'][0], y=60 / res['rr'], name='now', line=dict(color=CORAL if fatigued else NAVY, width=2)))
    fig.update_layout(xaxis_title='seconds into stable part', yaxis_title='bpm')
    g1.plotly_chart(style_fig(fig, 300, 'Beat-to-beat heart rate'), width='stretch')
    mb, mn = np.array(base['median_beat']), res['median_beat']; tb = (np.arange(len(mb)) - int(PRE*FS_ECG)) / FS_ECG * 1000
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tb, y=mb, name='baseline', line=dict(color=TEAL, width=2.5)))
    fig.add_trace(go.Scatter(x=tb[:len(mn)], y=mn, name='now', line=dict(color=CORAL if fatigued else NAVY, width=2.5)))
    fig.update_layout(xaxis_title='ms from R-peak', yaxis_title='mV')
    g2.plotly_chart(style_fig(fig, 300, 'Average heartbeat shape'), width='stretch')
    st.plotly_chart(ecg_figure(res, 'New recording: first 10 s of the stable part'), width='stretch')
    hist = USERS / uid / 'history.csv'
    key = f'logged_{uid}_{hash(data)}'
    if key not in st.session_state:
        pd.DataFrame([dict(time=datetime.now().isoformat(timespec='seconds'), recording=name, result=verdict,
                           p_fatigue=round(p, 3), hr=round(f['HR'], 1), rmssd=round(f['RMSSD'], 1),
                           beat_match=round(sim, 3))]).to_csv(hist, mode='a', header=not hist.exists(), index=False)
        st.session_state[key] = True
        if fatigued:                                    # fatigue -> one HR report per check
            rep = send_report(build_report(uid, name, p, model.threshold, f, base['features'], contrib, sim))
            st.session_state[f'hr_{key}'] = rep
    rep = st.session_state.get(f'hr_{key}')
    if rep:
        card('📨 Reported to HR',
             f"Severity <b>{rep['severity']}</b> ({rep['severity_score']}/100) at {rep['time'].replace('T', ' ')} · "
             f"delivery: {rep['delivery']}", 'purple')
    with st.expander('➕ Add this recording to the training data (improves the model)'):
        label = st.radio('How did you actually feel?', ['fatigue', 'normal'], horizontal=True,
                         format_func=lambda x: {'fatigue': 'Fatigued / tired', 'normal': 'Normal / rested'}[x])
        kss = st.slider('Sleepiness (Karolinska Sleepiness Scale): 1 = extremely alert … 9 = fighting sleep', 1, 9, 7 if fatigued else 3)
        note = st.text_input('Notes (optional, e.g. "after 3 h of work", "sick")')
        ok = st.checkbox('I agree that this recording is stored and used to train the model')
        if st.button('Add to training data', disabled=not ok):
            fpath = contribute(data, name, uid, label, kss, note)
            st.success(f'Saved {fpath.relative_to(ROOT)} and added to data_manifest.csv. Retrain under **Model**.')
# ---------------- history ----------------
elif page == 'History':
    section('History', 'past checks per person')
    if not users():
        st.info('No users yet.'); footer(); st.stop()
    uid = st.selectbox('Person', users())
    hist = USERS / uid / 'history.csv'
    if not hist.exists():
        st.info('No checks yet for this person.'); footer(); st.stop()
    h = pd.read_csv(hist)
    c = st.columns(3)
    stat(c[0], 'Checks', len(h))
    stat(c[1], 'Fatigue', int((h.result == 'FATIGUE').sum()))
    stat(c[2], 'Last check', h.time.iloc[-1][:16].replace('T', ' '), h.result.iloc[-1])
    fig = go.Figure(go.Scatter(x=h.time, y=h.p_fatigue, mode='lines+markers', line=dict(color='#b9cdd3'),
                               marker=dict(color=np.where(h.result == 'FATIGUE', CORAL, TEAL), size=11)))
    fig.add_hline(y=model.threshold if model else .5, line_dash='dash', line_color=CORAL, annotation_text='fatigue')
    fig.update_layout(yaxis=dict(range=[0, 1], title='P(fatigue)'))
    st.plotly_chart(style_fig(fig, 300), width='stretch')
    st.dataframe(h.iloc[::-1], hide_index=True, width='stretch')
    if st.button('🗑 Clear history (all people)'):
        clear_history()
        st.rerun()
# ---------------- HR reports ----------------
elif page == 'HR reports':
    section('Workload reports for HR', 'one report per fatigue alert')
    st.caption('Severity by P(fatigue): ' + ', '.join(f"{n} ≥ {lo:.1f}" for lo, n in SEVERITY_BANDS if lo)
               + ', HIGH below that (from 0.8; lower scores count as normal). Score 0–100: 0 = just at the alarm threshold, 100 = certain. '
               'Only summaries are reported, never raw ECG.')
    rep = load_reports()
    if rep.empty:
        st.info('No fatigue alerts yet.'); footer(); st.stop()
    who = st.multiselect('Person', sorted(rep.person.unique()), default=sorted(rep.person.unique()))
    rep = rep[rep.person.isin(who)].sort_values('time', ascending=False)
    c = st.columns(3)
    stat(c[0], 'Alerts', len(rep))
    for i, s in enumerate(['SEVERE', 'HIGH'], 1):
        stat(c[i], s.title(), int((rep.severity == s).sum()))
    sev_col = {'SEVERE': '#B3261E', 'HIGH': CORAL, 'MODERATE': AMBER}  # MODERATE only in old reports
    fig = go.Figure(go.Scatter(x=rep.time, y=rep.severity_score, mode='markers', text=rep.person,
                               marker=dict(size=13, color=rep.severity.map(sev_col), line=dict(width=1, color='#fff'))))
    fig.update_layout(yaxis=dict(range=[0, 105], title='severity (0–100)'))
    st.plotly_chart(style_fig(fig, 280), width='stretch')
    st.dataframe(rep[['time', 'person', 'severity', 'severity_score', 'p_fatigue', 'hr_now', 'hr_baseline', 'rmssd_now',
                      'rmssd_baseline', 'signal_reliable', 'reasons', 'delivery']], hide_index=True, width='stretch')
    b1, b2 = st.columns([1, 1])
    b1.download_button('⬇️ Download CSV', rep.to_csv(index=False), 'hr_reports.csv', 'text/csv')
    if b2.button('🗑 Clear HR reports'):
        REPORTS.unlink(missing_ok=True)
        st.rerun()
    with st.expander('Message text of the latest alert'):
        st.code(report_text(rep.iloc[0].to_dict()), language=None)
# ---------------- model ----------------
else:
    section('Model & retraining', 'personal-shift model')
    if model is None:
        st.error('No model trained yet.')
    else:
        m, tr, ev = model.meta, model.meta['training'], model.meta.get('evaluation', {})
        names = ', '.join(LABELS[k][0] for k in m['features'])
        c = st.columns(2, gap='large')
        with c[0]:
            card('What it learns',
                 "The <b>change</b> from a person's own rested baseline that indicates fatigue, measured as the percent "
                 f"change (log ratio) of <b>{names}</b>. Training pairs come from the <b>same person</b>:"
                 f"<ul><li><b>fatigue:</b> each fatigue recording vs. each of that person's normal recordings "
                 f"({tr['fatigue_pairs']} pairs)</li>"
                 f"<li><b>no change:</b> normal vs. another normal of the same person, and first vs. second half of "
                 f"long normal recordings ({tr['no_change_pairs']} pairs)</li></ul>")
        with c[1]:
            card('Training data',
                 f"Standardised logistic regression trained {m['trained_at'][:16].replace('T', ' ')} on "
                 f"<b>{tr['recordings']} recordings from {tr['people']} people</b> ({tr['normal_recordings']} normal, "
                 f"{tr['fatigue_recordings']} fatigue). FATIGUE when <b>P ≥ {model.threshold:.2f}</b>.", 'navy')
        how_model_a_works(model)
        if ev:
            section('Details', 'per comparison (each hidden recording vs. each rested baseline)')
            c1, c2 = st.columns([1, 2])
            c1.dataframe(pd.DataFrame(ev['confusion'], index=['true normal', 'true fatigue'],
                                      columns=['predicted normal', 'predicted fatigue']), width='stretch')
            coef = pd.Series(m['coef'], [LABELS[k][0] for k in m['features']]).sort_values()
            fig = go.Figure(go.Bar(x=coef.values, y=coef.index, orientation='h',
                                   marker_color=np.where(coef.values > 0, CORAL, TEAL)))
            c2.plotly_chart(style_fig(fig, 260, 'What the model uses (+ : an increase points to fatigue)'), width='stretch')
            with st.expander('All held-out comparisons'):
                st.dataframe(pd.DataFrame(ev['details']), hide_index=True, width='stretch')
        fatigue_types()
        card('Limitations',
             'Two subjects and a small number of usable recordings. The model detects a shift away from rest '
             '(HR up, HRV down) and cannot yet separate fatigue from illness, exercise or stress.', 'amber')
    section('Training data', 'data_manifest.csv')
    man = pd.read_csv(MANIFEST)
    st.dataframe(man, hide_index=True, width='stretch')
    st.caption(f'{len(man)} recordings in data_manifest.csv. Add rows (file, person, condition) or contribute from the ② page, then retrain.')
    if st.button('🔁 Retrain model now', type='primary'):
        from train_delta_model import train
        with st.spinner('Preprocessing every recording and retraining…'):
            new = train(verbose=False)
        e = new.get('evaluation', {})
        st.success(f"Retrained on {new['training']['recordings']} recordings. "
                   + (f"Held-out: {e['correct']}/{e['comparisons']} correct." if e else ''))
        st.rerun()
footer()
