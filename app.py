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
    col.markdown(
        f"<div class='bb-verdict'><div class='lv' style='color:{'#fff' if fatigued else GREEN}'>"
        f"{'FATIGUE' if fatigued else 'NORMAL'}</div>"
        f"<svg viewBox='0 0 220 38' preserveAspectRatio='none'><polyline points='{pts}' fill='none' stroke='{color}' "
        f"stroke-width='2.5'/></svg>"
        f"<span class='pill' style='background:{AMBER if fatigued else GREEN}'>{'Take a breath' if fatigued else 'Keep going'}</span>"
        f"<div class='p'>P(fatigue) = {p:.2f} · fatigue at ≥ {threshold:.2f}</div></div>", unsafe_allow_html=True)


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


@st.cache_resource
def clear_session_logs_on_start():
    """Runs once per app launch (`streamlit run app.py`): every session starts with no HR reports and
    no check history. Saved baselines are kept."""
    REPORTS.unlink(missing_ok=True)
    clear_history()
    return True


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


# ---------------- sidebar ----------------
html(CSS)
clear_session_logs_on_start()
PAGES = ['① Input baseline', '② Input recording', 'History', 'HR reports', 'Model']
with st.sidebar:
    html("<div class='bb-side-kicker'>UTD HACKEDGE 2026</div><div class='bb-side-brand'>🫀 BreathBreak</div>")
    page = st.radio('Go to', PAGES, label_visibility='collapsed')
    model = get_model()
    if model:
        ev = model.meta.get('evaluation', {})
        html(f"<div class='bb-side-box'><b>Model</b><br>Trained {model.meta['trained_at'][:16].replace('T', ' ')} on "
             f"{model.meta['training']['recordings']} recordings from {model.meta['training']['people']} people."
             + (f"<br>Held-out: <b>{ev['correct']}/{ev['comparisons']}</b> correct." if ev else '') + "</div>")
    else:
        st.error('No model yet. Run `python train_delta_model.py`.')
    html("<div class='bb-side-box'><b>How it works</b><br>① Everyone records a <b>rested baseline</b> once.<br>"
         "② New recordings are compared with <b>their own</b> baseline.<br>HR up and HRV down → <b>fatigue</b> → "
         "breathing break.</div>")

hero(page)

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
