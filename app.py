"""BreathBreak — personal-baseline & zero-baseline physiological fatigue check (web app).

Run:  python -m streamlit run app.py
  ① Record baseline  : upload a rested recording once per person -> saved as their baseline
  ② Check for fatigue: compare a new recording with their saved baseline OR run a Zero-Baseline Quick Check on a new person
  History            : past checks per person
  HR reports         : serious-fatigue alerts logged for HR
  Model & retraining : how both models (Personal-Baseline Delta & Zero-Baseline Single-Recording) work and retrain button
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fatigue_pipeline import (
    FEATURES, FS_ECG, LABELS, MODEL_PATH, PRE, ROOT, SINGLE_FEATURES,
    DeltaModel, analyze, beat_similarity, ensure_full_features,
    load_recording, quality_warnings,
)
from hr_report import SEVERITY_BANDS, build_report, load_reports, report_text, send_report

USERS = ROOT / 'user_data'
CONTRIB = ROOT / 'data' / 'contributed'
MANIFEST = ROOT / 'data_manifest.csv'
st.set_page_config(page_title='BreathBreak — Fatigue Check', page_icon='🫀', layout='wide')


# ---------------- helpers ----------------
@st.cache_data(show_spinner='Preprocessing ECG & PPG (filter → artifact mask → R-peaks → HRV & PPG features)…')
def run_analysis(data: bytes):
    ecg, opt = load_recording(data)
    return analyze(ecg, opt)


def get_model():
    return DeltaModel(MODEL_PATH) if MODEL_PATH.exists() else None


def safe_id(s):
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in s.strip()) or 'user'


def users():
    return sorted(p.name for p in USERS.glob('*') if (p / 'baseline.json').exists()) if USERS.exists() else []


def load_baseline(uid):
    b = json.loads((USERS / uid / 'baseline.json').read_text())
    b['features'] = ensure_full_features(b['features'], uid=uid)
    return b


def load_baseline_analysis(uid=None, base=None):
    """Load full analyzed ECG + PPG signals for the user's saved baseline (or a default rested reference in Zero-Baseline mode)."""
    if uid and (USERS / uid / 'baseline.csv').exists():
        return run_analysis((USERS / uid / 'baseline.csv').read_bytes()), f"{uid}'s original baseline ({base['source'] if base else 'baseline.csv'})"
    if base and base.get('source'):
        for cand in (ROOT / 'data' / base['source'], ROOT / base['source']):
            if cand.exists():
                return run_analysis(cand.read_bytes()), f"{uid}'s original baseline ({base['source']})"
    for ref_rel in ('normal/personC_rested_baseline.csv', 'normal/sixth_attempt.csv', 'normal/first_attempt.csv'):
        for cand in (ROOT / 'data' / ref_rel, ROOT / ref_rel):
            if cand.exists():
                return run_analysis(cand.read_bytes()), f"Healthy rested reference ({ref_rel})"
    return None, None


def samples():
    out = {}
    for folder in ('normal', 'fatigue'):
        for base_dir in (ROOT / 'data' / folder, ROOT / folder):
            if base_dir.exists():
                for p in sorted(base_dir.glob('*.csv')):
                    out.setdefault(f'{folder}/{p.name}', p)
    return out


def recording_input(key):
    up = st.file_uploader('TI AFE4950 export (.csv)', type='csv', key=f'up_{key}')
    if up is not None:
        return up.getvalue(), up.name
    opts = samples()
    pick = st.selectbox('…or use a sample recording', ['—'] + list(opts), key=f'sample_{key}')
    if pick != '—':
        return opts[pick].read_bytes(), pick
    return None, None


def show_warnings(res, role):
    for level, msg in quality_warnings(res, role):
        (st.error if level == 'error' else st.warning)(msg)


def ecg_figure(res, title, seconds=10):
    t, x, rp = res['t'], res['ecg_clean'], res['rpeaks']
    a = res['keep'][0]; w = (t >= a) & (t <= a + seconds)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t[w], y=x[w], line=dict(width=1, color='#2a7fb8'), name='clean ECG'))
    r = rp[(t[rp] >= a) & (t[rp] <= a + seconds)]
    fig.add_trace(go.Scatter(x=t[r], y=x[r], mode='markers', marker=dict(color='orange', size=8), name='R-peak'))
    fig.update_layout(height=230, title=title, margin=dict(l=10, r=10, t=35, b=10), xaxis_title='time (s)',
                      yaxis_title='mV', showlegend=False)
    return fig


def ppg_figure(res, title='Optical (PPG) pulse wave — first 10 s of stable part', seconds=10):
    opt = res.get('optical', {})
    if not opt.get('present'):
        return None
    t_o, x_o, pk = opt['t_opt'], opt['ppg_clean'], opt['ppg_peaks']
    a = res['keep'][0]; w = (t_o >= a) & (t_o <= a + seconds)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_o[w], y=x_o[w], line=dict(width=1.2, color='#e67e22'), name='clean PPG'))
    p = pk[(t_o[pk] >= a) & (t_o[pk] <= a + seconds)]
    fig.add_trace(go.Scatter(x=t_o[p], y=x_o[p], mode='markers', marker=dict(color='#2c3e50', size=7), name='PPG peak'))
    pat_str = f" · median ECG→PPG PAT = {opt['features']['PAT_ms']:.0f} ms" if opt['features'].get('PAT_ms') else ''
    fig.update_layout(height=210, title=title + pat_str, margin=dict(l=10, r=10, t=35, b=10),
                      xaxis_title='time (s)', yaxis_title='PPG (mV)', showlegend=False)
    return fig


def ecg_comparison_figure(base_res, new_res, base_label, new_col='#c9302c', seconds=10):
    """Overlay original/baseline ECG and new check ECG over the first 10 s of each recording's stable part."""
    fig = go.Figure()
    if base_res is not None:
        tb, xb, rpb = base_res['t'], base_res['ecg_clean'], base_res['rpeaks']
        ab = base_res['keep'][0]; wb = (tb >= ab) & (tb <= ab + seconds)
        rb = rpb[(tb[rpb] >= ab) & (tb[rpb] <= ab + seconds)]
        fig.add_trace(go.Scatter(
            x=tb[wb] - ab, y=xb[wb],
            line=dict(width=1.3, color='#2e8b57', dash='solid'),
            opacity=0.75,
            name=f"Original / Baseline ECG ({base_res['features']['HR']:.0f} bpm, {len(rb)} beats in 10s)",
        ))
        fig.add_trace(go.Scatter(
            x=tb[rb] - ab, y=xb[rb],
            mode='markers', marker=dict(color='#2e8b57', symbol='circle-open', size=9, line=dict(width=2)),
            name='Original R-peaks',
        ))
    tn, xn, rpn = new_res['t'], new_res['ecg_clean'], new_res['rpeaks']
    an = new_res['keep'][0]; wn = (tn >= an) & (tn <= an + seconds)
    rn = rpn[(tn[rpn] >= an) & (tn[rpn] <= an + seconds)]
    fig.add_trace(go.Scatter(
        x=tn[wn] - an, y=xn[wn],
        line=dict(width=1.4, color=new_col),
        name=f"New Recording ECG ({new_res['features']['HR']:.0f} bpm, {len(rn)} beats in 10s)",
    ))
    fig.add_trace(go.Scatter(
        x=tn[rn] - an, y=xn[rn],
        mode='markers', marker=dict(color='orange', size=8),
        name='New R-peaks',
    ))
    fig.update_layout(
        height=270,
        title=f"ECG Signal Comparison — First {seconds} s of Stable Part (Original Baseline vs. New Recording)",
        margin=dict(l=10, r=10, t=38, b=10),
        xaxis_title='seconds into stable part (0–10 s aligned)',
        yaxis_title='ECG (mV)',
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1),
    )
    return fig


def ppg_comparison_figure(base_res, new_res, new_col='#e67e22', seconds=10):
    """Overlay original/baseline Optical PPG and new check Optical PPG over the first 10 s of the stable part."""
    b_opt = (base_res or {}).get('optical', {})
    n_opt = new_res.get('optical', {})
    if not b_opt.get('present') and not n_opt.get('present'):
        return None
    fig = go.Figure()
    title_parts = []
    if b_opt.get('present'):
        t_b, x_b, pk_b = b_opt['t_opt'], b_opt['ppg_clean'], b_opt['ppg_peaks']
        ab = base_res['keep'][0]; wb = (t_b >= ab) & (t_b <= ab + seconds)
        pb = pk_b[(t_b[pk_b] >= ab) & (t_b[pk_b] <= ab + seconds)]
        pat_b = f", PAT {b_opt['features']['PAT_ms']:.0f} ms" if b_opt['features'].get('PAT_ms') else ''
        fig.add_trace(go.Scatter(
            x=t_b[wb] - ab, y=x_b[wb],
            line=dict(width=1.6, color='#2e8b57', dash='dot'),
            name=f"Original / Baseline PPG ({len(pb)} pulses{pat_b})",
        ))
        fig.add_trace(go.Scatter(
            x=t_b[pb] - ab, y=x_b[pb],
            mode='markers', marker=dict(color='#2e8b57', symbol='circle-open', size=8, line=dict(width=2)),
            name='Original PPG peaks',
        ))
        if b_opt['features'].get('PAT_ms'):
            title_parts.append(f"Original PAT = {b_opt['features']['PAT_ms']:.0f} ms")
    if n_opt.get('present'):
        t_n, x_n, pk_n = n_opt['t_opt'], n_opt['ppg_clean'], n_opt['ppg_peaks']
        an = new_res['keep'][0]; wn = (t_n >= an) & (t_n <= an + seconds)
        pn = pk_n[(t_n[pk_n] >= an) & (t_n[pk_n] <= an + seconds)]
        pat_n = f", PAT {n_opt['features']['PAT_ms']:.0f} ms" if n_opt['features'].get('PAT_ms') else ''
        fig.add_trace(go.Scatter(
            x=t_n[wn] - an, y=x_n[wn],
            line=dict(width=1.8, color=new_col),
            name=f"New Recording PPG ({len(pn)} pulses{pat_n})",
        ))
        fig.add_trace(go.Scatter(
            x=t_n[pn] - an, y=x_n[pn],
            mode='markers', marker=dict(color='#2c3e50', size=7),
            name='New PPG peaks',
        ))
        if n_opt['features'].get('PAT_ms'):
            title_parts.append(f"New PAT = {n_opt['features']['PAT_ms']:.0f} ms")
    pat_summary = f" · ({' → '.join(title_parts)})" if title_parts else ''
    fig.update_layout(
        height=250,
        title=f"Optical (PPG) Signal Comparison — First {seconds} s of Stable Part{pat_summary}",
        margin=dict(l=10, r=10, t=38, b=10),
        xaxis_title='seconds into stable part (0–10 s aligned)',
        yaxis_title='PPG (mV)',
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1),
    )
    return fig


def overview_figure(res, title):
    t, x = res['t'], res['ecg_clean']
    step = 5
    fig = go.Figure(go.Scattergl(x=t[::step], y=x[::step], line=dict(width=.6, color='#2a7fb8')))
    a, b = res['keep']
    fig.add_vrect(x0=0, x1=a, fillcolor='grey', opacity=.3, line_width=0)
    fig.add_vrect(x0=b, x1=t[-1], fillcolor='grey', opacity=.3, line_width=0)
    fig.update_layout(height=180, title=title, margin=dict(l=10, r=10, t=35, b=10), xaxis_title='time (s)', yaxis_title='mV')
    return fig


def quality_table(res):
    q, f = res['quality'], res['features'] or {}
    rows = [
        ('Recording length', f"{q['duration_s']:.1f} s"),
        ('Stable part used', f"{q['stable_start']:.0f}–{q['stable_end']:.0f} s ({q['stable_s']:.0f} s)"),
        ('Clean beats', str(q['clean_beats'])),
        ('Masked (motion / saturation)', f"{q['masked_pct']:.1f} %"),
        ('Optical (PPG) channel', f"working ({q['ppg_peaks']} peaks)" if q['optical_present'] else f"empty ({q['optical_pp_mV']:.2f} mV)"),
        ('Heart rate', f"{f['HR']:.1f} bpm" if f else '–'),
        ('RMSSD (vagal HRV)', f"{f['RMSSD']:.1f} ms" if f else '–'),
        ('Poincaré SD1/SD2', f"{f['SD1_SD2']:.2f}" if f and 'SD1_SD2' in f else '–'),
        ('LF/HF autonomic ratio', f"{f['LF_HF']:.2f} (HF {f['HF_nu']:.0f} %)" if f and 'LF_HF' in f else '–'),
    ]
    if q.get('pat_ms') is not None:
        rows.append(('ECG→PPG Pulse Arrival Time (PAT)', f"{q['pat_ms']:.0f} ± {q['pat_sd_ms']:.0f} ms"))
    return pd.DataFrame(rows, columns=['Check', 'Result'])


def contribute(data, name, uid, condition, kss, note):
    d = CONTRIB / uid; d.mkdir(parents=True, exist_ok=True)
    fname = f"{datetime.now():%Y%m%d_%H%M%S}_{condition}.csv"
    (d / fname).write_bytes(data)
    row = pd.DataFrame([dict(file=str((d / fname).relative_to(ROOT)), person=uid, condition=condition,
                             kss=kss, notes=f'contributed via app; source {name}; {note}'.strip('; '))])
    row.to_csv(MANIFEST, mode='a', header=False, index=False)
    return d / fname


# ---------------- sidebar ----------------
st.sidebar.title('🫀 BreathBreak')
page = st.sidebar.radio('Go to', ['① Record baseline', '② Check for fatigue', 'History', 'HR reports', 'Model & retraining'])
model = get_model()
if model:
    ev = model.meta.get('evaluation', {})
    sev = (model.single_meta or {}).get('evaluation', {})
    st.sidebar.caption(
        f"Model trained {model.meta['trained_at'][:16].replace('T', ' ')} on "
        f"{model.meta['training']['recordings']} recordings / {model.meta['training']['people']} people. "
        + (f"Personal-baseline LORO: {ev['correct']}/{ev['comparisons']} ({ev['accuracy']:.0%}). " if ev else '')
        + (f"Zero-baseline LORO: {sev['clean_correct']}/{sev['clean_recordings']} ({sev['clean_accuracy']:.0%})." if sev else '')
    )
else:
    st.sidebar.error('No model yet — run `python train_delta_model.py`.')
st.sidebar.info('Supports **two modes**:\n'
                '1. **Personal-Baseline Check:** compares a person against their own saved rested baseline.\n'
                '2. **New Person Quick Check (Zero-Baseline):** classifies a single recording using subject-invariant HRV & waveform ratios.')

# ---------------- ① baseline ----------------
if page == '① Record baseline':
    st.title('① Record your baseline')
    st.markdown('Record while **rested and healthy**: sit quietly for 5 minutes, then capture **3–5 minutes** without '
                'talking or moving. Check that the ECG and optical (PPG) waves look clean in the TI GUI, then export '
                '`AFE4950_CAPTURED_DATA.csv` (manual §6.3) and upload it here. **Mark the electrode spots** so later '
                'recordings use the same placement.')
    uid = safe_id(st.text_input('Name or ID', ''))
    data, name = recording_input('baseline')
    if data:
        res = run_analysis(data)
        show_warnings(res, 'baseline')
        c1, c2 = st.columns([1, 2])
        c1.dataframe(quality_table(res), hide_index=True, width='stretch')
        c2.plotly_chart(overview_figure(res, 'Whole recording (grey = trimmed unstable start/end)'), width='stretch')
        c2.plotly_chart(ecg_figure(res, 'First 10 s of the stable part — Clean ECG & R-peaks'), width='stretch')
        pfig = ppg_figure(res)
        if pfig is not None:
            c2.plotly_chart(pfig, width='stretch')
        if res['usable']:
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
                st.success(f'Baseline saved for **{uid}**. Next time, go to **② Check for fatigue**.')
            if not uid or uid == 'user':
                st.caption('Enter a name or ID to save.')

# ---------------- ② check ----------------
elif page == '② Check for fatigue':
    st.title('② Check for fatigue')
    if model is None:
        st.error('No trained model. Run `python train_delta_model.py`.'); st.stop()

    has_users = bool(users())
    modes = ['Compare with saved personal baseline (Delta Mode)', 'New person / Single recording Quick Check (Zero-Baseline Mode)']
    mode_pick = st.radio('Classification mode', modes if has_users else modes[1:], horizontal=True)
    is_single = mode_pick.startswith('New person')

    if not is_single:
        uid = st.selectbox('Person', users())
        base = load_baseline(uid)
        st.caption(f"Baseline: {base['source']} (saved {base['saved_at'][:16].replace('T', ' ')}) — "
                   f"HR {base['features']['HR']:.0f} bpm, RMSSD {base['features']['RMSSD']:.0f} ms, "
                   f"SD1/SD2 {base['features']['SD1_SD2']:.2f}")
    else:
        uid = safe_id(st.text_input('Person Name or ID (optional, for logging)', 'new_person'))
        base = None
        st.caption('Zero-Baseline Mode: classifies a brand-new person from a single recording using placement-invariant physiological HRV & rate-corrected timing features.')

    data, name = recording_input('check')
    if not data:
        st.stop()
    res = run_analysis(data)
    show_warnings(res, 'check')
    if not res['usable']:
        st.stop()

    if not is_single:
        sim = beat_similarity(np.array(base['median_beat']), res['median_beat'])
        p, contrib = model.predict(res['features'], base['features'], beat_match=sim)
        lvl = model.level(p, mode='delta')
        thr_used = model.threshold
    else:
        sim = 1.0
        p, contrib = model.predict_single(res['features'])
        lvl = model.level(p, mode='single')
        thr_used = (model.single_meta or {}).get('threshold', model.threshold)

    fatigued = lvl == 'fatigue'
    col = {'fatigue': '#c9302c', 'watch': '#d98c00', 'normal': '#2e8b57'}[lvl]
    verdict = {'fatigue': 'FATIGUE', 'watch': 'WATCH', 'normal': 'NORMAL'}[lvl]
    c = st.columns([1.3, 1, 1, 1])
    c[0].markdown(f"<div style='background:{col};color:#fff;border-radius:10px;padding:18px;text-align:center'>"
                  f"<div style='font-size:28px;font-weight:700'>{verdict}</div>"
                  f"<div>P(fatigue) = {p:.2f} · alarm at ≥ {thr_used:.2f}</div></div>", unsafe_allow_html=True)

    ref_feats = base['features'] if base else (model.single_meta or {}).get('rested_reference', res['features'])
    ref_label = 'vs baseline' if base else 'vs rested norm'
    c[1].metric('Heart rate', f"{res['features']['HR']:.0f} bpm", f"{res['features']['HR'] - ref_feats['HR']:+.0f} {ref_label}", delta_color='off')
    c[2].metric('RMSSD (vagal HRV)', f"{res['features']['RMSSD']:.1f} ms", f"{res['features']['RMSSD'] - ref_feats['RMSSD']:+.1f} {ref_label}", delta_color='off')
    if not is_single:
        c[3].metric('Beat-shape match', f'{sim:.3f}', help='1.000 = identical average heartbeat shape. Below 0.98, R_mV is automatically muted so electrode shift does not cause false alarms.')
        if sim < 0.98:
            st.warning(f'The heartbeat shape differs from baseline (match {sim:.3f}). Electrodes were probably placed differently — '
                       'the model automatically muted R-peak voltage (`R_mV`) and judged purely on HRV and normalized timing features.')
    else:
        c[3].metric('Poincaré SD1/SD2', f"{res['features']['SD1_SD2']:.2f}", f"{res['features']['SD1_SD2'] - ref_feats['SD1_SD2']:+.2f} {ref_label}", delta_color='off')

    if fatigued:
        st.error('Serious fatigue signs detected. Take a **breathing break** (e.g. 2 min of slow breathing, 6 breaths/min) and re-check later.')
    elif lvl == 'watch':
        st.info('Mild autonomic shift toward fatigue, below the serious-fatigue alarm threshold. Re-check later if you feel tired.')

    st.subheader('Why: physiological feature breakdown')
    t = contrib.copy()
    t['Measure'] = [f"{LABELS[k][0]} ({LABELS[k][1]})" if LABELS[k][1] else LABELS[k][0] for k in t.feature]
    t['Direction'] = np.where(t.push_to_fatigue > 0, '→ fatigue', np.where(t.push_to_fatigue < 0, '→ normal', 'muted (placement)'))
    base_col_name = 'Baseline' if base else 'Rested Ref'
    st.dataframe(t[['Measure', 'baseline', 'new', 'change', 'pct_change', 'push_to_fatigue', 'Direction']].rename(columns={
        'baseline': base_col_name, 'new': 'Now', 'change': 'Change', 'pct_change': 'Change %', 'push_to_fatigue': 'Effect on decision'}).round(2),
        hide_index=True, width='stretch')

    base_res, base_wave_label = load_baseline_analysis(uid if not is_single else None, base)

    st.subheader('Visual Comparison: Original Baseline vs. New Recording')
    if base_wave_label:
        st.caption(f"Comparing **New recording (`{name}`)** against **{base_wave_label}** (green = original/rested, {verdict.lower()} color = new check).")

    g1, g2 = st.columns(2)
    fig = go.Figure()
    if base:
        fig.add_trace(go.Scatter(x=np.array(base['hr_t']) - base['hr_t'][0], y=base['hr'], name='original / baseline', line=dict(color='#2e8b57', width=2)))
    elif base_res is not None:
        fig.add_trace(go.Scatter(x=base_res['t_rr'] - base_res['t_rr'][0], y=60 / base_res['rr'], name='rested reference', line=dict(color='#2e8b57', width=2)))
    fig.add_trace(go.Scatter(x=res['t_rr'] - res['t_rr'][0], y=60 / res['rr'], name='new recording (now)', line=dict(color=col, width=2)))
    fig.update_layout(height=300, title='Beat-to-beat heart rate (Original vs. Now)', xaxis_title='seconds into stable part', yaxis_title='bpm',
                      margin=dict(l=10, r=10, t=35, b=10))
    g1.plotly_chart(fig, width='stretch')

    mb = np.array(base['median_beat']) if base else (base_res['median_beat'] if base_res is not None else res['median_beat'])
    mn = res['median_beat']
    tb = (np.arange(len(mb)) - int(PRE*FS_ECG)) / FS_ECG * 1000
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tb, y=mb, name='original / baseline', line=dict(color='#2e8b57', width=2)))
    fig.add_trace(go.Scatter(x=tb[:len(mn)], y=mn, name='new recording (now)', line=dict(color=col, width=2)))
    fig.update_layout(height=300, title='Average heartbeat shape (P–QRS–T Waveform: Original vs. Now)', xaxis_title='ms from R-peak', yaxis_title='mV',
                      margin=dict(l=10, r=10, t=35, b=10))
    g2.plotly_chart(fig, width='stretch')

    # Overlaid 10 s ECG comparison (Original vs. New)
    st.plotly_chart(ecg_comparison_figure(base_res, res, base_wave_label, new_col=col, seconds=10), width='stretch')

    # Overlaid 10 s Optical (PPG) comparison (Original vs. New)
    pfig_comp = ppg_comparison_figure(base_res, res, new_col='#e67e22', seconds=10)
    if pfig_comp is not None:
        st.plotly_chart(pfig_comp, width='stretch')

    with st.expander('🔍 View side-by-side / separate 10 s ECG & PPG traces (Original vs. New)'):
        s1, s2 = st.columns(2)
        if base_res is not None:
            s1.plotly_chart(ecg_figure(base_res, f"Original / Baseline ECG — first 10 s ({base_res['features']['HR']:.0f} bpm)"), width='stretch')
            bp_fig = ppg_figure(base_res, 'Original / Baseline Optical PPG — first 10 s')
            if bp_fig is not None:
                s1.plotly_chart(bp_fig, width='stretch')
        s2.plotly_chart(ecg_figure(res, f"New Recording ECG — first 10 s ({res['features']['HR']:.0f} bpm)"), width='stretch')
        np_fig = ppg_figure(res, 'New Recording Optical PPG — first 10 s')
        if np_fig is not None:
            s2.plotly_chart(np_fig, width='stretch')

    user_dir = USERS / uid
    user_dir.mkdir(parents=True, exist_ok=True)
    hist = user_dir / 'history.csv'
    key = f'logged_{uid}_{is_single}_{hash(data)}'
    if key not in st.session_state:
        pd.DataFrame([dict(time=datetime.now().isoformat(timespec='seconds'), recording=name, result=verdict,
                           p_fatigue=round(p, 3), hr=round(res['features']['HR'], 1), rmssd=round(res['features']['RMSSD'], 1),
                           beat_match=round(sim, 3))]).to_csv(hist, mode='a', header=not hist.exists(), index=False)
        st.session_state[key] = True
        if fatigued:
            rep = send_report(build_report(uid, name, p, thr_used, res['features'], ref_feats, contrib, sim))
            st.session_state[f'hr_{key}'] = rep
    rep = st.session_state.get(f'hr_{key}')
    if rep:
        st.warning(f"📨 Reported to HR: **{rep['severity']}** (severity {rep['severity_score']}/100) at "
                   f"{rep['time'].replace('T', ' ')} · delivery: {rep['delivery']}")

    with st.expander('➕ Add this recording to the training data (improves the model)'):
        label = st.radio('How did you actually feel?', ['fatigue', 'normal'], horizontal=True,
                         format_func=lambda x: {'fatigue': 'Fatigued / tired', 'normal': 'Normal / rested'}[x])
        kss = st.slider('Sleepiness (Karolinska Sleepiness Scale): 1 = extremely alert … 9 = fighting sleep', 1, 9, 7 if fatigued else 3)
        note = st.text_input('Notes (optional, e.g. "after 3 h of work", "sick")')
        ok = st.checkbox('I agree that this recording is stored and used to train the model')
        if st.button('Add to training data', disabled=not ok):
            f = contribute(data, name, uid, label, kss, note)
            st.success(f'Saved {f.relative_to(ROOT)} and added to data_manifest.csv. Retrain under **Model & retraining**.')

# ---------------- history ----------------
elif page == 'History':
    st.title('History')
    all_users = sorted(p.name for p in USERS.glob('*') if (p / 'history.csv').exists()) if USERS.exists() else []
    if not all_users:
        st.info('No check history yet.'); st.stop()
    uid = st.selectbox('Person', all_users)
    hist = USERS / uid / 'history.csv'
    h = pd.read_csv(hist)
    fig = go.Figure(go.Scatter(x=h.time, y=h.p_fatigue, mode='lines+markers',
                               marker=dict(color=h.result.map({'FATIGUE': '#c9302c', 'WATCH': '#d98c00'}).fillna('#2e8b57'), size=10)))
    fig.add_hline(y=model.threshold if model else .5, line_dash='dash', annotation_text='fatigue alarm')
    if model and model.watch_threshold < model.threshold:
        fig.add_hline(y=model.watch_threshold, line_dash='dot', annotation_text='watch')
    fig.update_layout(height=300, yaxis=dict(range=[0, 1], title='P(fatigue)'), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width='stretch')
    st.dataframe(h.iloc[::-1], hide_index=True, width='stretch')

# ---------------- HR reports ----------------
elif page == 'HR reports':
    st.title('HR reports')
    st.caption('One report is created whenever a check reaches the **serious-fatigue** alarm. Severity by P(fatigue): '
               + ', '.join(f"{n} ≥ {lo:.1f}" for lo, n in SEVERITY_BANDS if lo) + ', MODERATE below that. '
               'Score 0–100: 0 = just at the alarm threshold, 100 = certain. Only summaries are reported, never raw ECG.')
    rep = load_reports()
    if rep.empty:
        st.info('No fatigue alerts yet.'); st.stop()
    who = st.multiselect('Person', sorted(rep.person.unique()), default=sorted(rep.person.unique()))
    rep = rep[rep.person.isin(who)].sort_values('time', ascending=False)
    c = st.columns(4)
    c[0].metric('Alerts', len(rep))
    for i, s in enumerate(['SEVERE', 'HIGH', 'MODERATE'], 1):
        c[i].metric(s.title(), int((rep.severity == s).sum()))
    sev_col = {'SEVERE': '#8b0000', 'HIGH': '#c9302c', 'MODERATE': '#d98c00'}
    fig = go.Figure(go.Scatter(x=rep.time, y=rep.severity_score, mode='markers', text=rep.person,
                               marker=dict(size=12, color=rep.severity.map(sev_col))))
    fig.update_layout(height=280, yaxis=dict(range=[0, 105], title='severity (0–100)'), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width='stretch')
    st.dataframe(rep[['time', 'person', 'severity', 'severity_score', 'p_fatigue', 'hr_now', 'hr_baseline', 'rmssd_now',
                      'rmssd_baseline', 'signal_reliable', 'reasons', 'delivery']], hide_index=True, width='stretch')
    st.download_button('⬇️ Download CSV', rep.to_csv(index=False), 'hr_reports.csv', 'text/csv')
    with st.expander('Message text of the latest alert'):
        st.code(report_text(rep.iloc[0].to_dict()), language=None)

# ---------------- model ----------------
else:
    st.title('Model & retraining')
    if model is None:
        st.error('No model trained yet.')
    else:
        m, tr, ev = model.meta, model.meta['training'], model.meta.get('evaluation', {})
        sm = model.single_meta or {}
        sev = sm.get('evaluation', {})
        st.markdown(rf"""
**Dual Physiological Architecture (14 ECG/HRV features + Optical PPG analysis):**
1. **Personal-Baseline Delta Model:** learns the *within-person change* ($\Delta = \text{{new}} - \text{{baseline}}$) across 14 physiological HRV & waveform features (`HR`, `HR_range`, `SDNN`, `RMSSD`, `pNN50`, `CVSD`, `SD1/SD2`, `log10 Stress Index`, `LF/HF`, `HF_nu`, `R_mV`, `T/R ratio`, `RT_ms`, `RTc_ms`), regularized with human autonomic priors and protected against electrode shift (`R_mV` is muted when `beat_similarity < 0.98`).
2. **Zero-Baseline Single-Recording Model:** classifies a brand-new person from a single recording using 10 placement-invariant HRV & rate-corrected timing features (`SINGLE_FEATURES`), while filtering out un-rested baselines (`HR > 95 bpm` and `RMSSD < 15 ms`).

Trained {m['trained_at'].replace('T', ' ')} on **{tr['recordings']} recordings from {tr['people']} people** ({tr['normal_recordings']} normal, {tr['fatigue_recordings']} fatigue).
""")
        if ev:
            st.subheader('1. Personal-Baseline Delta Model — Leave-One-Recording-Out')
            st.markdown(f"Held-out recording vs. each valid rested normal baseline of the same person: "
                        f"**{ev['correct']}/{ev['comparisons']} correct ({ev['accuracy']:.0%})** · "
                        f"serious fatigue caught **{ev['fatigue_caught']}** · normal left alone **{ev['normal_correct']}** "
                        f"(false alarms: **{ev.get('false_alarms', 0)}**).")
            c1, c2 = st.columns([1, 2])
            c1.dataframe(pd.DataFrame(ev['confusion'], index=['true normal', 'true fatigue'],
                                      columns=['predicted normal', 'predicted fatigue']), width='stretch')
            coef = pd.Series(m['coef'], [LABELS[k][0] for k in m['features']]).sort_values()
            fig = go.Figure(go.Bar(x=coef.values, y=coef.index, orientation='h',
                                   marker_color=np.where(coef.values > 0, '#c9302c', '#2e8b57')))
            fig.update_layout(height=360, title='Personal-Baseline Delta Weights (+ : increase points to fatigue)', margin=dict(l=10, r=10, t=35, b=10))
            c2.plotly_chart(fig, width='stretch')
            with st.expander('All held-out personal-baseline comparisons'):
                st.dataframe(pd.DataFrame(ev['details']), hide_index=True, width='stretch')

        if sev:
            st.subheader('2. Zero-Baseline Single-Recording Model (New Person) — Leave-One-Recording-Out')
            st.markdown(f"Classifies a single recording without any prior baseline: **{sev['clean_correct']}/{sev['clean_recordings']} "
                        f"correct ({sev['clean_accuracy']:.0%})** across all valid rested baselines and fatigue recordings "
                        f"(un-rested baselines automatically excluded from reference norm: `{', '.join(sm.get('unrested_baselines_excluded', []))}`).")
            s_coef = pd.Series(sm['coef'], [LABELS[k][0] for k in sm['features']]).sort_values()
            fig2 = go.Figure(go.Bar(x=s_coef.values, y=s_coef.index, orientation='h',
                                    marker_color=np.where(s_coef.values > 0, '#c9302c', '#2e8b57')))
            fig2.update_layout(height=320, title='Zero-Baseline Single-Recording Weights (+ : higher value points to fatigue)', margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig2, width='stretch')
            with st.expander('All held-out single-recording predictions'):
                st.dataframe(pd.DataFrame(sev['details']), hide_index=True, width='stretch')

    st.subheader('Training data')
    man = pd.read_csv(MANIFEST)
    st.dataframe(man, hide_index=True, width='stretch')
    st.caption(f'{len(man)} rows in data_manifest.csv. Add rows (file, person, condition) or contribute from the ② page, then retrain.')
    if st.button('🔁 Retrain model now', type='primary'):
        from train_delta_model import train
        with st.spinner('Preprocessing every recording and retraining both models…'):
            new = train(verbose=False)
        e = new.get('evaluation', {})
        se = new.get('single_model', {}).get('evaluation', {})
        st.success(f"Retrained on {new['training']['recordings']} recordings. "
                   + (f"Personal-baseline LORO: {e['correct']}/{e['comparisons']} ({e['accuracy']:.0%}). " if e else '')
                   + (f"Zero-baseline LORO: {se['clean_correct']}/{se['clean_recordings']} ({se['clean_accuracy']:.0%})." if se else ''))
        st.rerun()
