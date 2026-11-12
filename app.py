"""BreathBreak — personal-baseline fatigue check (web app).

Run:  python3 -m streamlit run app.py
  ① Record baseline  : upload a rested recording once per person -> saved as their baseline
  ② Check for fatigue: upload a new recording -> compared with the baseline -> FATIGUE / NORMAL
  History            : past checks per person
  Model & retraining : how the model was trained, honest held-out results, retrain button
"""
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fatigue_pipeline import (FEATURES, FS_ECG, LABELS, MODEL_PATH, PRE, ROOT, DeltaModel, analyze,
                              beat_similarity, load_recording, quality_warnings)
from hr_report import SEVERITY_BANDS, build_report, load_reports, report_text, send_report

USERS = ROOT / 'user_data'
CONTRIB = ROOT / 'data' / 'contributed'
MANIFEST = ROOT / 'data_manifest.csv'
st.set_page_config(page_title='BreathBreak — Fatigue Check', page_icon='🫀', layout='wide')


# ---------------- helpers ----------------
@st.cache_data(show_spinner='Preprocessing ECG (filter → artifact mask → R-peaks → features)…')
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
    return json.loads((USERS / uid / 'baseline.json').read_text())


def samples():
    out = {}
    for folder in ('normal', 'fatigue'):
        for p in sorted((ROOT / folder).glob('*.csv')):
            out[f'{folder}/{p.name}'] = p
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
    return pd.DataFrame([
        ('Recording length', f"{q['duration_s']:.1f} s"),
        ('Stable part used', f"{q['stable_start']:.0f}–{q['stable_end']:.0f} s ({q['stable_s']:.0f} s)"),
        ('Clean beats', str(q['clean_beats'])),
        ('Masked (motion / saturation)', f"{q['masked_pct']:.1f} %"),
        ('Optical (PPG) channel', 'working' if q['optical_present'] else f"empty ({q['optical_pp_mV']:.2f} mV)"),
        ('Heart rate', f"{f['HR']:.1f} bpm" if f else '–'),
        ('RMSSD', f"{f['RMSSD']:.1f} ms" if f else '–'),
    ], columns=['Check', 'Result'])


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
    st.sidebar.caption(f"Model trained {model.meta['trained_at'][:16].replace('T', ' ')} on "
                       f"{model.meta['training']['recordings']} recordings / {model.meta['training']['people']} people. "
                       + (f"Held-out accuracy {ev['correct']}/{ev['comparisons']} comparisons." if ev else ''))
else:
    st.sidebar.error('No model yet — run `python train_delta_model.py`.')
st.sidebar.info('How it works: everyone records a **rested baseline** once. Later recordings are compared with '
                '**their own** baseline, and the model judges whether the change looks like fatigue.')

# ---------------- ① baseline ----------------
if page == '① Record baseline':
    st.title('① Record your baseline')
    st.markdown('Record while **rested and healthy**: sit quietly for 5 minutes, then capture **3–5 minutes** without '
                'talking or moving. Check that the ECG and optical (PPG) waves look clean in the TI GUI, then export '
                '`AFE4950_CAPTURED_DATA.csv` (manual §6.3) and upload it here. **Mark the electrode spots** so later '
                'recordings use exactly the same placement.')
    uid = safe_id(st.text_input('Name or ID', ''))
    data, name = recording_input('baseline')
    if data:
        res = run_analysis(data)
        show_warnings(res, 'baseline')
        c1, c2 = st.columns([1, 2])
        c1.dataframe(quality_table(res), hide_index=True, width='stretch')
        c2.plotly_chart(overview_figure(res, 'Whole recording (grey = trimmed unstable start/end)'), width='stretch')
        c2.plotly_chart(ecg_figure(res, 'First 10 s of the stable part'), width='stretch')
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
    if not users():
        st.info('No baselines yet. Start with **① Record baseline**.'); st.stop()
    if model is None:
        st.error('No trained model. Run `python train_delta_model.py`.'); st.stop()
    uid = st.selectbox('Person', users())
    base = load_baseline(uid)
    st.caption(f"Baseline: {base['source']} (saved {base['saved_at'][:16].replace('T', ' ')}) — "
               f"HR {base['features']['HR']:.0f} bpm, RMSSD {base['features']['RMSSD']:.0f} ms")
    data, name = recording_input('check')
    if not data:
        st.stop()
    res = run_analysis(data)
    show_warnings(res, 'check')
    if not res['usable']:
        st.stop()

    sim = beat_similarity(np.array(base['median_beat']), res['median_beat'])
    p, contrib = model.predict(res['features'], base['features'])
    lvl = model.level(p)
    fatigued = lvl == 'fatigue'
    col = {'fatigue': '#c9302c', 'watch': '#d98c00', 'normal': '#2e8b57'}[lvl]
    verdict = {'fatigue': 'FATIGUE', 'watch': 'WATCH', 'normal': 'NORMAL'}[lvl]
    c = st.columns([1.3, 1, 1, 1])
    c[0].markdown(f"<div style='background:{col};color:#fff;border-radius:10px;padding:18px;text-align:center'>"
                  f"<div style='font-size:28px;font-weight:700'>{verdict}</div>"
                  f"<div>P(fatigue) = {p:.2f} · alarm at ≥ {model.threshold:.2f}</div></div>", unsafe_allow_html=True)
    c[1].metric('Heart rate', f"{res['features']['HR']:.0f} bpm", f"{res['features']['HR'] - base['features']['HR']:+.0f} vs baseline", delta_color='off')
    c[2].metric('RMSSD', f"{res['features']['RMSSD']:.0f} ms", f"{res['features']['RMSSD'] - base['features']['RMSSD']:+.0f} vs baseline", delta_color='off')
    c[3].metric('Beat-shape match', f'{sim:.3f}', help='1.000 = identical average heartbeat shape. Below ~0.98 the electrodes were probably placed differently.')
    if sim < 0.98:
        st.warning(f'The heartbeat shape differs from the baseline (match {sim:.3f}). Electrodes were probably placed '
                   'differently, so R-peak height and T-wave changes, which the model relies on most, may reflect placement, not fatigue.')
    if fatigued:
        st.error('Serious fatigue signs compared with your baseline. Take a **breathing break** (e.g. 2 min of slow '
                 'breathing, 6 breaths/min) and re-check later.')
    elif lvl == 'watch':
        st.info('Some change from your baseline, but not enough to call it fatigue. No action needed; re-check later if you feel tired.')

    st.subheader('Why: change from your baseline')
    t = contrib.copy()
    t['Measure'] = [f"{LABELS[k][0]} ({LABELS[k][1]})" if LABELS[k][1] else LABELS[k][0] for k in t.feature]
    t['Direction'] = np.where(t.push_to_fatigue > 0, '→ fatigue', '→ normal')
    st.dataframe(t[['Measure', 'baseline', 'new', 'change', 'pct_change', 'push_to_fatigue', 'Direction']].rename(columns={
        'baseline': 'Baseline', 'new': 'Now', 'change': 'Change', 'pct_change': 'Change %', 'push_to_fatigue': 'Effect on decision'}).round(2),
        hide_index=True, width='stretch')

    g1, g2 = st.columns(2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.array(base['hr_t']) - base['hr_t'][0], y=base['hr'], name='baseline', line=dict(color='#2e8b57')))
    fig.add_trace(go.Scatter(x=res['t_rr'] - res['t_rr'][0], y=60 / res['rr'], name='now', line=dict(color=col)))
    fig.update_layout(height=300, title='Beat-to-beat heart rate', xaxis_title='seconds into stable part', yaxis_title='bpm',
                      margin=dict(l=10, r=10, t=35, b=10))
    g1.plotly_chart(fig, width='stretch')
    mb, mn = np.array(base['median_beat']), res['median_beat']; tb = (np.arange(len(mb)) - int(PRE*FS_ECG)) / FS_ECG * 1000
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tb, y=mb, name='baseline', line=dict(color='#2e8b57', width=2)))
    fig.add_trace(go.Scatter(x=tb[:len(mn)], y=mn, name='now', line=dict(color=col, width=2)))
    fig.update_layout(height=300, title='Average heartbeat shape', xaxis_title='ms from R-peak', yaxis_title='mV',
                      margin=dict(l=10, r=10, t=35, b=10))
    g2.plotly_chart(fig, width='stretch')
    st.plotly_chart(ecg_figure(res, 'New recording — first 10 s of the stable part'), width='stretch')

    hist = USERS / uid / 'history.csv'
    key = f'logged_{uid}_{hash(data)}'
    if key not in st.session_state:
        pd.DataFrame([dict(time=datetime.now().isoformat(timespec='seconds'), recording=name, result=verdict,
                           p_fatigue=round(p, 3), hr=round(res['features']['HR'], 1), rmssd=round(res['features']['RMSSD'], 1),
                           beat_match=round(sim, 3))]).to_csv(hist, mode='a', header=not hist.exists(), index=False)
        st.session_state[key] = True
        if fatigued:                                    # serious fatigue -> one HR report per check
            rep = send_report(build_report(uid, name, p, model.threshold, res['features'], base['features'], contrib, sim))
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
    if not users():
        st.info('No users yet.'); st.stop()
    uid = st.selectbox('Person', users())
    hist = USERS / uid / 'history.csv'
    if not hist.exists():
        st.info('No checks yet for this person.'); st.stop()
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
        st.markdown(f"""
**What it learns:** the *change* from a person's own rested baseline that indicates fatigue.
Training examples are pairs of recordings from the **same person**:
- **fatigue examples:** a fatigue recording minus each of that person's normal recordings ({tr['fatigue_pairs']} pairs)
- **no-change examples:** a normal recording minus another normal of the same person, and the first half vs. the second half of long normal recordings ({tr['no_change_pairs']} pairs)

**Model:** standardised logistic regression on 8 changes (heart rate, HR range, SDNN, RMSSD, pNN50, R-peak height, T/R ratio, R-to-T time).
Trained {m['trained_at'].replace('T', ' ')} on **{tr['recordings']} recordings from {tr['people']} people** ({tr['normal_recordings']} normal, {tr['fatigue_recordings']} fatigue).
""")
        if ev:
            st.subheader('Honest test: leave-one-recording-out')
            st.markdown(f"Each recording is removed from training, then compared with each of that person's other normal baselines "
                        f"as if it were a new upload: **{ev['correct']}/{ev['comparisons']} correct ({ev['accuracy']:.0%})**. "
                        f"The alarm is tuned to fire only for **serious** fatigue (P ≥ {model.threshold:.2f}, set just above the "
                        f"highest held-out normal score): serious fatigue caught **{ev['fatigue_caught']}**, normal correctly left alone "
                        f"**{ev['normal_correct']}** (false alarms: {ev.get('false_alarms', '–')}). "
                        f"Fatigue at least at *watch* level: {ev.get('fatigue_watch_or_alarm', '–')}.")
            c1, c2 = st.columns([1, 2])
            c1.dataframe(pd.DataFrame(ev['confusion'], index=['true normal', 'true fatigue'],
                                      columns=['predicted normal', 'predicted fatigue']), width='stretch')
            coef = pd.Series(m['coef'], [LABELS[k][0] for k in m['features']]).sort_values()
            fig = go.Figure(go.Bar(x=coef.values, y=coef.index, orientation='h',
                                   marker_color=np.where(coef.values > 0, '#c9302c', '#2e8b57')))
            fig.update_layout(height=300, title='What the model uses (+ : an increase points to fatigue)', margin=dict(l=10, r=10, t=35, b=10))
            c2.plotly_chart(fig, width='stretch')
            with st.expander('All held-out comparisons'):
                st.dataframe(pd.DataFrame(ev['details']), hide_index=True, width='stretch')
        st.warning('**Limits right now:** only 2 people and about 30 s recordings with an empty optical channel. The strongest '
                   'signal is a drop in R-peak height, which also depends on electrode placement. Treat results as a prototype '
                   'until more recordings (3–5 min, same electrode spots, working PPG) are collected.')
    st.subheader('Training data')
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
