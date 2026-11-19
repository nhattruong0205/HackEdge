"""Train the personal-baseline (delta) AND zero-baseline (single-recording) physiological fatigue models.

1. Personal-Baseline Delta Classifier (DeltaModel.predict):
   Every recording in data_manifest.csv is listed with its PERSON and CONDITION (normal / fatigue).
   Training examples are CHANGES within the same person:
     - fatigue example  : features(fatigue recording) − features(normal recording of the same person)
     - no-change example: features(normal) − features(another normal of the same person),
                          and first half vs. second half of each normal recording (≥ 40 s stable)
   Combined with physiological autonomic priors (so HR ↑, Stress Index ↑, LF/HF ↑, RMSSD ↓, CVSD ↓,
   SD1/SD2 ↓, HF_nu ↓ always carry biologically valid signs even on small sample sizes, and R_mV is
   regularized so electrode placement shifts don't dominate).

2. Zero-Baseline Single-Recording Classifier (DeltaModel.predict_single):
   For classifying a brand-new person from a single recording (no saved baseline required) using the 10
   placement-invariant physiological HRV & rate-corrected timing features (SINGLE_FEATURES).
   Normal recordings where the subject was not truly at rest (HR > 100 bpm and RMSSD < 12 ms) are
   flagged as 'unrested_baseline' so they do not corrupt the healthy rested reference distribution.

Run:  python train_delta_model.py          (re-run whenever data_manifest.csv gets new rows)
"""
import hashlib
import json
from datetime import datetime
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fatigue_pipeline import (
    FEATURES, HEALTHY_NORMS, MODEL_PATH, ROOT, SINGLE_FEATURES,
    analyze, load_recording, resolve_recording_path,
)

MANIFEST = ROOT / 'data_manifest.csv'
WATCH_THRESHOLD = 0.5      # above this: "watch" (some change from baseline, no alarm)
MIN_SERIOUS = 0.6          # serious-fatigue threshold is never below this …
MAX_SERIOUS = 0.9          # … nor above this
MARGIN = 0.05              # safety margin above the highest held-out normal score

# Literature-grounded autonomic delta prototypes (delta = fatigued - rested) to regularize small-sample training
# Ensures physiological HRV features carry biologically valid directions alongside empirical within-person pairs.
PHYSIO_DELTA_PRIORS = [
    # Clear vagal withdrawal + sympathetic activation (fatigue)
    (dict(HR=+14.0, HR_range=-3.0, SDNN=-14.0, RMSSD=-18.0, pNN50=-11.0, CVSD=-1.9, SD1_SD2=-0.18,
          log_SI=+0.48, LF_HF=+3.5, HF_nu=-18.0, R_mV=-1.5, T_over_R=-0.03, RT_ms=-12.0, RTc_ms=-6.0), 1),
    (dict(HR=+9.0, HR_range=-2.0, SDNN=-10.0, RMSSD=-14.0, pNN50=-8.0, CVSD=-1.4, SD1_SD2=-0.14,
          log_SI=+0.35, LF_HF=+2.4, HF_nu=-14.0, R_mV=-1.0, T_over_R=-0.02, RT_ms=-8.0, RTc_ms=-4.0), 1),
    (dict(HR=+18.0, HR_range=-4.0, SDNN=-18.0, RMSSD=-22.0, pNN50=-14.0, CVSD=-2.3, SD1_SD2=-0.22,
          log_SI=+0.60, LF_HF=+4.8, HF_nu=-22.0, R_mV=-2.0, T_over_R=-0.04, RT_ms=-16.0, RTc_ms=-8.0), 1),
    # Normal day-to-day rested variation (no fatigue)
    (dict(HR=+1.5, HR_range=+1.0, SDNN=+2.0, RMSSD=+1.5, pNN50=+1.0, CVSD=+0.15, SD1_SD2=+0.02,
          log_SI=-0.05, LF_HF=-0.3, HF_nu=+2.0, R_mV=+0.5, T_over_R=+0.005, RT_ms=+2.0, RTc_ms=+1.0), 0),
    (dict(HR=-2.0, HR_range=+0.5, SDNN=+3.5, RMSSD=+4.0, pNN50=+2.5, CVSD=+0.40, SD1_SD2=+0.04,
          log_SI=-0.12, LF_HF=-0.6, HF_nu=+4.5, R_mV=+0.2, T_over_R=+0.002, RT_ms=+4.0, RTc_ms=+2.0), 0),
    (dict(HR=+0.0, HR_range=-0.5, SDNN=-1.0, RMSSD=-0.5, pNN50=-0.5, CVSD=-0.05, SD1_SD2=-0.01,
          log_SI=+0.02, LF_HF=+0.1, HF_nu=-1.0, R_mV=-3.0, T_over_R=+0.00, RT_ms=+0.0, RTc_ms=+0.0), 0),
]


def is_truly_rested_normal(feats):
    """A normal recording is considered truly rested if HR <= 95 bpm or RMSSD >= 15 ms."""
    return feats['HR'] <= 95.0 or feats['RMSSD'] >= 15.0


def load_dataset(manifest=MANIFEST, verbose=True):
    man = pd.read_csv(manifest)
    recs, halves, seen = {}, {}, {}
    for _, row in man.iterrows():
        path = resolve_recording_path(row.file)
        if not path.exists():
            print(f'  skip (missing): {row.file}'); continue
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest in seen:
            if verbose:
                print(f'  skip (duplicate of {seen[digest]}): {row.file}')
            continue
        seen[digest] = row.file
        ecg, opt = load_recording(path)
        res = analyze(ecg, opt)
        if not res['usable']:
            print(f'  skip (not enough clean signal): {row.file}'); continue
        rested_ok = is_truly_rested_normal(res['features']) if row.condition == 'normal' else True
        recs[row.file] = dict(
            person=str(row.person), condition=row.condition,
            feats=res['features'], rested_ok=rested_ok,
        )
        a, b = res['keep']
        if row.condition == 'normal' and b - a >= 40:                      # split-half "no change" pairs
            mid = (a + b) / 2
            h1, h2 = analyze(ecg, opt, (a, mid)), analyze(ecg, opt, (mid, b))
            if h1['usable'] and h2['usable']:
                halves[row.file] = (h1['features'], h2['features'])
    return recs, halves


def build_pairs(recs, halves, exclude=(), include_priors=True):
    X, y, info = [], [], []
    keep = {k: v for k, v in recs.items() if k not in exclude}
    for person in {v['person'] for v in keep.values()}:
        mine = {k: v for k, v in keep.items() if v['person'] == person}
        # Only use physiologically valid rested normal recordings as baselines
        normals = [k for k, v in mine.items() if v['condition'] == 'normal' and v['rested_ok']]
        fatigued = [k for k, v in mine.items() if v['condition'] == 'fatigue']
        for f in fatigued:
            for n in normals:
                X.append([mine[f]['feats'][c] - mine[n]['feats'][c] for c in FEATURES])
                y.append(1)
                info.append((f, n))
        for a, b in permutations(normals, 2):
            X.append([mine[a]['feats'][c] - mine[b]['feats'][c] for c in FEATURES])
            y.append(0)
            info.append((a, b))
    for k, (h1, h2) in halves.items():
        if k in exclude:
            continue
        for p, q in ((h2, h1), (h1, h2)):
            X.append([p[c] - q[c] for c in FEATURES])
            y.append(0)
            info.append((k + ' (2nd half)', k + ' (1st half)'))
    if include_priors:
        for proto, lbl in PHYSIO_DELTA_PRIORS:
            X.append([proto[c] for c in FEATURES])
            y.append(lbl)
            info.append(('physio_prior', 'physio_prior'))
    return np.array(X, float), np.array(y, int), info


def make_model(c_reg=0.4):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c_reg, class_weight='balanced', max_iter=2000))


def evaluate(recs, halves):
    """Leave-one-recording-out: held-out recording vs. each remaining valid rested normal baseline of the same person."""
    rows = []
    for held in recs:
        if recs[held]['condition'] == 'normal' and not recs[held]['rested_ok']:
            continue
        X, y, _ = build_pairs(recs, halves, exclude={held}, include_priors=True)
        if len(set(y)) < 2:
            continue
        m = make_model().fit(X, y)
        person = recs[held]['person']
        baselines = [
            k for k, v in recs.items()
            if k != held and v['person'] == person and v['condition'] == 'normal' and v['rested_ok']
        ]
        for base in baselines:
            d = np.array([[recs[held]['feats'][c] - recs[base]['feats'][c] for c in FEATURES]], float)
            p = float(m.predict_proba(d)[0, 1])
            rows.append(dict(new_recording=held, baseline=base, person=person, true=recs[held]['condition'],
                             p_fatigue=round(p, 3), predicted='fatigue' if p >= .5 else 'normal'))
    return pd.DataFrame(rows)


def train_single_recording_model(recs):
    """Train & LORO-evaluate the Zero-Baseline Single-Recording model on SINGLE_FEATURES."""
    # Clean training pool: rested normal recordings + all fatigue recordings
    clean_keys = [k for k, v in recs.items() if v['condition'] == 'fatigue' or v['rested_ok']]
    unrested_normals = [k for k, v in recs.items() if v['condition'] == 'normal' and not v['rested_ok']]

    X_clean = np.array([[recs[k]['feats'][c] for c in SINGLE_FEATURES] for k in clean_keys], float)
    y_clean = np.array([1 if recs[k]['condition'] == 'fatigue' else 0 for k in clean_keys], int)

    m = make_model(c_reg=0.5).fit(X_clean, y_clean)
    sc, lr = m.named_steps['standardscaler'], m.named_steps['logisticregression']

    # Leave-one-recording-out evaluation across clean recordings (and also score unrested normals for transparency)
    loro_rows = []
    for held in recs:
        train_keys = [k for k in clean_keys if k != held]
        X_tr = np.array([[recs[k]['feats'][c] for c in SINGLE_FEATURES] for k in train_keys], float)
        y_tr = np.array([1 if recs[k]['condition'] == 'fatigue' else 0 for k in train_keys], int)
        if len(set(y_tr)) < 2:
            continue
        m_fold = make_model(c_reg=0.5).fit(X_tr, y_tr)
        x_te = np.array([[recs[held]['feats'][c] for c in SINGLE_FEATURES]], float)
        p = float(m_fold.predict_proba(x_te)[0, 1])
        loro_rows.append(dict(
            recording=held, person=recs[held]['person'], true=recs[held]['condition'],
            rested_baseline_ok=recs[held]['rested_ok'], p_fatigue=round(p, 3),
        ))
    ev_s = pd.DataFrame(loro_rows)
    clean_ev = ev_s[ev_s['rested_baseline_ok']] if len(ev_s) else ev_s
    normal_scores = clean_ev.loc[clean_ev.true == 'normal', 'p_fatigue'] if len(clean_ev) else pd.Series(dtype=float)
    top_norm = float(normal_scores.max()) if len(normal_scores) else 0.5
    s_thr = round(float(np.clip(top_norm + MARGIN, 0.55, 0.85)), 3)

    if len(ev_s):
        ev_s['predicted'] = np.where(ev_s.p_fatigue >= s_thr, 'fatigue', 'normal')
        ev_s['correct'] = ev_s.true == ev_s.predicted
    clean_ev = ev_s[ev_s['rested_baseline_ok']] if len(ev_s) else ev_s

    rested_norms = {
        c: round(float(np.mean([recs[k]['feats'][c] for k in clean_keys if recs[k]['condition'] == 'normal'])), 3)
        for c in SINGLE_FEATURES
    } if any(recs[k]['condition'] == 'normal' for k in clean_keys) else HEALTHY_NORMS

    return dict(
        type='zero_baseline_physiological_logistic_regression',
        features=SINGLE_FEATURES,
        mean=sc.mean_.tolist(),
        scale=sc.scale_.tolist(),
        coef=lr.coef_[0].tolist(),
        intercept=float(lr.intercept_[0]),
        threshold=s_thr,
        watch_threshold=0.45,
        rested_reference=rested_norms,
        unrested_baselines_excluded=unrested_normals,
        evaluation=dict(
            clean_recordings=int(len(clean_ev)),
            clean_correct=int(clean_ev.correct.sum()) if len(clean_ev) else 0,
            clean_accuracy=float(clean_ev.correct.mean()) if len(clean_ev) else 0.0,
            all_recordings=int(len(ev_s)),
            all_correct=int(ev_s.correct.sum()) if len(ev_s) else 0,
            details=ev_s.to_dict('records'),
        ),
    )


def calibrate_threshold(ev):
    """Serious-fatigue threshold: just above the highest held-out score of a NORMAL recording,
    so a normal session vs. the person's own baseline does not raise an alarm."""
    normal_scores = ev.loc[ev.true == 'normal', 'p_fatigue'] if len(ev) else pd.Series(dtype=float)
    top = float(normal_scores.max()) if len(normal_scores) else WATCH_THRESHOLD
    return round(float(np.clip(top + MARGIN, MIN_SERIOUS, MAX_SERIOUS)), 3)


def level(p, threshold):
    return 'fatigue' if p >= threshold else 'watch' if p >= WATCH_THRESHOLD else 'normal'


def train(manifest=MANIFEST, out=MODEL_PATH, verbose=True):
    recs, halves = load_dataset(manifest, verbose)
    X_emp, y_emp, _ = build_pairs(recs, halves, include_priors=False)
    X, y, info = build_pairs(recs, halves, include_priors=True)
    if len(set(y)) < 2:
        raise SystemExit('Need at least one person with both a normal and a fatigue recording.')
    ev = evaluate(recs, halves)
    threshold = calibrate_threshold(ev)
    m = make_model().fit(X, y)
    sc, lr = m.named_steps['standardscaler'], m.named_steps['logisticregression']
    summary = {}
    if len(ev):
        ev['level'] = [level(p, threshold) for p in ev.p_fatigue]
        ev['predicted'] = np.where(ev.level == 'fatigue', 'fatigue', 'normal')   # only serious fatigue alarms
        ev['correct'] = ev.true == ev.predicted
        cm = pd.crosstab(ev.true, ev.predicted).reindex(index=['normal', 'fatigue'], columns=['normal', 'fatigue'], fill_value=0)
        per_rec = ev.groupby(['new_recording', 'true'], as_index=False).p_fatigue.mean()
        per_rec['level'] = [level(p, threshold) for p in per_rec.p_fatigue]
        summary = dict(comparisons=int(len(ev)), correct=int(ev.correct.sum()), accuracy=float(ev.correct.mean()),
                       fatigue_caught=f"{cm.loc['fatigue', 'fatigue']}/{cm.loc['fatigue'].sum()}",
                       normal_correct=f"{cm.loc['normal', 'normal']}/{cm.loc['normal'].sum()}",
                       false_alarms=int(cm.loc['normal', 'fatigue']),
                       fatigue_watch_or_alarm=f"{int((ev[ev.true == 'fatigue'].level != 'normal').sum())}/{cm.loc['fatigue'].sum()}",
                       confusion=cm.values.tolist(), details=ev.to_dict('records'),
                       per_recording=per_rec.round(3).to_dict('records'))

    single_model = train_single_recording_model(recs)

    model = dict(
        type='baseline_delta_and_zero_baseline_physiological_logistic_regression',
        features=FEATURES,
        mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=lr.coef_[0].tolist(),
        intercept=float(lr.intercept_[0]), threshold=threshold, watch_threshold=WATCH_THRESHOLD,
        trained_at=datetime.now().isoformat(timespec='seconds'),
        training=dict(
            recordings=len(recs), people=len({v['person'] for v in recs.values()}),
            normal_recordings=sum(v['condition'] == 'normal' for v in recs.values()),
            fatigue_recordings=sum(v['condition'] == 'fatigue' for v in recs.values()),
            fatigue_pairs=int(y_emp.sum()), no_change_pairs=int((y_emp == 0).sum()),
            physio_priors=len(PHYSIO_DELTA_PRIORS),
        ),
        evaluation=summary,
        single_model=single_model,
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(model, indent=2))
    if verbose:
        print(f"trained on {model['training']}")
        print(f'serious-fatigue threshold {threshold} (watch from {WATCH_THRESHOLD})')
        if summary:
            print(f"personal-baseline LORO: {summary['correct']}/{summary['comparisons']} comparisons correct "
                  f"({summary['accuracy']:.0%}); serious fatigue caught {summary['fatigue_caught']}, "
                  f"fatigue at least 'watch' {summary['fatigue_watch_or_alarm']}, "
                  f"normal left alone {summary['normal_correct']} (false alarms: {summary['false_alarms']})")
        sev = single_model['evaluation']
        print(f"zero-baseline (new person single-recording) LORO on rested baselines + fatigue: "
              f"{sev['clean_correct']}/{sev['clean_recordings']} ({sev['clean_accuracy']:.0%}) "
              f"[unrested baselines flagged: {single_model['unrested_baselines_excluded']}]")
        print('\npersonal-baseline delta coefficients (+ = increase pushes towards fatigue):')
        print(pd.Series(lr.coef_[0], FEATURES).sort_values(key=abs, ascending=False).round(3).to_string())
        print('\nzero-baseline single-recording coefficients (+ = higher value points to fatigue):')
        print(pd.Series(single_model['coef'], SINGLE_FEATURES).sort_values(key=abs, ascending=False).round(3).to_string())
        print(f'saved {Path(out).relative_to(ROOT)}')
    return model


if __name__ == '__main__':
    train()
