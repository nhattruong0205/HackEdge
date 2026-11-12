"""Train the baseline-comparison fatigue model.

Every recording is listed in data_manifest.csv with the PERSON it belongs to and its CONDITION
(normal / fatigue). Training examples are CHANGES within the same person:
  fatigue example  : features(fatigue recording) − features(normal recording of the same person)
  no-change example: features(normal) − features(another normal of the same person),
                     and first half vs. second half of each normal recording (≥ 40 s stable)
A standardised logistic regression learns which changes mean fatigue.

Honest evaluation: leave-one-recording-out. Each recording is removed from training, then
compared against every normal baseline of the same person as if it were a new upload.

Alert only when fatigue is SERIOUS: the decision threshold is calibrated on the held-out scores so
that normal recordings (compared with the same person's baseline) are not flagged. Scores between
WATCH_THRESHOLD and that threshold are reported as "watch" (mild change, no alarm).

Identical files listed twice in the manifest (e.g. a sample re-contributed via the app under a new
name) are counted once, so the same recording never appears as two people or two sessions.

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

from fatigue_pipeline import FEATURES, MODEL_PATH, ROOT, analyze, load_recording

MANIFEST = ROOT / 'data_manifest.csv'
WATCH_THRESHOLD = 0.5      # above this: "watch" (some change from baseline, no alarm)
MIN_SERIOUS = 0.6          # serious-fatigue threshold is never below this …
MAX_SERIOUS = 0.9          # … nor above this
MARGIN = 0.05              # safety margin above the highest held-out normal score


def load_dataset(manifest=MANIFEST, verbose=True):
    man = pd.read_csv(manifest)
    recs, halves, seen = {}, {}, {}
    for _, row in man.iterrows():
        path = ROOT / row.file
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
        recs[row.file] = dict(person=str(row.person), condition=row.condition, feats=res['features'])
        a, b = res['keep']
        if row.condition == 'normal' and b - a >= 40:                      # split-half "no change" pairs
            mid = (a + b) / 2
            h1, h2 = analyze(ecg, opt, (a, mid)), analyze(ecg, opt, (mid, b))
            if h1['usable'] and h2['usable']:
                halves[row.file] = (h1['features'], h2['features'])
    return recs, halves


def build_pairs(recs, halves, exclude=()):
    X, y, info = [], [], []
    keep = {k: v for k, v in recs.items() if k not in exclude}
    for person in {v['person'] for v in keep.values()}:
        mine = {k: v for k, v in keep.items() if v['person'] == person}
        normals = [k for k, v in mine.items() if v['condition'] == 'normal']
        fatigued = [k for k, v in mine.items() if v['condition'] == 'fatigue']
        for f in fatigued:
            for n in normals:
                X.append([mine[f]['feats'][c] - mine[n]['feats'][c] for c in FEATURES]); y.append(1); info.append((f, n))
        for a, b in permutations(normals, 2):
            X.append([mine[a]['feats'][c] - mine[b]['feats'][c] for c in FEATURES]); y.append(0); info.append((a, b))
    for k, (h1, h2) in halves.items():
        if k in exclude:
            continue
        for p, q in ((h2, h1), (h1, h2)):
            X.append([p[c] - q[c] for c in FEATURES]); y.append(0); info.append((k + ' (2nd half)', k + ' (1st half)'))
    return np.array(X), np.array(y), info


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000))


def evaluate(recs, halves):
    """Leave-one-recording-out: held-out recording vs. each remaining normal baseline of the same person."""
    rows = []
    for held in recs:
        X, y, _ = build_pairs(recs, halves, exclude={held})
        if len(set(y)) < 2:
            continue
        m = make_model().fit(X, y)
        person = recs[held]['person']
        baselines = [k for k, v in recs.items() if k != held and v['person'] == person and v['condition'] == 'normal']
        for base in baselines:
            d = np.array([[recs[held]['feats'][c] - recs[base]['feats'][c] for c in FEATURES]])
            p = float(m.predict_proba(d)[0, 1])
            rows.append(dict(new_recording=held, baseline=base, person=person, true=recs[held]['condition'],
                             p_fatigue=round(p, 3), predicted='fatigue' if p >= .5 else 'normal'))
    ev = pd.DataFrame(rows)
    return ev


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
    X, y, info = build_pairs(recs, halves)
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
        # one verdict per held-out recording: average score over all of that person's baselines
        per_rec = ev.groupby(['new_recording', 'true'], as_index=False).p_fatigue.mean()
        per_rec['level'] = [level(p, threshold) for p in per_rec.p_fatigue]
        summary = dict(comparisons=int(len(ev)), correct=int(ev.correct.sum()), accuracy=float(ev.correct.mean()),
                       fatigue_caught=f"{cm.loc['fatigue', 'fatigue']}/{cm.loc['fatigue'].sum()}",
                       normal_correct=f"{cm.loc['normal', 'normal']}/{cm.loc['normal'].sum()}",
                       false_alarms=int(cm.loc['normal', 'fatigue']),
                       fatigue_watch_or_alarm=f"{int((ev[ev.true == 'fatigue'].level != 'normal').sum())}/{cm.loc['fatigue'].sum()}",
                       confusion=cm.values.tolist(), details=ev.to_dict('records'),
                       per_recording=per_rec.round(3).to_dict('records'))
    model = dict(type='baseline_delta_logistic_regression', features=FEATURES,
                 mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=lr.coef_[0].tolist(),
                 intercept=float(lr.intercept_[0]), threshold=threshold, watch_threshold=WATCH_THRESHOLD,
                 trained_at=datetime.now().isoformat(timespec='seconds'),
                 training=dict(recordings=len(recs), people=len({v['person'] for v in recs.values()}),
                               normal_recordings=sum(v['condition'] == 'normal' for v in recs.values()),
                               fatigue_recordings=sum(v['condition'] == 'fatigue' for v in recs.values()),
                               fatigue_pairs=int(y.sum()), no_change_pairs=int((y == 0).sum())),
                 evaluation=summary)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(model, indent=2))
    if verbose:
        print(f"trained on {model['training']}")
        print(f'serious-fatigue threshold {threshold} (watch from {WATCH_THRESHOLD})')
        if summary:
            print(f"leave-one-recording-out: {summary['correct']}/{summary['comparisons']} comparisons correct "
                  f"({summary['accuracy']:.0%}); serious fatigue caught {summary['fatigue_caught']}, "
                  f"fatigue at least 'watch' {summary['fatigue_watch_or_alarm']}, "
                  f"normal left alone {summary['normal_correct']} (false alarms: {summary['false_alarms']})")
            print(ev.to_string(index=False))
            print('per recording (score averaged over all baselines of that person):')
            print(pd.DataFrame(summary['per_recording']).to_string(index=False))
        print('coefficients (+ = change pushes towards fatigue):')
        print(pd.Series(lr.coef_[0], FEATURES).sort_values(key=abs, ascending=False).round(3).to_string())
        print(f'saved {Path(out).relative_to(ROOT)}')
    return model


if __name__ == '__main__':
    train()
