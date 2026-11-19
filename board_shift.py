"""Personal-baseline fatigue model that trains and runs ON THE BOARD (needs only numpy).

Idea: fatigue = a SHIFT away from the person's own rested state. For every recording we measure
heart rate, SDNN and RMSSD, and compare it with the same person's normal recordings as a percent
change (log ratio). A logistic regression learns which shifts mean fatigue (HR up, HRV down).

Commands (run in the folder that holds board_shift.py, board_classifier.py, board_manifest.csv, data/):
    python3 board_shift.py train                         # train + honest test, save models/shift_model.json
    python3 board_shift.py check A new_recording.csv     # NORMAL / WATCH / FATIGUE vs person A's baseline
    python3 board_shift.py enroll NAME rested1.csv ...   # give a new person a rested baseline
    python3 board_shift.py people                        # list people with a baseline

board_manifest.csv columns: file, person, condition (normal / fatigue / anything else = ignored),
source (a recording id; recordings that share a tag are held out together in the honest test).
"""
import csv
import json
import sys
import time
from itertools import permutations
from pathlib import Path

import numpy as np

from board_classifier import features, fit_logreg, load_ecg, preprocess, prob

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'board_manifest.csv'
MODEL = ROOT / 'models' / 'shift_model.json'
SHIFT = ['HR', 'SDNN', 'RMSSD']
WATCH, MIN_ALARM, MAX_ALARM, MARGIN = 0.5, 0.6, 0.9, 0.05


def rec_features(path):
    """Whole-recording features (stable part) + first/second half for long recordings."""
    p = preprocess(load_ecg(path))
    a, b = p['stable']
    whole = features(p, a, b, min_beats=15)
    halves = None
    if whole and b - a >= 40:
        h1, h2 = features(p, a, (a + b) / 2, 15), features(p, (a + b) / 2, b, 15)
        halves = (h1, h2) if h1 and h2 else None
    return whole, halves


def shift(new, base):
    return [float(np.log(new[k] / base[k])) for k in SHIFT]


def load_manifest():
    rows, t0 = [], time.time()
    with open(MANIFEST, newline='') as fh:
        for r in csv.DictReader(fh):
            path = ROOT / r['file']
            if r['condition'] not in ('normal', 'fatigue'):
                continue
            if not path.exists():
                print(f'  skip (missing): {r["file"]}'); continue
            t = time.time(); whole, halves = rec_features(path)
            if whole is None:
                print(f'  skip (not enough clean signal): {r["file"]}'); continue
            rows.append(dict(file=r['file'], person=r['person'], condition=r['condition'],
                             source=set(r['source'].split('+')), f=whole, halves=halves))
            print(f'  {r["file"]:34s} person {r["person"]}  {r["condition"]:7s}  HR {whole["HR"]:5.1f}  '
                  f'SDNN {whole["SDNN"]:5.1f}  RMSSD {whole["RMSSD"]:5.1f}  ({time.time() - t:.1f} s)')
    print(f'Preprocessed {len(rows)} recordings in {time.time() - t0:.0f} s on this device.')
    return rows


def pairs(rows):
    """Training examples: fatigue vs a normal of the same person (1); normal vs normal, and
    second half vs first half of a long normal (0)."""
    X, y = [], []
    for person in {r['person'] for r in rows}:
        mine = [r for r in rows if r['person'] == person]
        normals = [r for r in mine if r['condition'] == 'normal']
        for f in (r for r in mine if r['condition'] == 'fatigue'):
            for n in normals:
                X.append(shift(f['f'], n['f'])); y.append(1)
        for a, b in permutations(normals, 2):
            if not (a['source'] & b['source']):          # never pair a recording with itself
                X.append(shift(a['f'], b['f'])); y.append(0)
        for n in normals:
            if n['halves']:
                h1, h2 = n['halves']
                X += [shift(h2, h1), shift(h1, h2)]; y += [0, 0]
    return np.array(X), np.array(y, float)


def level(p, alarm):
    return 'FATIGUE' if p >= alarm else 'WATCH' if p >= WATCH else 'NORMAL'


def train():
    t0 = time.time()
    print('Preprocessing recordings (filters, R-peaks, HR / SDNN / RMSSD)…')
    rows = load_manifest()
    X, y = pairs(rows)
    print(f'\nTraining pairs: {int(y.sum())} fatigue, {int((1 - y).sum())} no-change')

    print('\nHonest test: hold out one recording, retrain, '
          'compare it with the remaining normals of that person:')
    results = []
    for held in rows:
        rest = [r for r in rows if not (r['source'] & held['source'])]
        Xt, yt = pairs(rest)
        if len(set(yt)) < 2:
            continue
        m = fit_logreg(Xt, yt)
        bases = [r for r in rest if r['person'] == held['person'] and r['condition'] == 'normal']
        if not bases:
            continue
        p = float(np.mean([prob(m, np.array([shift(held['f'], b['f'])]))[0] for b in bases]))
        results.append((held, p))
    normal_ps = [p for r, p in results if r['condition'] == 'normal']
    alarm = round(float(np.clip(max(normal_ps, default=WATCH) + MARGIN, MIN_ALARM, MAX_ALARM)), 3)
    ok = 0
    for r, p in results:
        lv = level(p, alarm); good = (lv == 'FATIGUE') == (r['condition'] == 'fatigue'); ok += good
        print(f'  {r["file"]:34s} true {r["condition"]:7s} P(fatigue) {p:.2f}  -> {lv:7s} {"OK" if good else "WRONG"}')
    print(f'  {ok}/{len(results)} correct (alarm at P >= {alarm}, watch from {WATCH})')

    m = fit_logreg(X, y)
    people = {}
    for person in sorted({r['person'] for r in rows}):
        normals = [r for r in rows if r['person'] == person and r['condition'] == 'normal']
        people[person] = dict(baselines=[{k: r['f'][k] for k in SHIFT} for r in normals],
                              files=[r['file'] for r in normals])
    model = dict(type='personal_shift_logreg', features=SHIFT, **m, alarm=alarm, watch=WATCH, people=people,
                 trained_on=len(rows), honest_test=f'{ok}/{len(results)}', trained_at=time.strftime('%Y-%m-%d %H:%M'))
    MODEL.parent.mkdir(exist_ok=True)
    MODEL.write_text(json.dumps(model, indent=1))
    print('\nWeights (+ = pushes towards fatigue, per standardised % change):',
          {k: round(c, 2) for k, c in zip(SHIFT, m['coef'])})
    print(f'Saved {MODEL.relative_to(ROOT)}  with baselines for people {list(people)}  '
          f'(total {time.time() - t0:.0f} s on this device)')


def load_model():
    if not MODEL.exists():
        sys.exit('No model yet: run  python3 board_shift.py train')
    return json.loads(MODEL.read_text())


def check(person, paths):
    m = load_model()
    if person not in m['people']:
        sys.exit(f'No baseline for {person}. Known: {list(m["people"])}. Use: python3 board_shift.py enroll {person} rested.csv')
    bases = m['people'][person]['baselines']
    ref = {k: float(np.mean([b[k] for b in bases])) for k in SHIFT}
    for path in paths:
        t = time.time(); f, _ = rec_features(path)
        if f is None:
            print(f'{path}: not enough clean signal (need >= 15 clean beats)'); continue
        p = float(np.mean([prob(m, np.array([shift(f, b)]))[0] for b in bases]))
        changes = '  '.join(f'{k} {ref[k]:.0f}->{f[k]:.0f} ({100 * (f[k] / ref[k] - 1):+.0f}%)' for k in SHIFT)
        print(f'{path}: {level(p, m["alarm"])}  P(fatigue) {p:.2f}  | {changes}  ({time.time() - t:.1f} s)')


def enroll(person, paths):
    m = load_model(); base, files = [], []
    for path in paths:
        f, _ = rec_features(path)
        if f is None:
            print(f'skip {path}: not enough clean signal'); continue
        base.append({k: f[k] for k in SHIFT}); files.append(path)
        print(f'{path}: HR {f["HR"]:.0f}  SDNN {f["SDNN"]:.0f}  RMSSD {f["RMSSD"]:.0f}')
        if f['HR'] > 90:
            print('   warning: HR > 90 bpm does not look rested; sit quietly 5 min and re-record if possible')
    if not base:
        sys.exit('No usable recordings.')
    m['people'][person] = dict(baselines=base, files=files)
    MODEL.write_text(json.dumps(m, indent=1))
    print(f'Saved baseline for {person} ({len(base)} recording(s)).')


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[:1] == ['train']:
        train()
    elif a[:1] == ['check'] and len(a) >= 3:
        check(a[1], a[2:])
    elif a[:1] == ['enroll'] and len(a) >= 3:
        enroll(a[1], a[2:])
    elif a[:1] == ['people']:
        for k, v in load_model()['people'].items():
            print(k, v['files'])
    else:
        print(__doc__)
