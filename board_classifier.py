"""Normal-vs-fatigue classifier that TRAINS and PREDICTS on the SK-AM62B with numpy only
(no scipy / pandas / scikit-learn needed).

  python3 board_classifier.py train              # train on data/normal + data/fatigue, honest evaluation, save model
  python3 board_classifier.py train --hrv-only   # use only heart-rate / HRV features (less sensitive to electrode placement)
  python3 board_classifier.py predict FILE.csv   # NORMAL / FATIGUE for a new recording
  python3 board_classifier.py split --train normal/first_attempt.csv fatigue/first_attempt.csv ... \
                                    --test normal/third_attempt.csv ...     # custom train / test split
  (The personal-baseline fatigue model is in board_shift.py, which reuses the preprocessing below.)

How it works
  1. Preprocess each ECG recording (same steps as fatigue_pipeline.py: filters, artifact mask,
     Pan-Tompkins R-peaks, RR cleaning, stable part).
  2. Cut the stable part into 20 s windows (10 s step) and compute 8 features per window.
  3. Logistic regression (standardised, L2) trained with Newton's method in numpy.
  4. Honest test: leave-one-recording-out. Each recording is held out, the model is trained on
     the others, and the held-out recording is classified by the average of its windows.
Labels come from the folder name: data/normal -> 0, data/fatigue -> 1.
"""
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'models' / 'window_classifier.json'
FS = 500
WIN, STEP, MIN_BEATS = 20.0, 10.0, 12
ALL_FEATURES = ['HR', 'HR_range', 'SDNN', 'RMSSD', 'pNN50', 'R_mV', 'T_over_R', 'RT_ms']
HRV_FEATURES = ['HR', 'HR_range', 'SDNN', 'RMSSD', 'pNN50']
PRE, POST = 0.2, 0.4

# Filter coefficients for fs = 500 Hz (scipy.signal.butter order 2 / iirnotch Q=30), precomputed.
COEF = {
    'hp':    ([0.9955669720176472, -1.9911339440352944, 0.9955669720176472], [1.0, -1.9911142922016536, 0.9911535958689355]),
    'notch': ([0.9875889380903247, -1.4398427053125467, 0.9875889380903247], [1.0, -1.4398427053125467, 0.9751778761806493]),
    'lp':    ([0.04613180209331293, 0.09226360418662587, 0.04613180209331293], [1.0, -1.3072850288493236, 0.4918122372225753]),
    'bp':    ([0.007820208033497191, 0.0, -0.015640416066994383, 0.0, 0.007820208033497191],
              [1.0, -3.705188856470422, 5.1803044553998845, -3.2409032364051638, 0.7660066009432637]),
    'lp1':   ([3.913020539914434e-05, 7.826041079828868e-05, 3.913020539914434e-05], [1.0, -1.9822289297925284, 0.9823854506141251]),
}


# ---------------- loading ----------------
def load_ecg(path):
    """TI AFE4950 GUI export -> ECG in volts. Columns have different lengths; blanks are skipped."""
    with open(path, newline='') as fh:
        rows = csv.reader(fh)
        idx = {name: i for i, name in enumerate(next(rows))}
        val, raw = [], []
        for r in rows:
            for name, out in (('ECG_Value', val), ('ECG_RAW_Value', raw)):
                i = idx.get(name)
                if i is not None and i < len(r) and r[i].strip():
                    out.append(float(r[i]))
    if val:
        return np.array(val)
    raw = np.array(raw)
    return np.where(raw >= 2**23, raw - 2**24, raw) / (2**23 / 4.8)


# ---------------- scipy replacements (numerically identical) ----------------
def lfilter(b, a, x, zi):
    n, z, y = len(a), list(zi), np.empty(len(x))
    for k, xk in enumerate(x.tolist()):
        yk = b[0] * xk + z[0]
        for i in range(n - 2):
            z[i] = b[i + 1] * xk + z[i + 1] - a[i + 1] * yk
        z[n - 2] = b[n - 1] * xk - a[n - 1] * yk
        y[k] = yk
    return y


def lfilter_zi(b, a):
    b, a = np.asarray(b, float), np.asarray(a, float)
    n = len(a)
    comp = np.zeros((n - 1, n - 1)); comp[0] = -a[1:] / a[0]; comp[1:, :-1] += np.eye(n - 2)
    return np.linalg.solve(np.eye(n - 1) - comp.T, b[1:] - a[1:] * b[0])


def filtfilt(name, x):
    b, a = COEF[name]
    pad = 3 * max(len(a), len(b))
    ext = np.concatenate((2 * x[0] - x[pad:0:-1], x, 2 * x[-1] - x[-2:-pad - 2:-1]))
    zi = lfilter_zi(b, a)
    y = lfilter(b, a, ext, zi * ext[0])
    return lfilter(b, a, y[::-1], zi * y[-1])[::-1][pad:-pad]


def find_peaks(x, distance, height):
    xl, n, peaks, i = x.tolist(), len(x), [], 1
    while i < n - 1:
        if xl[i - 1] < xl[i]:
            j = i + 1
            while j < n - 1 and xl[j] == xl[i]:
                j += 1
            if xl[j] < xl[i]:
                peaks.append((i + j - 1) // 2)
                i = j
        i += 1
    pk = np.array(peaks, int)
    pk = pk[x[pk] >= height]
    keep = np.ones(len(pk), bool)
    for j in np.argsort(x[pk], kind='stable')[::-1]:
        if keep[j]:
            k = j - 1
            while k >= 0 and pk[j] - pk[k] < distance:
                keep[k] = False; k -= 1
            k = j + 1
            while k < len(pk) and pk[k] - pk[j] < distance:
                keep[k] = False; k += 1
    return pk[keep]


def per_second(v, how):
    starts = np.arange(0, len(v), FS)
    mx = np.maximum.reduceat(v, starts)
    return mx if how == 'max' else mx - np.minimum.reduceat(v, starts)


# ---------------- preprocessing (mirrors fatigue_pipeline.py) ----------------
def preprocess(ecg_v):
    x = 1000 * (ecg_v - np.median(ecg_v))
    for f in ('hp', 'notch', 'lp'):
        x = filtfilt(f, x)
    ecg_c = x
    sec = np.arange(len(ecg_c)) // FS
    peak = per_second(np.abs(ecg_c), 'max')
    bad = np.isin(sec, np.nonzero(peak > 2.0 * np.median(peak))[0]) | (np.abs(ecg_v) > 1.19)
    mask = np.convolve(bad, np.ones(FS), 'same') > 0

    # Pan-Tompkins R-peaks
    xz = np.where(mask, 0, ecg_c)
    w = int(.12 * FS)
    mwi = np.convolve(np.gradient(filtfilt('bp', xz)) ** 2, np.ones(w) / w, 'same')
    ref = mwi[~mask] if (~mask).any() else mwi
    pk = find_peaks(mwi, int(.33 * FS), .3 * np.percentile(ref, 99))
    hw = int(.06 * FS)
    pk = np.array([p for p in pk if hw <= p < len(xz) - hw], int)
    if len(pk):
        absmax = np.array([p - hw + np.argmax(np.abs(ecg_c[p - hw:p + hw])) for p in pk])
        pol = np.sign(np.median(ecg_c[absmax])) or 1.0
        pk = np.array([p - hw + np.argmax(pol * ecg_c[p - hw:p + hw]) for p in pk], int)
        pk = pk[~mask[pk]]
        amp = pol * ecg_c[pk]; med = np.median(amp) if len(pk) else 0
        pk = pk[(amp >= .5 * med) & (amp <= 1.6 * med)]
    t = np.arange(len(ecg_v)) / FS

    # RR cleaning
    if len(pk) > 2:
        tp = t[pk]; rr = np.diff(tp); t_rr = tp[1:]
        ok = (rr > .33) & (rr < 1.5)
        ok &= ~np.array([mask[int(a * FS):int(b * FS) + 1].any() for a, b in zip(tp[:-1], tp[1:])], bool)
        v = np.where(ok, rr, np.nan)
        med = np.array([np.nanmedian(v[max(0, i - 2):i + 3]) if np.isfinite(v[max(0, i - 2):i + 3]).any() else np.nan
                        for i in range(len(v))])
        with np.errstate(invalid='ignore'):
            ok &= np.abs(rr - med) <= .2 * med
    else:
        t_rr, rr, ok = np.array([]), np.array([]), np.array([], bool)

    # stable part (trim unstable start/end)
    drift = per_second(filtfilt('lp1', 1000 * (ecg_v - np.median(ecg_v))), 'ptp')
    r_amp = np.median(np.abs(ecg_c[pk])) if len(pk) else np.inf
    badsec = (drift > .3 * r_amp) | (peak > 1.6 * r_amp) | (peak < .5 * r_amp)
    s, e = 2, len(badsec) - 2
    while s < e and badsec[s:s + 3].any(): s += 1
    while e > s and badsec[e - 3:e].any(): e -= 1
    return dict(t=t, ecg_c=ecg_c, mask=mask, rpeaks=pk, t_rr=t_rr, rr=rr, ok=ok, stable=(float(s), float(e)))


def features(p, a, b, min_beats=MIN_BEATS):
    """8 features from the part a..b seconds; None if too few clean beats."""
    t, ecg_c, mask, rp = p['t'], p['ecg_c'], p['mask'], p['rpeaks']
    sel = p['ok'] & (p['t_rr'] > a) & (p['t_rr'] < b)
    pk = rp[(t[rp] > a + PRE) & (t[rp] < b - POST)]
    pk = pk[~np.array([mask[q - int(PRE * FS):q + int(POST * FS)].any() for q in pk], bool)]
    if sel.sum() < min_beats or len(pk) < min(10, min_beats):
        return None
    beat = np.median(np.array([ecg_c[q - int(PRE * FS):q + int(POST * FS)] for q in pk]), 0)
    rr = p['rr'][sel]; ms = 1000 * rr; d = np.diff(ms)
    i0 = int(PRE * FS); base = np.median(beat[:int(.08 * FS)]); R = beat[i0] - base
    tseg = beat[i0 + int(.15 * FS): i0 + int(.38 * FS)] - base; ti = int(np.argmax(np.abs(tseg)))
    return dict(HR=60000 / ms.mean(), HR_range=60 / rr.min() - 60 / rr.max(), SDNN=ms.std(ddof=1),
                RMSSD=float(np.sqrt(np.mean(d ** 2))), pNN50=100 * float(np.mean(np.abs(d) > 50)),
                R_mV=float(R), T_over_R=float(tseg[ti] / R), RT_ms=(int(.15 * FS) + ti) / FS * 1000)


def window_features(path):
    p = preprocess(load_ecg(path))
    a, b = p['stable']
    wins = []
    start = a
    while start + WIN <= b + 1e-9:
        f = features(p, start, start + WIN)
        if f:
            wins.append(f)
        start += STEP
    if not wins:                                   # short recording: use the whole stable part once
        f = features(p, a, b)
        wins = [f] if f else []
    return wins, p


# ---------------- logistic regression in numpy ----------------
def fit_logreg(X, y, l2=1.0, iters=50):
    """Standardised, class-balanced, L2-regularised logistic regression via Newton's method."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w_cls = np.where(y == 1, len(y) / (2 * max(1, y.sum())), len(y) / (2 * max(1, (1 - y).sum())))
    beta = np.zeros(Z.shape[1])
    reg = np.full(Z.shape[1], l2); reg[-1] = 0.0                # no penalty on the intercept
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ beta))
        g = Z.T @ (w_cls * (p - y)) + reg * beta
        H = (Z * (w_cls * p * (1 - p))[:, None]).T @ Z + np.diag(reg)
        step = np.linalg.solve(H, g)
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    return dict(mean=mu.tolist(), scale=sd.tolist(), coef=beta[:-1].tolist(), intercept=float(beta[-1]))


def prob(model, X):
    z = (np.asarray(X) - np.array(model['mean'])) / np.array(model['scale'])
    return 1 / (1 + np.exp(-(z @ np.array(model['coef']) + model['intercept'])))


# ---------------- commands ----------------
def load_dataset(feats):
    recs = []
    for label, folder in ((0, 'normal'), (1, 'fatigue')):
        for path in sorted((ROOT / 'data' / folder).glob('*.csv')):
            t0 = time.time()
            wins, _ = window_features(path)
            print(f'  {folder}/{path.name:22s} {len(wins):2d} windows  ({time.time() - t0:.1f} s)')
            if wins:
                recs.append(dict(name=f'{folder}/{path.name}', y=label,
                                 X=np.array([[w[k] for k in feats] for w in wins])))
    return recs


def train(feats):
    t_start = time.time()
    print(f'Preprocessing recordings ({WIN:.0f} s windows, {STEP:.0f} s step)…')
    recs = load_dataset(feats)
    n0, n1 = sum(r['y'] == 0 for r in recs), sum(r['y'] == 1 for r in recs)
    if not n0 or not n1:
        sys.exit('Need recordings in BOTH data/normal and data/fatigue.')

    print('\nHonest test: leave-one-recording-out')
    results, conf = [], np.zeros((2, 2), int)
    for i, r in enumerate(recs):
        train_set = [q for j, q in enumerate(recs) if j != i]
        X = np.vstack([q['X'] for q in train_set]); y = np.concatenate([[q['y']] * len(q['X']) for q in train_set])
        m = fit_logreg(X, y)
        p = float(prob(m, r['X']).mean())
        pred = int(p >= 0.5)
        conf[r['y'], pred] += 1
        results.append(dict(recording=r['name'], true='fatigue' if r['y'] else 'normal',
                            predicted='fatigue' if pred else 'normal', p_fatigue=round(p, 3), windows=len(r['X'])))
        print(f"  {r['name']:30s} true {results[-1]['true']:8s} -> predicted {results[-1]['predicted']:8s} "
              f"P(fatigue)={p:.2f} {'OK' if pred == r['y'] else 'WRONG'}")
    acc = np.trace(conf) / conf.sum()
    print(f'\n  Accuracy {np.trace(conf)}/{conf.sum()} = {acc:.0%}')
    print(f'  Confusion (rows = true normal/fatigue, cols = predicted normal/fatigue):\n  {conf[0]}\n  {conf[1]}')

    X = np.vstack([r['X'] for r in recs]); y = np.concatenate([[r['y']] * len(r['X']) for r in recs])
    model = fit_logreg(X, y)
    model.update(features=feats, window_s=WIN, step_s=STEP, threshold=0.5,
                 trained_at=datetime.now().isoformat(timespec='seconds'),
                 training=dict(recordings=len(recs), normal=n0, fatigue=n1, windows=int(len(X))),
                 evaluation=dict(method='leave-one-recording-out', accuracy=float(acc), confusion=conf.tolist(),
                                 details=results))
    MODEL_PATH.parent.mkdir(exist_ok=True)
    MODEL_PATH.write_text(json.dumps(model, indent=1))
    print('\nWhat the final model uses (+ pushes toward fatigue):')
    for k, c in sorted(zip(feats, model['coef']), key=lambda kv: -abs(kv[1])):
        print(f'  {k:9s} {c:+.2f}')
    print(f'\nSaved {MODEL_PATH.relative_to(ROOT)}  (total {time.time() - t_start:.0f} s on this device)')


def split(train_names, test_names, feats):
    """Train on the named recordings only, then test on other named recordings (e.g. normal/third_attempt.csv)."""
    print('Preprocessing recordings…')
    recs = {r['name']: r for r in load_dataset(feats)}
    missing = [n for n in train_names + test_names if n not in recs]
    if missing:
        sys.exit(f'Not found (or not enough clean signal): {missing}')
    overlap = set(train_names) & set(test_names)
    if overlap:
        print(f'\nWARNING: {sorted(overlap)} is in BOTH train and test, so its test result is not honest.')
    tr = [recs[n] for n in train_names]
    X = np.vstack([r['X'] for r in tr]); y = np.concatenate([[r['y']] * len(r['X']) for r in tr])
    model = fit_logreg(X, y)
    print(f"\nTrained on {len(tr)} recordings ({int((y == 0).sum())} normal / {int(y.sum())} fatigue windows). Test:")
    for n in test_names:
        r = recs[n]; ps = prob(model, r['X']); p = float(ps.mean()); pred = int(p >= 0.5)
        print(f"  {n:30s} true {'fatigue' if r['y'] else 'normal':8s} -> predicted {'fatigue' if pred else 'normal':8s} "
              f"P(fatigue)={p:.2f} {'OK' if pred == r['y'] else 'WRONG'}   windows {np.round(ps, 2).tolist()}")
    print('\nModel weights (+ pushes toward fatigue):')
    for k, c in sorted(zip(feats, model['coef']), key=lambda kv: -abs(kv[1])):
        print(f'  {k:9s} {c:+.2f}')


def predict(paths):
    model = json.loads(MODEL_PATH.read_text())
    for path in paths:
        t0 = time.time()
        wins, p = window_features(path)
        if not wins:
            print(f'{path}: not enough clean signal'); continue
        ps = prob(model, [[w[k] for k in model['features']] for w in wins])
        pm = float(ps.mean())
        print(f"{path}: {'FATIGUE' if pm >= model['threshold'] else 'NORMAL'}  P(fatigue)={pm:.2f}  "
              f"({len(wins)} windows, per window {np.round(ps, 2).tolist()}, {time.time() - t0:.1f} s)")


if __name__ == '__main__':
    args = sys.argv[1:]
    if args[:1] == ['train']:
        train(HRV_FEATURES if '--hrv-only' in args else ALL_FEATURES)
    elif args[:1] == ['split'] and '--train' in args and '--test' in args:
        i, j = args.index('--train'), args.index('--test')
        names = lambda a, b: [x for x in args[a + 1:b] if not x.startswith('--')]
        split(names(i, j if j > i else len(args)), names(j, i if i > j else len(args)),
              HRV_FEATURES if '--hrv-only' in args else ALL_FEATURES)
    elif args[:1] == ['predict'] and len(args) > 1:
        predict(args[1:])
    else:
        print(__doc__)
