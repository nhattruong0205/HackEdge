"""Model B (from the slides): transfer learning  WESAD (pre-train) -> DROZY (fine-tune) -> our recordings.

All models use the same input as our personal-shift model: the change of heart rate, SDNN and RMSSD
from the person's own baseline, as log ratios  log(new / baseline).

  Source 1  WESAD  (Schmidt et al. 2018): chest ECG, 15 people, baseline vs. TSST stress.
  Source 2  DROZY  (Massoz et al. 2016):  PSG ECG, 14 people, 3 sessions of increasing sleep deprivation,
                                          rated with the Karolinska Sleepiness Scale (KSS).
  Target    our AFE4950 recordings (board_manifest.csv): 2 people, 11 recordings.

Steps
  1. Extract 60-s windows from WESAD and DROZY, compute HR / SDNN / RMSSD with OUR preprocessing
     (fatigue_pipeline.analyze, after resampling to 500 Hz), and turn them into baseline-shift pairs.
  2. Pre-train a logistic regression on WESAD; fine-tune on DROZY with an L2 prior pulling the weights
     towards the pre-trained ones; fine-tune again on our data the same way.
     (Also: a small 3-8-1 neural network whose hidden layer is pre-trained on WESAD and frozen,
     with only the output layer re-trained, as drawn on the slides.)
  3. Test on our recordings with the same leave-one-recording-out protocol as the current model.

The external datasets are licensed for research/evaluation only and must NOT be added to this repo.
Usage:  python3 transfer_model_b.py /path/to/WESAD /path/to/DROZY
"""
import csv
import io
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from fatigue_pipeline import analyze, load_recording

ROOT = Path(__file__).resolve().parent
SHIFT = ['HR', 'SDNN', 'RMSSD']
WIN = 60.0
FS = 500


# ---------------------------------------------------------------- readers
class _NumpyOnly(pickle.Unpickler):
    """WESAD ships as pickles. Only allow the numpy classes needed to rebuild arrays (no arbitrary code)."""
    OK = {('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'),
          ('numpy', 'ndarray'), ('numpy', 'dtype'), ('numpy.core.multiarray', 'scalar'),
          ('numpy._core.multiarray', 'scalar'), ('_codecs', 'encode')}

    def find_class(self, module, name):
        if (module, name) in self.OK:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'blocked {module}.{name}')


def read_wesad(path):
    with open(path, 'rb') as fh:
        d = _NumpyOnly(io.BytesIO(fh.read()), encoding='latin1').load()
    return d['signal']['chest']['ECG'].ravel().astype(float), np.asarray(d['label']).ravel()   # ECG in mV, 700 Hz


def read_edf(path, channel='ECG'):
    """Minimal EDF reader -> (signal in physical units, sampling rate)."""
    raw = Path(path).read_bytes()
    n_rec, dur, ns = int(raw[236:244]), float(raw[244:252]), int(raw[252:256])
    h = raw[256:256 + ns * 256]
    field = lambda off, w, i: h[ns * off + i * w: ns * off + (i + 1) * w].decode('latin1').strip()
    labels = [field(0, 16, i) for i in range(ns)]
    o = 16 + 80 + 8                                                  # label, transducer, dimension
    pmin = [float(field(o, 8, i)) for i in range(ns)]; o += 8
    pmax = [float(field(o, 8, i)) for i in range(ns)]; o += 8
    dmin = [float(field(o, 8, i)) for i in range(ns)]; o += 8
    dmax = [float(field(o, 8, i)) for i in range(ns)]; o += 8 + 80  # + prefiltering
    spr = [int(field(o, 8, i)) for i in range(ns)]
    k = next(i for i, l in enumerate(labels) if channel.lower() in l.lower())
    data = np.frombuffer(raw[256 + ns * 256:], dtype='<i2')[: n_rec * sum(spr)].reshape(n_rec, sum(spr))
    s = sum(spr[:k]); x = data[:, s:s + spr[k]].ravel().astype(float)
    x = (x - dmin[k]) / (dmax[k] - dmin[k]) * (pmax[k] - pmin[k]) + pmin[k]
    return x, spr[k] / dur, labels[k]


# ---------------------------------------------------------------- features
def windows(ecg_mv, fs_in, up, down):
    """Resample to 500 Hz, cut into 60-s windows, HR/SDNN/RMSSD per window with our preprocessing."""
    x = resample_poly(ecg_mv, up, down) / 1000.0                     # mV -> V (what our pipeline expects)
    n, out = int(WIN * FS), []
    for s in range(0, len(x) - n + 1, n):
        r = analyze(x[s:s + n])
        f = r['features']
        if r['usable'] and r['quality']['masked_pct'] < 20 and 40 < f['HR'] < 160:
            out.append([f[k] for k in SHIFT])
    return np.array(out)


def shift_pairs(base, test, label, rng, max_n=None):
    """Each test window vs the mean of the baseline windows -> log-ratio shift."""
    ref = base.mean(0)
    X = np.log(test / ref)
    if max_n and len(X) > max_n:
        X = X[rng.choice(len(X), max_n, replace=False)]
    return X, np.full(len(X), label, float)


def null_pairs(base, rng, max_n=None):
    """No-change examples: half of the baseline windows vs the mean of the other half."""
    idx = rng.permutation(len(base)); a, b = base[idx[: len(idx) // 2]], base[idx[len(idx) // 2:]]
    X = np.r_[np.log(a / b.mean(0)), np.log(b / a.mean(0))]
    if max_n and len(X) > max_n:
        X = X[rng.choice(len(X), max_n, replace=False)]
    return X, np.zeros(len(X))


def wesad_dataset(root, rng):
    X, y, t0 = [], [], time.time()
    for p in sorted(Path(root).glob('S*/S*.pkl')):
        ecg, lab = read_wesad(p)
        ecg = ecg[: len(lab)]
        base = windows(ecg[lab == 1], 700, 5, 7)                      # 1 = baseline
        stress = windows(ecg[lab == 2], 700, 5, 7)                    # 2 = TSST stress
        if len(base) < 4 or len(stress) < 2:
            continue
        for Xi, yi in (shift_pairs(base, stress, 1, rng), null_pairs(base, rng, max_n=len(stress))):
            X.append(Xi); y.append(yi)
        print(f'  WESAD {p.stem}: baseline {len(base)} windows, stress {len(stress)} windows  '
              f'HR {base[:, 0].mean():.0f} -> {stress[:, 0].mean():.0f}')
    print(f'  WESAD done in {time.time() - t0:.0f} s')
    return np.vstack(X), np.concatenate(y)


def drozy_dataset(root, rng, hi=7, lo=5):
    root = Path(root)
    kss = np.loadtxt(next(root.rglob('KSS.txt')))
    psg = next(root.rglob('psg'))
    X, y, t0 = [], [], time.time()
    for s in range(1, len(kss) + 1):
        w = {}
        for t in (1, 2, 3):
            f = psg / f'{s}-{t}.edf'
            if f.exists() and kss[s - 1, t - 1] > 0:
                ecg, fs, _ = read_edf(f)
                w[t] = windows(ecg * (1000 if np.abs(ecg).max() < 20 else 1), fs, 125, 128)
        if 1 not in w or len(w[1]) < 4:
            continue
        base = w[1]
        X_, y_ = [null_pairs(base, rng)[0]], [np.zeros(len(null_pairs(base, rng)[0]))]
        for t in (2, 3):
            if t in w and len(w[t]):
                k = kss[s - 1, t - 1]
                if k >= hi and k > kss[s - 1, 0]:
                    Xi, yi = shift_pairs(base, w[t], 1, rng); X_.append(Xi); y_.append(yi)
                elif k <= lo:
                    Xi, yi = shift_pairs(base, w[t], 0, rng); X_.append(Xi); y_.append(yi)
        X += X_; y += y_
        print(f'  DROZY subject {s:2d}: KSS {kss[s - 1].astype(int).tolist()}  HR per test '
              + ' '.join(f'{t}:{w[t][:, 0].mean():.0f}' for t in w))
    print(f'  DROZY done in {time.time() - t0:.0f} s')
    return np.vstack(X), np.concatenate(y)


def our_dataset():
    rows = []
    with open(ROOT / 'board_manifest.csv', newline='') as fh:
        for r in csv.DictReader(fh):
            if r['condition'] in ('normal', 'fatigue'):
                f = analyze(*load_recording(ROOT / r['file']))['features']
                rows.append(dict(file=r['file'], person=r['person'], y=int(r['condition'] == 'fatigue'),
                                 f=np.array([f[k] for k in SHIFT])))
    return rows


def our_pairs(rows):
    X, y = [], []
    for p in {r['person'] for r in rows}:
        mine = [r for r in rows if r['person'] == p]; normals = [r for r in mine if r['y'] == 0]
        for a in mine:
            for b in normals:
                if a is not b:
                    X.append(np.log(a['f'] / b['f'])); y.append(a['y'])
    return np.array(X), np.array(y, float)


# ---------------------------------------------------------------- models
def fit_lr(X, y, prior=None, lam=1e-3, iters=100):
    """Class-balanced logistic regression on raw log-ratios, L2 penalty towards `prior` (transfer)."""
    Z = np.c_[X, np.ones(len(X))]
    w0 = np.zeros(Z.shape[1]) if prior is None else np.asarray(prior, float)
    cw = np.where(y == 1, .5 / max(y.mean(), 1e-9), .5 / max(1 - y.mean(), 1e-9)) / len(y)
    w = w0.copy(); R = np.full(len(w), lam); R[-1] = lam * 0.01
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (cw * (p - y)) + R * (w - w0)
        H = (Z * (cw * p * (1 - p))[:, None]).T @ Z + np.diag(R)
        step = np.linalg.solve(H, g); w -= step
        if np.abs(step).max() < 1e-10:
            break
    return w


def p_lr(w, X):
    return 1 / (1 + np.exp(-np.c_[X, np.ones(len(X))] @ w))


def fit_mlp(X, y, hidden=8, epochs=3000, lr=0.05, W1=None, b1=None, seed=0, freeze=False):
    """Tiny 3-8-1 network (tanh). freeze=True keeps the pre-trained hidden layer fixed (slides: frozen/tuned)."""
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, .5, (X.shape[1], hidden)) if W1 is None else W1.copy()
    b1 = np.zeros(hidden) if b1 is None else b1.copy()
    W2, b2 = rng.normal(0, .5, hidden), 0.0
    cw = np.where(y == 1, .5 / max(y.mean(), 1e-9), .5 / max(1 - y.mean(), 1e-9)) / len(y)
    for _ in range(epochs):
        H = np.tanh(X @ W1 + b1); p = 1 / (1 + np.exp(-(H @ W2 + b2)))
        d = cw * (p - y)
        gW2, gb2 = H.T @ d + 1e-3 * W2, d.sum()
        if not freeze:
            dH = np.outer(d, W2) * (1 - H ** 2)
            W1 -= lr * (X.T @ dH + 1e-3 * W1); b1 -= lr * dH.sum(0)
        W2 -= lr * gW2; b2 -= lr * gb2
    return dict(W1=W1, b1=b1, W2=W2, b2=b2)


def p_mlp(m, X):
    return 1 / (1 + np.exp(-(np.tanh(X @ m['W1'] + m['b1']) @ m['W2'] + m['b2'])))


# ---------------------------------------------------------------- evaluation on our recordings
def evaluate(rows, predict_fn, train_fn=None):
    """Leave-one-recording-out on our data. train_fn(X, y) -> model, or None for zero-shot."""
    res = []
    for held in rows:
        rest = [r for r in rows if r is not held]
        m = train_fn(*our_pairs(rest)) if train_fn else None
        bases = [r for r in rest if r['person'] == held['person'] and r['y'] == 0]
        p = float(np.mean([predict_fn(m, np.log(held['f'] / b['f'])[None])[0] for b in bases]))
        res.append((held, p))
    return res


def summary(name, res):
    y = np.array([r['y'] for r, _ in res]); p = np.array([p for _, p in res])
    acc = np.mean((p >= .5) == y)
    pos, neg = p[y == 1], p[y == 0]
    auc = np.mean([(a > b) + .5 * (a == b) for a in pos for b in neg])
    brier = np.mean((p - y) ** 2)
    margin = min(pos.min() - .5, .5 - neg.max())
    print(f'  {name:46s} acc {int(acc * len(y))}/{len(y)}  AUC {auc:.2f}  Brier {brier:.3f}  '
          f'worst margin {margin:+.2f}   fatigue {np.round(pos, 2).tolist()}  rest {np.round(neg, 2).tolist()}')
    return dict(name=name, acc=acc, auc=auc, brier=brier, margin=margin)


def main(wesad_root, drozy_root):
    rng = np.random.default_rng(0)
    print('1) WESAD (pre-training data)'); Xw, yw = wesad_dataset(wesad_root, rng)
    print(f'   {int(yw.sum())} stress pairs, {int((1 - yw).sum())} no-change pairs')
    print('2) DROZY (fine-tuning data)'); Xd, yd = drozy_dataset(drozy_root, rng)
    print(f'   {int(yd.sum())} sleepy (KSS>=7) pairs, {int((1 - yd).sum())} no-change pairs')
    for nm, X_, y_ in (('WESAD stress', Xw, yw), ('DROZY sleepy', Xd, yd)):
        pct = 100 * (np.exp(X_[y_ == 1].mean(0)) - 1)
        print(f'   mean shift in {nm}: ' + '  '.join(f'{k} {v:+.0f}%' for k, v in zip(SHIFT, pct)))
    rows = our_dataset()
    Xo, yo = our_pairs(rows)
    pct = 100 * (np.exp(Xo[yo == 1].mean(0)) - 1)
    print('   mean shift in our fatigue: ' + '  '.join(f'{k} {v:+.0f}%' for k, v in zip(SHIFT, pct)))

    w_w = fit_lr(Xw, yw)
    w_wd = fit_lr(Xd, yd, prior=w_w, lam=0.01)
    w_d = fit_lr(Xd, yd)
    print('\nWeights [HR, SDNN, RMSSD, bias] (+ = towards fatigue/stress):')
    for nm, w in (('WESAD', w_w), ('DROZY only', w_d), ('WESAD -> DROZY', w_wd)):
        print(f'  {nm:16s} {np.round(w, 2).tolist()}')

    print('\n3) Test on our 11 recordings (leave-one-recording-out):')
    lr = lambda m, X: p_lr(m, X)
    out = [summary('Current model (ours only)', evaluate(rows, lr, lambda X, y: fit_lr(X, y))),
           summary('Model B zero-shot: WESAD only', evaluate(rows, lambda m, X: p_lr(w_w, X))),
           summary('Model B zero-shot: DROZY only', evaluate(rows, lambda m, X: p_lr(w_d, X))),
           summary('Model B zero-shot: WESAD -> DROZY', evaluate(rows, lambda m, X: p_lr(w_wd, X))),
           summary('Model B: W->D -> fine-tune ours (prior 0.01)', evaluate(rows, lr, lambda X, y: fit_lr(X, y, prior=w_wd, lam=0.01))),
           summary('Model B: W->D -> fine-tune ours (prior 0.1)', evaluate(rows, lr, lambda X, y: fit_lr(X, y, prior=w_wd, lam=0.1))),
           summary('Model B: WESAD -> fine-tune ours (prior 0.01)', evaluate(rows, lr, lambda X, y: fit_lr(X, y, prior=w_w, lam=0.01)))]

    net = fit_mlp(Xw, yw)
    netd = fit_mlp(Xd, yd, W1=net['W1'], b1=net['b1'], freeze=True)
    mp = lambda m, X: p_mlp(m, X)
    out += [summary('NN 3-8-1 zero-shot: WESAD -> DROZY (frozen)', evaluate(rows, lambda m, X: p_mlp(netd, X))),
            summary('NN 3-8-1: WESAD -> DROZY -> ours (frozen)', evaluate(rows, mp, lambda X, y: fit_mlp(X, y, W1=net['W1'], b1=net['b1'], freeze=True))),
            summary('NN 3-8-1: ours only', evaluate(rows, mp, lambda X, y: fit_mlp(X, y)))]

    print('\nHow much of our data does each model need? (train on ONE subject, test on the OTHER)')
    for nm, tr in (('Current model (ours only)', lambda X, y: fit_lr(X, y)),
                   ('Model B: WESAD -> DROZY -> ours', lambda X, y: fit_lr(X, y, prior=w_wd, lam=0.01))):
        res = []
        for test_p in ('A', 'C'):
            train_rows = [r for r in rows if r['person'] != test_p]
            m = tr(*our_pairs(train_rows))
            for held in (r for r in rows if r['person'] == test_p):
                bases = [r for r in rows if r['person'] == test_p and r['y'] == 0 and r is not held]
                res.append((held, float(np.mean([p_lr(m, np.log(held['f'] / b['f'])[None])[0] for b in bases]))))
        summary(nm + ' [cross-subject]', res)
    return out


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
