"""Preprocessing + baseline-comparison fatigue model for TI AFE4950 exports.

Same steps as dataset_exploration.ipynb (Track 1 manual §7–9):
  P1–P4  DC removal, 0.5 Hz high-pass, 60 Hz notch, 40 Hz low-pass
  P5     artifact mask (saturation, motion spikes > 2× typical R-peak)
  P6     Pan–Tompkins R-peaks + polarity/amplitude validation
  P7     RR cleaning (40–180 bpm, ±20 % of local median)
  P7b    trim unstable start/end
Features (stable part only): HR, HR range, SDNN, RMSSD, pNN50, R-peak height, T/R ratio, R-to-T time.
The model works on the CHANGE from the person's own baseline: delta = features(new) − features(baseline).
"""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

FS_ECG, FS_PPG = 500, 50
FEATURES = ['HR', 'HR_range', 'SDNN', 'RMSSD', 'pNN50', 'R_mV', 'T_over_R', 'RT_ms']
LABELS = {'HR': ('Heart rate', 'bpm'), 'HR_range': ('Heart-rate range', 'bpm'), 'SDNN': ('SDNN', 'ms'),
          'RMSSD': ('RMSSD', 'ms'), 'pNN50': ('pNN50', '%'), 'R_mV': ('R-peak height', 'mV'),
          'T_over_R': ('T-wave / R-peak ratio', ''), 'RT_ms': ('R-to-T time', 'ms')}
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'models' / 'delta_model.json'
PRE, POST = 0.2, 0.4                       # median-beat window around each R-peak (s)


# ---------------- loading ----------------
def load_recording(src):
    """TI GUI export (path or bytes) -> (ecg_volts, optical_volts). index_col=False: trailing comma (manual Step 24)."""
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    df = pd.read_csv(src, index_col=False)

    def volts(ch):
        if f'{ch}_Value' in df:
            return df[f'{ch}_Value'].dropna().to_numpy(float)
        raw = df[f'{ch}_RAW_Value'].dropna().to_numpy(float)            # 24-bit two's complement
        return np.where(raw >= 2**23, raw - 2**24, raw) / (2**23 / 4.8)
    return volts('ECG'), volts('TIA1-3')


# ---------------- preprocessing ----------------
def butter(x, fc, kind, fs=FS_ECG, order=2):
    b, a = signal.butter(order, fc, btype=kind, fs=fs)
    return signal.filtfilt(b, a, x)


def clean_ecg(ecg_v):
    x = 1000 * (ecg_v - np.median(ecg_v))                                 # P1: mV, no DC
    x = butter(x, 0.5, 'highpass')                                        # P2: baseline wander
    b, a = signal.iirnotch(60, 30, FS_ECG); x = signal.filtfilt(b, a, x)  # P3: mains
    return butter(x, 40, 'lowpass')                                       # P4: EMG / HF noise


def artifact_mask(ecg_v, ecg_clean, k=2.0):
    sec = np.arange(len(ecg_clean)) // FS_ECG
    peak = pd.Series(np.abs(ecg_clean)).groupby(sec).max()
    bad = np.isin(sec, peak.index[peak > k * peak.median()]) | (np.abs(ecg_v) > 1.19)
    return np.convolve(bad, np.ones(FS_ECG), 'same') > 0


def pan_tompkins(ecg_clean, bad):
    x = np.where(bad, 0, ecg_clean)
    bp = butter(x, [5, 20], 'bandpass')
    mwi = np.convolve(np.gradient(bp)**2, np.ones(int(.12*FS_ECG))/int(.12*FS_ECG), 'same')
    ref = mwi[~bad] if (~bad).any() else mwi
    pk, _ = signal.find_peaks(mwi, distance=int(.33*FS_ECG), height=.3*np.percentile(ref, 99))
    hw = int(.06*FS_ECG)
    pk = np.array([p for p in pk if hw <= p < len(x) - hw], int)
    if not len(pk):
        return pk
    absmax = np.array([p - hw + np.argmax(np.abs(ecg_clean[p-hw:p+hw])) for p in pk])
    pol = np.sign(np.median(ecg_clean[absmax])) or 1.0
    pk = np.array([p - hw + np.argmax(pol * ecg_clean[p-hw:p+hw]) for p in pk], int)
    pk = pk[~bad[pk]]
    amp = pol * ecg_clean[pk]; med = np.median(amp) if len(pk) else 0
    return pk[(amp >= .5*med) & (amp <= 1.6*med)]


def clean_rr(t_peaks, bad, tol=0.2):
    rr = np.diff(t_peaks); t = t_peaks[1:]
    ok = (rr > .33) & (rr < 1.5)
    ok &= ~np.array([bad[int(a*FS_ECG):int(b*FS_ECG)+1].any() for a, b in zip(t_peaks[:-1], t_peaks[1:])], bool)
    med = pd.Series(np.where(ok, rr, np.nan)).rolling(5, center=True, min_periods=1).median().values
    ok &= np.abs(rr - med) <= tol * med
    return t, rr, ok


def stable_range(ecg_v, ecg_clean, rpeaks, trim_min=2, run=3):
    ecg_mv = 1000 * (ecg_v - np.median(ecg_v)); sec = np.arange(len(ecg_mv)) // FS_ECG
    drift = pd.Series(butter(ecg_mv, 1.0, 'lowpass')).groupby(sec).agg(np.ptp).values
    peak = pd.Series(np.abs(ecg_clean)).groupby(sec).max().values
    r_amp = np.median(np.abs(ecg_clean[rpeaks])) if len(rpeaks) else np.inf
    bad = (drift > .3*r_amp) | (peak > 1.6*r_amp) | (peak < .5*r_amp)
    s, e = trim_min, len(bad) - trim_min
    while s < e and bad[s:s+run].any(): s += 1
    while e > s and bad[e-run:e].any(): e -= 1
    return (float(s), float(e)), bad


# ---------------- features ----------------
def features_from(rr, median_beat):
    ms = 1000 * rr; d = np.diff(ms)
    i0 = int(PRE * FS_ECG); base = np.median(median_beat[:int(.08*FS_ECG)]); R = median_beat[i0] - base
    tseg = median_beat[i0 + int(.15*FS_ECG): i0 + int(.38*FS_ECG)] - base; ti = int(np.argmax(np.abs(tseg)))
    return dict(HR=60000/ms.mean(), HR_range=60/rr.min() - 60/rr.max(), SDNN=ms.std(ddof=1),
                RMSSD=np.sqrt(np.mean(d**2)), pNN50=100*np.mean(np.abs(d) > 50),
                R_mV=R, T_over_R=tseg[ti]/R, RT_ms=(int(.15*FS_ECG) + ti) / FS_ECG * 1000)


def analyze(ecg_v, opt_v=None, segment=None):
    """Full preprocessing + features. segment=(start_s, end_s) restricts features to part of the stable range."""
    t = np.arange(len(ecg_v)) / FS_ECG
    ecg_c = clean_ecg(ecg_v)
    mask = artifact_mask(ecg_v, ecg_c)
    rp = pan_tompkins(ecg_c, mask)
    t_rr, rr, ok = clean_rr(t[rp], mask) if len(rp) > 2 else (np.array([]), np.array([]), np.array([], bool))
    keep, unstable = stable_range(ecg_v, ecg_c, rp)
    a, b = segment if segment else keep
    sel = ok & (t_rr > a) & (t_rr < b)
    pk = rp[(t[rp] > a + PRE) & (t[rp] < b - POST)]
    pk = pk[~np.array([mask[p-int(PRE*FS_ECG):p+int(POST*FS_ECG)].any() for p in pk], bool)]
    beats = np.array([ecg_c[p-int(PRE*FS_ECG):p+int(POST*FS_ECG)] for p in pk])
    opt_pp = float(np.ptp(opt_v) * 1000) if opt_v is not None and len(opt_v) else 0.0
    q = dict(duration_s=len(ecg_v)/FS_ECG, stable_start=keep[0], stable_end=keep[1], stable_s=keep[1]-keep[0],
             r_peaks=int(len(rp)), clean_beats=int(sel.sum()) + 1, masked_pct=100*mask.mean(),
             saturated_pct=100*np.mean(np.abs(ecg_v) > 1.19), optical_present=opt_pp > 5, optical_pp_mV=opt_pp)
    ok_enough = sel.sum() >= 15 and len(beats) >= 10
    feats = features_from(rr[sel], np.median(beats, 0)) if ok_enough else None
    return dict(t=t, ecg_clean=ecg_c, mask=mask, rpeaks=rp, t_rr=t_rr[sel], rr=rr[sel], keep=keep, unstable=unstable,
                median_beat=np.median(beats, 0) if len(beats) else None, features=feats, quality=q, usable=ok_enough)


def quality_warnings(res, role):
    q, f, w = res['quality'], res['features'], []
    if not res['usable']:
        w.append(('error', 'Not enough clean signal: need ≥ 15 clean beats in the stable part. Check electrode contact and re-record.'))
        return w
    if q['stable_s'] < 60:
        w.append(('warn', f"Only {q['stable_s']:.0f} s of stable signal. Record 3–5 min for reliable HRV (LF/HF and breathing need ≥ 2 min)."))
    if not q['optical_present']:
        w.append(('warn', 'Optical (PPG) channel is empty: sensor not connected or LED off. Only ECG is used.'))
    if q['masked_pct'] > 10:
        w.append(('warn', f"{q['masked_pct']:.0f} % of the ECG was masked as motion/saturation. Sit still during recording."))
    if role == 'baseline' and f['HR'] > 100:
        w.append(('warn', f"Baseline heart rate is {f['HR']:.0f} bpm. A rested baseline is usually < 90 bpm. Sit quietly 5 min and re-record if possible."))
    return w


def beat_similarity(a, b):
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    z = lambda x: (x - x.mean()) / x.std()
    return float(0.5*np.corrcoef(z(a), z(b))[0, 1] + 0.5*np.corrcoef(np.diff(a), np.diff(b))[0, 1])


# ---------------- model ----------------
class DeltaModel:
    def __init__(self, path=MODEL_PATH):
        m = json.loads(Path(path).read_text())
        self.meta = m
        self.features = m['features']
        self.mean, self.scale = np.array(m['mean']), np.array(m['scale'])
        self.coef, self.intercept, self.threshold = np.array(m['coef']), m['intercept'], m['threshold']
        self.watch_threshold = m.get('watch_threshold', self.threshold)

    def level(self, p):
        """'fatigue' = serious (alarm), 'watch' = mild change from baseline, 'normal'."""
        return 'fatigue' if p >= self.threshold else 'watch' if p >= self.watch_threshold else 'normal'

    def predict(self, new_feats, base_feats):
        delta = np.array([new_feats[k] - base_feats[k] for k in self.features])
        z = (delta - self.mean) / self.scale
        p = float(1 / (1 + np.exp(-(z @ self.coef + self.intercept))))
        contrib = pd.DataFrame({'feature': self.features, 'baseline': [base_feats[k] for k in self.features],
                                'new': [new_feats[k] for k in self.features], 'change': delta,
                                'push_to_fatigue': z * self.coef})
        contrib['pct_change'] = 100 * contrib.change / contrib.baseline.abs().replace(0, np.nan)
        return p, contrib.sort_values('push_to_fatigue', key=abs, ascending=False)
