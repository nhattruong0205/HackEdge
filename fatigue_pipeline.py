"""Preprocessing + personal-baseline & zero-baseline physiological fatigue models for TI AFE4950 exports.

Same preprocessing steps as dataset_exploration.ipynb (Track 1 manual §7–9):
  P1–P4  DC removal, 0.5 Hz high-pass, 60 Hz notch, 40 Hz low-pass
  P5     artifact mask (saturation, motion spikes > 2× typical R-peak)
  P6     Pan–Tompkins R-peaks + polarity/amplitude validation
  P7     RR cleaning (40–180 bpm, ±20 % of local median)
  P7b    trim unstable start/end
  P8     Optical (PPG) 0.5–4 Hz band-pass, systolic pulse peak detection, and ECG–PPG Pulse Arrival Time (PAT)

Features (stable part only):
  Time-domain HRV      : HR, HR_range, SDNN, RMSSD, pNN50
  Normalized/Non-linear: CVSD (100*RMSSD/mean_RR), SD1_SD2 (Poincaré ratio), log_SI (log10 Baevsky Stress Index)
  Frequency-domain HRV : LF_HF (0.04–0.15 Hz / 0.15–0.40 Hz power ratio), HF_nu (normalized vagal HF power, %)
  Waveform morphology  : R_mV, T_over_R, RT_ms, RTc_ms (Bazett rate-corrected R-to-T time)
  Optical PPG & Fusion : PPG_HR, PAT_ms (ECG R-peak -> PPG peak delay), PAT_sd_ms, PPG_amp_cv

Supports two classification modes:
  1. Personal-Baseline Delta Mode: delta = features(new) − features(baseline), with automatic R_mV gating when
     beat_similarity < 0.98 so electrode placement shifts cannot trigger false alarms.
  2. Zero-Baseline Single-Recording Mode: classifies a brand-new person from a single recording using
     placement-invariant physiological HRV + morphology features (SINGLE_FEATURES).
"""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

FS_ECG, FS_PPG = 500, 50
FEATURES = [
    'HR', 'HR_range', 'SDNN', 'RMSSD', 'pNN50',
    'CVSD', 'SD1_SD2', 'log_SI', 'LF_HF', 'HF_nu',
    'R_mV', 'T_over_R', 'RT_ms', 'RTc_ms',
]
SINGLE_FEATURES = [
    'HR', 'SDNN', 'RMSSD', 'pNN50',
    'CVSD', 'SD1_SD2', 'log_SI', 'LF_HF', 'HF_nu', 'RTc_ms',
]
PPG_FEATURES = ['PPG_HR', 'PAT_ms', 'PAT_sd_ms', 'PPG_amp_cv']

LABELS = {
    'HR': ('Heart rate', 'bpm'),
    'HR_range': ('Heart-rate range', 'bpm'),
    'SDNN': ('SDNN (overall HRV)', 'ms'),
    'RMSSD': ('RMSSD (vagal beat-to-beat HRV)', 'ms'),
    'pNN50': ('pNN50', '%'),
    'CVSD': ('Normalized HRV (CVSD = RMSSD/RR)', '%'),
    'SD1_SD2': ('Poincaré SD1/SD2 ratio', ''),
    'log_SI': ('Baevsky Stress Index (log10 SI)', ''),
    'LF_HF': ('Sympathetic/Vagal LF/HF ratio', ''),
    'HF_nu': ('Normalized Vagal HF power', '%'),
    'R_mV': ('R-peak height', 'mV'),
    'T_over_R': ('T-wave / R-peak ratio', ''),
    'RT_ms': ('R-to-T time', 'ms'),
    'RTc_ms': ('Rate-corrected R-to-T (RTc)', 'ms'),
    'PPG_HR': ('Optical (PPG) pulse rate', 'bpm'),
    'PAT_ms': ('ECG→PPG Pulse Arrival Time (PAT)', 'ms'),
    'PAT_sd_ms': ('PAT variability (SD)', 'ms'),
    'PPG_amp_cv': ('PPG pulse amplitude CV', '%'),
}

# Expected healthy rested reference ranges for single-recording display
HEALTHY_NORMS = {
    'HR': 70.0, 'SDNN': 38.0, 'RMSSD': 34.0, 'pNN50': 14.0,
    'CVSD': 3.8, 'SD1_SD2': 0.48, 'log_SI': 2.35,
    'LF_HF': 1.8, 'HF_nu': 36.0, 'RTc_ms': 225.0,
}

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'models' / 'delta_model.json'
PRE, POST = 0.2, 0.4                       # median-beat window around each R-peak (s)


def resolve_recording_path(p):
    """Resolve 'normal/foo.csv' or 'data/normal/foo.csv' regardless of whether folders live in ROOT or ROOT/data."""
    path = Path(p)
    if path.is_absolute() and path.exists():
        return path
    for candidate in (ROOT / path, ROOT / 'data' / path):
        if candidate.exists():
            return candidate
    if len(path.parts) >= 2 and path.parts[0] == 'data':
        alt = ROOT / Path(*path.parts[1:])
        if alt.exists():
            return alt
    return ROOT / path


# ---------------- loading ----------------
def load_recording(src):
    """TI GUI export (path or bytes) -> (ecg_volts, optical_volts). index_col=False: trailing comma (manual Step 24)."""
    if isinstance(src, (str, Path)):
        src = resolve_recording_path(src)
    elif isinstance(src, (bytes, bytearray)):
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


def analyze_optical(opt_v, t_rpeaks, keep):
    """P8: Optical (PPG) filtering, peak detection, and ECG->PPG Pulse Arrival Time (PAT)."""
    if opt_v is None or len(opt_v) < FS_PPG * 4:
        return dict(present=False, pp_mV=0.0, t_opt=np.array([]), ppg_clean=np.array([]),
                    ppg_peaks=np.array([], int), pat_pairs=[], features={k: None for k in PPG_FEATURES})
    a, b = keep
    t_opt = np.arange(len(opt_v)) / FS_PPG
    w = (t_opt >= a) & (t_opt <= b)
    seg = opt_v[w] if w.any() else opt_v
    pp_mv = float(np.ptp(seg) * 1000.0)
    present = pp_mv > 5.0
    if not present:
        return dict(present=False, pp_mV=pp_mv, t_opt=t_opt, ppg_clean=1000.0 * (opt_v - np.median(opt_v)),
                    ppg_peaks=np.array([], int), pat_pairs=[], features={k: None for k in PPG_FEATURES})

    # Invert & bandpass 0.5-4.0 Hz in mV so systolic blood volume peaks point upward
    x = -butter(1000.0 * (opt_v - np.median(opt_v)), [0.5, 4.0], 'bandpass', fs=FS_PPG)
    prom = 0.25 * float(np.std(x[w])) if w.any() else 0.25 * float(np.std(x))
    pk, props = signal.find_peaks(x, distance=int(0.33 * FS_PPG), prominence=max(prom, 0.5))
    in_keep = (t_opt[pk] >= a) & (t_opt[pk] <= b)
    pk_keep = pk[in_keep]
    t_p = t_opt[pk_keep]

    # PPG pulse rate from clean PPG inter-beat intervals
    ppg_hr = None
    if len(t_p) >= 5:
        ppg_rr = np.diff(t_p)
        ppg_rr = ppg_rr[(ppg_rr > 0.33) & (ppg_rr < 1.5)]
        if len(ppg_rr) >= 3:
            ppg_hr = float(60.0 / np.mean(ppg_rr))

    # Match each ECG R-peak in stable range to the next PPG peak within [80 ms, 650 ms]
    delays_ms, pairs = [], []
    t_r_keep = t_rpeaks[(t_rpeaks >= a) & (t_rpeaks <= b)]
    for tr in t_r_keep:
        cand = t_p[(t_p - tr >= 0.08) & (t_p - tr <= 0.65)]
        if len(cand):
            d_ms = float((cand[0] - tr) * 1000.0)
            delays_ms.append(d_ms)
            pairs.append((float(tr), float(cand[0]), d_ms))

    pat_ms = float(np.median(delays_ms)) if len(delays_ms) >= 5 else None
    pat_sd = float(np.std(delays_ms, ddof=1)) if len(delays_ms) >= 5 else None
    amps = props['prominences'][in_keep]
    amp_cv = float(100.0 * np.std(amps, ddof=1) / np.mean(amps)) if len(amps) >= 5 and np.mean(amps) > 0 else None

    return dict(
        present=True, pp_mV=pp_mv, t_opt=t_opt, ppg_clean=x, ppg_peaks=pk_keep, pat_pairs=pairs,
        features=dict(PPG_HR=ppg_hr, PAT_ms=pat_ms, PAT_sd_ms=pat_sd, PPG_amp_cv=amp_cv),
    )


# ---------------- features ----------------
def features_from(rr, median_beat, t_rr=None):
    ms = 1000.0 * rr
    d = np.diff(ms)
    mean_rr = float(ms.mean())
    hr = 60000.0 / mean_rr
    hr_range = float(60.0 / rr.min() - 60.0 / rr.max())
    sdnn = float(ms.std(ddof=1))
    rmssd = float(np.sqrt(np.mean(d**2)))
    pnn50 = float(100.0 * np.mean(np.abs(d) > 50.0))

    # Normalized & non-linear autonomic HRV metrics
    cvsd = float(100.0 * rmssd / mean_rr)
    var_d = float(np.var(d, ddof=1)) if len(d) > 1 else 0.0
    sd1 = np.sqrt(0.5 * var_d)
    sd2 = np.sqrt(max(2.0 * (sdnn**2) - 0.5 * var_d, 1e-6))
    sd1_sd2 = float(sd1 / sd2)

    # Baevsky's Stress Index (SI = AMo(%) / (2 * Mo * MxDMn))
    bins = np.arange(0.30, 1.55, 0.05)
    hist, edges = np.histogram(rr, bins=bins)
    if hist.sum() > 0:
        idx_max = int(np.argmax(hist))
        amo = 100.0 * hist[idx_max] / len(rr)
        mo = 0.5 * (edges[idx_max] + edges[idx_max + 1])
    else:
        amo, mo = 50.0, mean_rr / 1000.0
    mxdmn = max(float(np.percentile(rr, 95) - np.percentile(rr, 5)), 0.02)
    si = amo / (2.0 * mo * mxdmn)
    log_si = float(np.log10(max(si, 1.0)))

    # Frequency-domain HRV (Lomb-Scargle periodogram on uneven RR timestamps)
    if t_rr is None or len(t_rr) != len(rr):
        t_rr = np.cumsum(rr)
    rr_centered = ms - mean_rr
    if np.std(rr_centered) > 1e-6 and len(t_rr) >= 8:
        f_lf = np.linspace(0.04, 0.15, 64)
        f_hf = np.linspace(0.15, 0.40, 128)
        p_lf = float(np.trapezoid(signal.lombscargle(t_rr, rr_centered, 2.0 * np.pi * f_lf, normalize=True), f_lf))
        p_hf = float(np.trapezoid(signal.lombscargle(t_rr, rr_centered, 2.0 * np.pi * f_hf, normalize=True), f_hf))
        lf_hf = float(np.clip(p_lf / max(p_hf, 1e-6), 0.05, 25.0))
        hf_nu = float(100.0 * p_hf / max(p_lf + p_hf, 1e-6))
    else:
        lf_hf, hf_nu = 1.0, 50.0

    # Waveform morphology features
    i0 = int(PRE * FS_ECG)
    base = float(np.median(median_beat[:int(.08 * FS_ECG)]))
    R = float(median_beat[i0] - base)
    tseg = median_beat[i0 + int(.15 * FS_ECG): i0 + int(.38 * FS_ECG)] - base
    ti = int(np.argmax(np.abs(tseg)))
    rt_ms = float((int(.15 * FS_ECG) + ti) / FS_ECG * 1000.0)
    rtc_ms = float(rt_ms / np.sqrt(mean_rr / 1000.0))

    return dict(
        HR=hr, HR_range=hr_range, SDNN=sdnn, RMSSD=rmssd, pNN50=pnn50,
        CVSD=cvsd, SD1_SD2=sd1_sd2, log_SI=log_si, LF_HF=lf_hf, HF_nu=hf_nu,
        R_mV=R, T_over_R=float(tseg[ti] / R), RT_ms=rt_ms, RTc_ms=rtc_ms,
    )


def analyze(ecg_v, opt_v=None, segment=None):
    """Full preprocessing + ECG & PPG features. segment=(start_s, end_s) restricts to part of stable range."""
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

    opt_info = analyze_optical(opt_v, t[rp], (a, b))
    opt_pp = opt_info['pp_mV']
    q = dict(
        duration_s=len(ecg_v)/FS_ECG, stable_start=keep[0], stable_end=keep[1], stable_s=keep[1]-keep[0],
        r_peaks=int(len(rp)), clean_beats=int(sel.sum()) + 1, masked_pct=100*mask.mean(),
        saturated_pct=100*np.mean(np.abs(ecg_v) > 1.19), optical_present=opt_info['present'],
        optical_pp_mV=opt_pp, ppg_peaks=int(len(opt_info['ppg_peaks'])),
        pat_ms=opt_info['features']['PAT_ms'], pat_sd_ms=opt_info['features']['PAT_sd_ms'],
        ppg_hr=opt_info['features']['PPG_HR'], ppg_amp_cv=opt_info['features']['PPG_amp_cv'],
    )
    ok_enough = sel.sum() >= 15 and len(beats) >= 10
    feats = features_from(rr[sel], np.median(beats, 0), t_rr[sel]) if ok_enough else None
    if feats is not None:
        feats.update({k: v for k, v in opt_info['features'].items() if v is not None})
    return dict(
        t=t, ecg_clean=ecg_c, mask=mask, rpeaks=rp, t_rr=t_rr[sel], rr=rr[sel], keep=keep, unstable=unstable,
        median_beat=np.median(beats, 0) if len(beats) else None, features=feats, quality=q, usable=ok_enough,
        optical=opt_info,
    )


def ensure_full_features(feats, uid=None):
    """Ensure a baseline feature dict (e.g. from an older 8-feature baseline.json) has all 14 FEATURES."""
    if all(k in feats for k in FEATURES):
        return feats
    if uid:
        csv_path = ROOT / 'user_data' / uid / 'baseline.csv'
        if csv_path.exists():
            ecg, opt = load_recording(csv_path)
            res = analyze(ecg, opt)
            if res['usable']:
                return res['features']
    # Fallback analytic computation from the 8 core features
    out = dict(feats)
    hr = out.get('HR', 75.0)
    mean_rr = 60000.0 / max(hr, 40.0)
    rmssd = out.get('RMSSD', 25.0)
    sdnn = max(out.get('SDNN', 30.0), 1.0)
    out.setdefault('CVSD', 100.0 * rmssd / mean_rr)
    sd1 = rmssd / np.sqrt(2.0)
    sd2 = np.sqrt(max(2.0 * (sdnn**2) - sd1**2, 1.0))
    out.setdefault('SD1_SD2', float(sd1 / sd2))
    out.setdefault('log_SI', float(np.clip(3.2 - 0.025 * rmssd, 2.0, 3.4)))
    out.setdefault('LF_HF', float(np.clip(80.0 / max(rmssd, 4.0) - 1.0, 0.5, 15.0)))
    out.setdefault('HF_nu', float(100.0 / (1.0 + out['LF_HF'])))
    out.setdefault('RTc_ms', float(out.get('RT_ms', 180.0) / np.sqrt(mean_rr / 1000.0)))
    return out


def quality_warnings(res, role):
    q, f, w = res['quality'], res['features'], []
    if not res['usable']:
        w.append(('error', 'Not enough clean signal: need ≥ 15 clean beats in the stable part. Check electrode contact and re-record.'))
        return w
    if q['stable_s'] < 60:
        w.append(('warn', f"Only {q['stable_s']:.0f} s of stable signal. Record 3–5 min for the most stable frequency-domain HRV (LF/HF)."))
    if not q['optical_present']:
        w.append(('warn', 'Optical (PPG) channel is empty: sensor not connected or LED off. Only ECG is used.'))
    if q['masked_pct'] > 10:
        w.append(('warn', f"{q['masked_pct']:.0f} % of the ECG was masked as motion/saturation. Sit still during recording."))
    if role == 'baseline' and (f['HR'] > 95 or f['RMSSD'] < 15):
        w.append(('warn', f"Baseline shows elevated heart rate or low HRV (HR {f['HR']:.0f} bpm, RMSSD {f['RMSSD']:.1f} ms). "
                          "A truly rested baseline usually has HR < 85–90 bpm and RMSSD > 20 ms. Sit quietly for 5 min and re-record if possible."))
    return w


def beat_similarity(a, b):
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    z = lambda x: (x - x.mean()) / (x.std() or 1.0)
    return float(0.5*np.corrcoef(z(a), z(b))[0, 1] + 0.5*np.corrcoef(np.diff(a), np.diff(b))[0, 1])


# ---------------- model ----------------
class DeltaModel:
    def __init__(self, path=MODEL_PATH):
        m = json.loads(Path(path).read_text())
        self.meta = m
        self.features = m['features']
        self.mean, self.scale = np.array(m['mean']), np.array(m['scale'])
        self.coef, self.intercept, self.threshold = np.array(m['coef']), m['intercept'], m['threshold']
        self.watch_threshold = m.get('watch_threshold', 0.5)
        self.single_meta = m.get('single_model')

    def level(self, p, mode='delta'):
        """'fatigue' = serious (alarm), 'watch' = mild change from baseline, 'normal'."""
        thr = self.single_meta['threshold'] if (mode == 'single' and self.single_meta) else self.threshold
        w_thr = self.single_meta.get('watch_threshold', 0.5) if (mode == 'single' and self.single_meta) else self.watch_threshold
        return 'fatigue' if p >= thr else 'watch' if p >= w_thr else 'normal'

    def predict(self, new_feats, base_feats, beat_match=1.0):
        """Personal-baseline delta prediction. Suppresses R_mV if beat_match < 0.98 (electrode shift protection)."""
        new_full = ensure_full_features(new_feats)
        base_full = ensure_full_features(base_feats)
        delta = np.array([new_full[k] - base_full[k] for k in self.features], float)
        z = (delta - self.mean) / self.scale
        eff_coef = self.coef.copy()
        if beat_match is not None and beat_match < 0.98 and 'R_mV' in self.features:
            eff_coef[self.features.index('R_mV')] = 0.0
        p = float(1.0 / (1.0 + np.exp(-(z @ eff_coef + self.intercept))))
        contrib = pd.DataFrame({
            'feature': self.features,
            'baseline': [base_full[k] for k in self.features],
            'new': [new_full[k] for k in self.features],
            'change': delta,
            'push_to_fatigue': z * eff_coef,
        })
        contrib['pct_change'] = 100.0 * contrib.change / contrib.baseline.abs().replace(0, np.nan)
        return p, contrib.sort_values('push_to_fatigue', key=abs, ascending=False)

    def predict_single(self, new_feats):
        """Zero-baseline single-recording classification for a brand-new person without a saved baseline."""
        new_full = ensure_full_features(new_feats)
        sm = self.single_meta
        if not sm:
            return self.predict(new_full, HEALTHY_NORMS, beat_match=0.0)
        s_feats = sm['features']
        x = np.array([new_full[k] for k in s_feats], float)
        mean, scale = np.array(sm['mean'], float), np.array(sm['scale'], float)
        coef, intercept = np.array(sm['coef'], float), float(sm['intercept'])
        z = (x - mean) / scale
        p = float(1.0 / (1.0 + np.exp(-(z @ coef + intercept))))
        ref = [sm.get('rested_reference', HEALTHY_NORMS).get(k, mean[i]) for i, k in enumerate(s_feats)]
        contrib = pd.DataFrame({
            'feature': s_feats,
            'baseline': ref,
            'new': x,
            'change': x - np.array(ref, float),
            'push_to_fatigue': z * coef,
        })
        contrib['pct_change'] = 100.0 * contrib.change / contrib.baseline.abs().replace(0, np.nan)
        return p, contrib.sort_values('push_to_fatigue', key=abs, ascending=False)
