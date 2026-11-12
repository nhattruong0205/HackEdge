"""Report serious-fatigue checks to HR.

Every check that the model flags as FATIGUE (P >= the serious threshold) creates one HR report:
  who, when it was checked, how extreme (severity + score), heart rate / RMSSD change vs. baseline,
  the main reasons, and whether the signal quality is trustworthy (beat-shape match).
Raw ECG is never sent, only the summary.

Delivery (each one optional, configured with environment variables):
  always        -> appended to hr_reports.csv (the HR page in the app reads it)
  email         -> HR_EMAIL_TO, SMTP_HOST, SMTP_PORT (587), SMTP_USER, SMTP_PASSWORD, SMTP_FROM (= SMTP_USER)
  chat webhook  -> HR_WEBHOOK_URL (Slack / Teams / Discord style incoming webhook, posts {"text": ...})
"""
import json
import os
import smtplib
import urllib.request
from datetime import datetime
from email.message import EmailMessage

import pandas as pd

from fatigue_pipeline import LABELS, ROOT

REPORTS = ROOT / 'hr_reports.csv'
SEVERITY_BANDS = [(0.9, 'SEVERE'), (0.8, 'HIGH'), (0.0, 'MODERATE')]   # by P(fatigue), only used once flagged


def severity(p, threshold):
    """Severity label plus a 0–100 score: 0 = just at the alarm threshold, 100 = certain fatigue."""
    label = next(name for lo, name in SEVERITY_BANDS if p >= lo)
    score = 100 * (p - threshold) / (1 - threshold) if threshold < 1 else 100
    return label, round(max(0.0, min(100.0, score)))


def build_report(uid, recording, p, threshold, new_feats, base_feats, contrib, beat_match, when=None):
    label, score = severity(p, threshold)
    top = contrib[contrib.push_to_fatigue > 0].head(3)
    reasons = [f"{LABELS[r.feature][0]} {r.baseline:.2f} → {r.new:.2f}"
               + (f" ({r.pct_change:+.0f} %)" if pd.notna(r.pct_change) else '') for r in top.itertuples()]
    return dict(
        time=(when or datetime.now()).isoformat(timespec='seconds'), person=uid, recording=recording,
        severity=label, severity_score=score, p_fatigue=round(p, 3), alarm_threshold=round(threshold, 3),
        hr_now=round(new_feats['HR'], 1), hr_baseline=round(base_feats['HR'], 1),
        rmssd_now=round(new_feats['RMSSD'], 1), rmssd_baseline=round(base_feats['RMSSD'], 1),
        beat_match=round(beat_match, 3), signal_reliable=beat_match >= 0.98, reasons='; '.join(reasons))


def report_text(r):
    lines = [f"Fatigue alert: {r['person']}: {r['severity']} (severity {r['severity_score']}/100)",
             f"Checked at: {r['time'].replace('T', ' ')}",
             f"P(fatigue): {r['p_fatigue']:.2f} (alarm threshold {r['alarm_threshold']:.2f})",
             f"Heart rate: {r['hr_now']} bpm (baseline {r['hr_baseline']})",
             f"RMSSD: {r['rmssd_now']} ms (baseline {r['rmssd_baseline']})",
             f"Main changes: {r['reasons'] if isinstance(r['reasons'], str) and r['reasons'] else '–'}"]
    if not r['signal_reliable']:
        lines.append(f"Note: heartbeat shape differs from baseline (match {r['beat_match']}); electrodes may have "
                     "been placed differently, so please confirm with the employee before acting.")
    lines.append('Suggested action: offer a break / breathing exercise and re-check later.')
    return '\n'.join(lines)


def _send_email(r):
    to, host = os.getenv('HR_EMAIL_TO'), os.getenv('SMTP_HOST')
    if not (to and host):
        return None
    msg = EmailMessage()
    msg['Subject'] = f"[BreathBreak] {r['severity']} fatigue: {r['person']} at {r['time'][11:16]}"
    msg['From'] = os.getenv('SMTP_FROM') or os.getenv('SMTP_USER') or 'breathbreak@localhost'
    msg['To'] = to
    msg.set_content(report_text(r))
    with smtplib.SMTP(host, int(os.getenv('SMTP_PORT', 587)), timeout=15) as s:
        s.starttls()
        if os.getenv('SMTP_USER'):
            s.login(os.getenv('SMTP_USER'), os.getenv('SMTP_PASSWORD', ''))
        s.send_message(msg)
    return f'email to {to}'


def _send_webhook(r):
    url = os.getenv('HR_WEBHOOK_URL')
    if not url:
        return None
    req = urllib.request.Request(url, data=json.dumps({'text': report_text(r)}).encode(),
                                 headers={'Content-Type': 'application/json'})
    urllib.request.urlopen(req, timeout=15).close()
    return 'webhook'


def send_report(r):
    """Log the report, then try every configured channel. Returns the report with a 'delivery' field."""
    sent, failed = [], []
    for fn in (_send_email, _send_webhook):
        try:
            ch = fn(r)
            if ch:
                sent.append(ch)
        except Exception as e:                       # never let a delivery problem break the check page
            failed.append(f'{fn.__name__[6:]} failed: {e}')
    r = dict(r, delivery='; '.join(sent + failed) or 'logged only (no email/webhook configured)')
    pd.DataFrame([r]).to_csv(REPORTS, mode='a', header=not REPORTS.exists(), index=False)
    return r


def load_reports():
    return pd.read_csv(REPORTS) if REPORTS.exists() else pd.DataFrame()
