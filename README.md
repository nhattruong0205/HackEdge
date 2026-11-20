# BreathBreak — edge fatigue detection from wearable ECG

UTD HackEdge 2026, Track 1 (Edge Digital Twin for Continuous Physiological Monitoring).
Nhat Truong · Dung Anh Nguyen · Anthony Tran

BreathBreak records ECG with a **TI AFE4950**, and on a **TI SK-AM62B** edge board it decides whether
the wearer is **normal** or **fatigued** by comparing heart rate, SDNN and RMSSD with that person's own
rested baseline (*personal-shift model*). A Streamlit dashboard on a laptop handles baseline enrollment,
checks, history and HR reports.

Result on our recordings (leave-one-recording-out): **11/11 correct** (6/6 fatigue, 5/5 rest, 0 false alarms).
Full details: [`report/breathbreak_report.pdf`](report/breathbreak_report.pdf).

---

## Repository layout

| Path | What it is |
|---|---|
| `board_shift.py` | **Runs on the board.** Trains the personal-shift model and checks new recordings (numpy only) |
| `board_classifier.py` | **Runs on the board.** ECG preprocessing in pure numpy (filters, Pan–Tompkins R-peaks, features) |
| `board_manifest.csv` | Which recording belongs to which person and condition (used by `board_shift.py`) |
| `send_repo_serial.py` | Sends the repo or any folder to the board over the USB serial console (no network needed) |
| `app.py` | Streamlit dashboard (laptop) |
| `fatigue_pipeline.py` | ECG preprocessing + features + model class used by the dashboard (SciPy version) |
| `train_delta_model.py` | Trains the dashboard's model from `data_manifest.csv` → `models/delta_model.json` |
| `hr_report.py` | HR fatigue reports (optional email / webhook delivery) |
| `data/normal/`, `data/fatigue/` | AFE4950 ECG recordings (TI GUI CSV exports, 500 Hz ECG) |
| `data_manifest.csv` | Person / condition labels for the dashboard's training |
| `transfer_model_b.py` | Transfer-learning experiment (pre-train on WESAD, fine-tune on DROZY, test on our data). The public datasets are **not** included (research-only licences); download them yourself and run `python3 transfer_model_b.py /path/to/WESAD /path/to/DROZY` |
| `dataset_exploration.ipynb` | Exploration notebook |
| `report/` | Project report (LaTeX + PDF); figures in `outputs/comparison/` |

Recordings labelled `unrested` in the manifests (subject A, normal 3–5: heart rate ~106 bpm) are kept
for reference but excluded from training.

---

## 1. Laptop setup (dashboard + training)

Requirements: Python 3.10+ (tested on 3.12).

```bash
git clone https://github.com/nhattruong0205/HackEdge.git
cd HackEdge
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Train the model and start the dashboard:

```bash
python3 train_delta_model.py          # prints the leave-one-recording-out evaluation, saves models/delta_model.json
streamlit run app.py                  # opens http://localhost:8501
```

Using the dashboard:
1. **① Record baseline** – create a person and upload a rested recording (or pick a sample, e.g. `normal/A_first_attempt.csv`).
2. **② Check for fatigue** – pick the person and upload a new recording → NORMAL / FATIGUE with the change in HR, SDNN, RMSSD.
3. **History** and **HR reports** show past checks. Optional delivery of HR reports: set `HR_EMAIL_TO`, `SMTP_HOST`,
   `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD` (email) or `HR_WEBHOOK_URL` (Slack/Teams webhook).

## 2. Run training and detection on the SK-AM62B

The board needs only **Python 3 + numpy** (preinstalled on the TI SDK image). No internet is required.

### 2a. Copy the code and data to the board

**Over the serial console** (USB cable to the board's UART port, 115200 baud). Log in as `root` first
(`python3 -m serial.tools.miniterm /dev/cu.usbserial-XXXX0 115200`), then **close miniterm** (Ctrl+]) and run on the laptop:

```bash
python3 send_repo_serial.py .          # sends the latest commit (code + data, no video/PDFs) to /root/hackedge
```

It is resumable (run it again if interrupted); do not open miniterm until it prints `Done`.
To send any other folder: `python3 send_repo_serial.py --dir LOCAL_FOLDER /root/target`.

**Or with a USB flash drive** in the board's USB-A port:

```bash
# on the board
mkdir -p /mnt/usb && mount /dev/sda1 /mnt/usb
mkdir -p /root/hackedge && cp -r /mnt/usb/HackEdge/. /root/hackedge/ && umount /mnt/usb
```

### 2b. Train on the board

```bash
cd /root/hackedge
python3 board_shift.py train
```

This preprocesses every recording in `board_manifest.csv`, trains the model, prints the honest
leave-one-recording-out test and saves `models/shift_model.json` (with each person's baseline).

### 2c. Check a new recording on the board

```bash
python3 board_shift.py check A path/to/new_recording.csv
# FATIGUE  P(fatigue) 0.99 | HR 67->107 (+60%)  SDNN 45->16 (-64%)  RMSSD 36->8 (-78%)
```

`P < 0.5` → NORMAL, `≥ 0.5` → FATIGUE.

A new person first records a rested baseline (sit quietly 5 min, then record 2–3 min):

```bash
python3 board_shift.py enroll D rested.csv
python3 board_shift.py check D later_recording.csv
python3 board_shift.py people                     # list enrolled people
```

## 3. Adding your own recordings

1. Export from the TI AFE4950 GUI as CSV and put the file in `data/normal/` or `data/fatigue/`.
2. Add a line to `board_manifest.csv` (and `data_manifest.csv` for the dashboard): `file, person, condition, ...`.
   Record the rested baseline at rest (HR normally < 90 bpm).
3. Re-run `python3 board_shift.py train` (board) or `python3 train_delta_model.py` (laptop).

## How it works (short)

1. **Preprocess** ECG: 0.5 Hz high-pass, 60 Hz notch, 40 Hz low-pass (zero-phase), artifact masking,
   Pan–Tompkins R-peaks, RR cleaning, trim unstable start/end.
2. **Features**: heart rate, SDNN, RMSSD.
3. **Shift** from the person's own rested baseline: `log(new / baseline)` for each feature.
4. **Logistic regression** trained on fatigue-vs-rest and rest-vs-rest pairs of the same person:
   HR up and SDNN / RMSSD down → fatigue.

## Limitations

Two subjects and 11 usable recordings; the model detects sympathetic activation relative to rest and
cannot yet separate fatigue from exercise or stress. See the report for details.
