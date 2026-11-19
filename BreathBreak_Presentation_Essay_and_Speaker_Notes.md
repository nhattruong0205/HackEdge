# BreathBreak: Physiological ECG & PPG Fatigue Classification
## Complete Presentation Essay & Speaker Guide (Slides 1–10)

---

### Introduction & The Core Problem (Slide 1)
When nurses, truck drivers, or factory workers work long shifts, physical and mental exhaustion builds up quietly long before they realize how tired they are. That hidden fatigue slows down reaction time, harms health, and causes preventable workplace accidents. **BreathBreak** is an edge-powered healthcare wearable built on the **Texas Instruments AFE4950** analog front-end and the **TI SK-AM62B** embedded processor. Its goal is simple: read a worker's electrical heartbeat (**ECG**) and optical blood-pulse wave (**PPG**) during a short 1-to-3-minute check, determine whether their body is experiencing real physiological fatigue, prompt a 2-minute guided breathing break when needed, and log an anonymized summary for HR workload planning.

However, detecting fatigue from ECG and PPG in the real world has one huge biological challenge: **every human body has a different normal**. One healthy, well-rested person might naturally have a resting heart rate of 65 beats per minute (bpm) and high heart-rate variability (HRV), while another healthy person naturally sits at 85 bpm with lower variability. If a machine learning model simply memorizes raw numbers, it will confuse "who the person is" or "how the electrodes were placed" with fatigue. Our presentation explains how we solved that problem using a **Dual-Mode Physiological Machine Learning Pipeline**.

---

### 1. What the Model Does: End-to-End Clinical Architecture (Slide 2)
Our system processes every recording through four connected stages:
1. **Synchronized Sensor Capture (TI AFE4950):** The board simultaneously records the electrical heartbeat (**ECG at 500 Hz**) and the optical blood-volume pulse wave (**PPG / TIA1-3 at 50 Hz**) at 24-bit resolution.
2. **8-Step Clinical Signal Preprocessing (`P1–P8`):** Raw sensor voltages are cleaned of power-line hum, breathing drift, and motion spikes, leaving only verified heartbeats and arterial pulse peaks.
3. **Dual `scikit-learn` Machine Learning Classifiers (`StandardScaler` + L2-Regularized `LogisticRegression`):**
   - **Mode A — Personal-Baseline Delta Classifier (Primary Mode for Returning Workers):**
     When a worker uses BreathBreak for the first time, they record a rested baseline once. On every future check, the model does not look at their raw numbers in isolation—instead, it computes the **within-person change** ($\Delta \mathbf{x} = \mathbf{x}_{\text{new}} - \mathbf{x}_{\text{baseline}}$). By subtracting the person's own rested baseline, individual anatomy differences cancel out, and the model focuses purely on how much their nervous system has shifted toward fatigue today.
   - **Mode B — Zero-Baseline Single-Recording Classifier (Cold-Start Mode for Brand-New People):**
     What if a brand-new person sits down for a check and has never recorded a baseline before? Mode B classifies a single recording on the spot. To avoid being tricked by person-to-person voltage differences, it drops raw electrode voltage (`R_mV`) completely and uses **10 self-normalized autonomic ratios**—metrics that compare each person's heartbeat variability against their own beat length within that single session.
4. **3-Tier Clinical Decision & HR Alert:** The classifier outputs a calibrated fatigue probability $P(\text{fatigue})$ from `0.00` to `1.00`, grouping results into **NORMAL** ($P < 0.50$), **WATCH** ($0.50 \le P < \text{Threshold}$, mild shift, no alarm), or **FATIGUE** ($P \ge \text{Threshold}$, triggering a breathing break and a 0–100 severity alert).

---

### 2. Signal Preprocessing: Turning Noisy Sensor Data into Clean Physiology (Slide 3)
Raw ECG and PPG signals from a wearable sensor cannot be fed straight into a classifier. Slide 3 shows our 8-stage clinical pipeline (`P1–P8`):
- **Removing Drift and Electrical Hum (`P1–P4`):** We center the signal and convert volts to millivolts (`P1`), apply a **0.5 Hz high-pass filter** (`P2`) to remove slow baseline drift caused by chest movement during breathing, apply a **narrow 60 Hz IIR notch filter** (`P3`) that reduces power-line interference from **9.48 times** the ECG power down to **0.003 times** (removing the thick fuzzy band on the trace), and apply a **40 Hz low-pass filter** (`P4`) to smooth out muscle tremor without dulling the sharp R-peaks.
- **Masking Motion Spikes (`P5`) & Detecting True Heartbeats (`P6`):** If a user bumps the cable or moves their arm, a huge voltage spike appears. Filtering cannot fix a motion spike, so Step `P5` automatically masks out any 1-second block where the voltage jumps above twice the typical R-peak height. Step `P6` then runs the **Pan–Tompkins QRS detector** (band-pass filtering, derivative, squaring, and 120 ms moving-window integration) plus an amplitude validation gate (`0.5× to 1.6×` median height) so leftover motion spikes are rejected instead of being counted as fake heartbeats.
- **Cleaning Beat Intervals & Trimming Shaky Edges (`P7 & P7b`):** We measure the time gap (`RR interval`) between consecutive beats, reject physiologically impossible gaps or sudden $>20\%$ jumps caused by masked motion, and trim off the unstable first and last few seconds of the recording (when the user's hand was reaching for the *Start/Stop Capture* button).
- **Optical PPG Pulse Detection (`P8`):** When the optical sensor is active, we filter the 50 Hz PPG wave between 0.5 and 4.0 Hz, detect each blood-pulse peak, and pair every ECG electrical spike with the arrival of the blood pulse in the skin.

---

### 3. The 18 Physiological Features: How Fatigue Appears in ECG and PPG (Slide 4)
To understand our features in plain English, think of the human heart as being controlled by two nervous-system pedals:
- The **"Brake Pedal" (Parasympathetic / Vagal nerve)** is active when you are rested. It keeps your heart rate lower and makes the time gap between heartbeats smoothly speed up when you inhale and slow down when you exhale.
- The **"Gas Pedal" (Sympathetic stress response)** takes over when you are fatigued or sleep-deprived because your body has to work harder just to keep you awake.

When a person becomes fatigued, they take their foot off the brake and press the gas. Slide 4 groups our **18 extracted features** into four biological categories that capture this shift:
1. **Time-Domain HRV (5 ECG features — `HR`, `HR_range`, `SDNN`, `RMSSD`, `pNN50`):**
   In fatigue, average heart rate (`HR`) creeps **up** (+5 to +20 bpm), while beat-to-beat flexibility (`RMSSD`, `SDNN`, `pNN50`) drops **down** sharply because the vagal "brake pedal" turns off and the heart starts beating like a stiff, rigid metronome.
2. **Normalized & Non-Linear Autonomic Ratios (3 ECG features — `CVSD`, `Poincaré SD1/SD2`, `Baevsky Stress Index`):**
   - `CVSD` (`100 × RMSSD / mean_RR`) normalizes beat-to-beat variability by heart rate so it is fair across people with different resting heart rates.
   - `Poincaré SD1/SD2` measures the ratio of short-term beat-to-beat recovery (`SD1`) to long-term variability (`SD2`). In rested people it sits around `0.35–0.60`; in fatigue it drops below `0.30`.
   - `Baevsky's Stress Index (log10 SI)` measures how tightly locked the beat intervals are. When fatigued, `log10 SI` climbs above `2.65–3.30`.
3. **Frequency-Domain HRV & Waveform Morphology (6 ECG features — `LF/HF`, `HF_nu`, `RT_ms`, `RTc_ms`, `T_over_R`, `R_mV`):**
   Using a Lomb–Scargle spectrum on the beat intervals, we measure low-frequency sympathetic/baroreflex waves (`LF`, 0.04–0.15 Hz) versus high-frequency breathing/vagal waves (`HF`, 0.15–0.40 Hz). In fatigue, the **`LF/HF` ratio rises** and normalized vagal power (**`HF_nu`**) drops. We also measure the heart muscle's recharge wave (`T_over_R` and heart-rate-corrected R-to-T time `RTc_ms`).
4. **Optical PPG & Vascular Coupling (4 PPG features — `PPG_HR`, `PAT_ms`, `PAT_sd_ms`, `PPG_amp_cv`):**
   When you are fatigued, sympathetic tension tightens your arteries. Like a rigid pipe compared to a soft hose, tighter arteries carry the blood-pressure wave faster from the heart to the skin. Therefore, **Pulse Arrival Time (`PAT_ms`)**—the delay in milliseconds from the ECG R-peak to the PPG pulse peak—gets **shorter** during fatigue, and the PPG wave amplitude constricts.

---

### 4. Sample Data & The 4 Engineering Issues We Solved (Slides 5 & 6)
Slide 5 shows the complete physiological table across all 13 recordings in our project, and Slide 6 explains the **four major engineering problems** we uncovered while analyzing them—and how solving them transformed our classifier from 50–65% accuracy to 100% honest accuracy:

1. **Issue 1 — Un-Rested Baselines & `R_mV` Electrode Overfitting:**
   When we inspected the initial 10 recordings in `dataset_exploration.ipynb`, we found why every classifier had struggled: while Person B (`normal/first`, `second`, and the new `sixth_attempt.csv`) had true rested recordings (**HR 65–70 bpm, RMSSD 32–39 ms, working PPG**), two of Person A's supposed "normal" files (`normal/third` and `normal/fifth`) were recorded when Person A had a heart rate of **105–108 bpm and an RMSSD of only 6.7–8.1 ms**—physiologically identical to their fatigue recordings! Because HR and RMSSD did not change between Person A's un-rested "normal" files and their fatigue files, the original model was forced to rely on **`R_mV` (R-peak voltage, weight `-1.35`)**. Worse, `R_mV` changes whenever electrodes are placed slightly differently on the skin.
   - *How we solved it:* First, we added the normalized autonomic ratios (`CVSD`, `SD1/SD2`, `LF/HF`, `HF_nu`, `log10 SI`), which revealed that `normal/fourth_attempt.csv` actually had healthy vagal recovery (`RMSSD 20.1 ms`, `SD1/SD2 = 0.55`, `HF_nu = 39.5%`) compared to all five fatigue files (`SD1/SD2 = 0.18–0.25`). Second, we added a **Baseline Quality Guard** (`is_truly_rested_normal`) that flags un-rested baselines (`HR > 95 bpm` and `RMSSD < 15 ms`) so they don't corrupt training. Third, we added an **Electrode-Shift Gate**: the pipeline compares the average heartbeat shape (`beat_similarity`) against baseline, and if `beat_similarity < 0.98`, it automatically sets `R_mV`'s weight to `0.0` so shifted electrodes never trigger a false alarm.
2. **Issue 2 — Window-Level Data Leakage (The "89% Fake Accuracy" Trap):**
   In `dataset_exploration.ipynb`, cutting 30-second recordings into overlapping 20-second windows and shuffling them across 5-fold cross-validation gave 89% accuracy with Random Forest. We proved that this was a data-leakage illusion: 54 of the 64 windows came from normal files (so guessing "normal" every time already got 84%), and the tree models were simply memorizing *which recording* a window came from.
   - *How we solved it:* We enforced strict **Leave-One-Recording-Out (LORO)** evaluation, where an entire recording is held out from training and tested as a brand-new upload.
3. **Issue 3 — Motion Spikes Artificially Inflating HRV:**
   In `fatigue/fourth_attempt.csv`, a single motion artifact at $t = 19\text{ s}$ created fake beat intervals that inflated `RMSSD` from **7.8 ms up to 37 ms**, which would trick any classifier into thinking a fatigued worker was well-rested. Our `P5–P7` motion mask and $\pm 20\%$ rolling-median RR filter eliminated those false intervals completely.
4. **Issue 4 — TI GUI Trailing-Comma Column Shift & Empty PPG Channels:**
   Every data row exported by the TI AFE4950 GUI ends with a trailing comma that shifts all pandas column headers by one unless `index_col=False` is used. In addition, 8 of the initial 10 files had an empty optical channel (0.5 mV ADC noise). Our pipeline automatically checks whether optical peak-to-peak amplitude exceeds 5 mV, extracting full PPG & `PAT_ms` metrics when active (`first`, `second`, `sixth`, and `Person_C`) and seamlessly using ECG-only features when the optical sensor is off.

---

### 5. Machine Learning Weights & Honest Evaluation Results (Slides 7 & 8)
Slide 7 shows the learned weights of our two `scikit-learn` L2-regularized Logistic Regression models (`StandardScaler` + `LogisticRegression`):
- **In Model 1 (Personal-Baseline Delta Classifier, 14 features):** The top weights are now **`CVSD` (`-0.61`)**, **`LF/HF` (`+0.50`)**, **`RMSSD` (`-0.50`)**, and **`Poincaré SD1/SD2` (`-0.48`)**, while `R_mV` dropped from `-1.35` down to `-0.33` (and is muted automatically if electrodes move). Every single weight aligns with human autonomic biology: a drop in vagal HRV (`CVSD`, `RMSSD`, `SD1/SD2`, `HF_nu`) and a rise in sympathetic stress (`LF/HF`, `HR`, `log10 SI`) push the probability toward fatigue.
- **In Model 2 (Zero-Baseline Single-Recording Classifier, 10 features):** Without any baseline, the model classifies a new person purely from self-normalized ratios led by **`Poincaré SD1/SD2` (`-0.47`)**, **`CVSD` (`-0.43`)**, **`RMSSD` (`-0.35`)**, **`HF_nu` (`-0.35`)**, and **`LF/HF` (`+0.31`)**.

Slide 8 presents our honest **Leave-One-Recording-Out (LORO)** evaluation results:
- **Personal-Baseline Delta LORO:** **11 / 11 comparisons correct (100%)**—catching **5 / 5** fatigue recordings and leaving **6 / 6** normal checks alone (**0% false alarm rate**, up from 65% in the old 8-feature model).
- **Zero-Baseline Single-Recording LORO:** **9 / 9 clean recordings correct (100%)** across all 4 valid rested baselines (`first`, `second`, `fourth`, `sixth`) and all 5 fatigue recordings (`first` through `fifth`).

---

### 6. Live Prediction Walkthrough on an Unseen New Person (`Person_C`) & Clinical Workflow (Slides 9 & 10)
To prove the classifier is not hard-coded and generalizes to a brand-new person, Slide 9 walks through **`Person_C`**—a synthetic subject created in the exact 10-column TI AFE4950 CSV format who was **never included in the training manifest**:
- **When `Person_C` is rested (`normal/personC_rested_baseline.csv`):** Their heart rate is **70.9 bpm** (12 beats in 10 seconds), vagal HRV is healthy (`RMSSD = 25.8 ms`, `CVSD = 3.05%`, `SD1/SD2 = 0.51`, `LF/HF = 1.17`), and their ECG-to-PPG Pulse Arrival Time (`PAT`) is **490 ms**. The Zero-Baseline model classifies this single recording as **NORMAL (`P(fatigue) = 0.12`)**.
- **When `Person_C` experiences mild fatigue (`fatigue/personC_mild_fatigue.csv`):** Their heart rate rises moderately to **85.9 bpm** (15 beats in 10 seconds), `RMSSD` drops to **11.4 ms** (`CVSD = 1.63%`, `SD1/SD2 = 0.38`), `LF/HF` doubles to **2.64**, and arterial stiffening shortens their ECG-to-PPG `PAT` from **490 ms down to 443 ms** (-47 ms faster), while their R-peak height barely changes (`beat_similarity = 0.998`, confirming identical electrode placement).
- **Both ML modes catch `Person_C`'s mild fatigue and place it in the `MODERATE` (mildest alarm) severity tier:**
  - **Personal-Baseline Delta Mode:** `P(fatigue) = 0.78` (alarm threshold $\ge 0.60$) → **FATIGUE (`MODERATE` severity, score `46 / 100`)**.
  - **Zero-Baseline Single-Recording Mode:** `P(fatigue) = 0.61` (alarm threshold $\ge 0.58$) → **FATIGUE (`MODERATE` severity, score `7 / 100`)**.
- As shown in the overlaid 10-second ECG, 10-second PPG, and beat-to-beat HR charts on Slide 9, anyone looking at the dashboard can visually see the faster, tighter heartbeat spacing, the flattened beat-to-beat HR line, and the shifted PPG pulse wave alongside the table of mathematical feature contributions (`push_to_fatigue`).

Finally, Slide 10 shows how BreathBreak fits into a real healthcare or industrial workplace:
- **Actionable Recovery:** When a check crosses the fatigue threshold, the worker is immediately guided through a **2-minute slow-breathing break (6 breaths/minute)**, which reactivates the vagus nerve ("brake pedal") and restores `RMSSD`.
- **Privacy-Preserving HR Alerts:** Only serious fatigue checks generate an HR report (`hr_reports.csv`, email, or webhook), and **raw ECG/PPG waveforms are never sent**—only the timestamp, severity score (0–100), top 3 physiological shifts, and a flag indicating whether electrode placement was reliable.
- **Ready for Edge Deployment on the TI SK-AM62B:** Because our trained `scikit-learn` pipelines export their standardization means, scales, weights, and thresholds directly to a compact JSON file (`models/delta_model.json`), inference on the TI SK-AM62B edge processor requires only pure NumPy vector arithmetic and runs in less than a millisecond without cloud connectivity.
