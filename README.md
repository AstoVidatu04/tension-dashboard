# 🇺🇸🇮🇷 USA–Iran Tension Dashboard

A Streamlit dashboard that tracks **publicly observable** tension signals between the United States and Iran using:

- **Structured news metadata** from **GDELT** (tone + themes)
- **Aggregate air-traffic anomaly** over Iran using **OpenSky** (count-only)
- Optional **stress amplifiers** (latest-score only): market stress and shipping disruption volume

⚠️ This project is an **indicator dashboard**, not a prediction system, and not financial or security advice.

---

## What you’ll see

- **Base risk score (0–100)** — structured news signal (meter)
- **Adjusted score (0–100)** — base score + optional amplifiers (meter)
- **Latest-day drivers table** explaining what moved the score (with tooltips)
- **Airborne aircraft over Iran** (aggregate counts only) + anomaly z-score
- Latest matching articles (deduplicated syndication)

---

## How the scoring works

### 1) Structured news signal (Base score)

The base score comes from GDELT’s **DOC 2.1** article stream:

1. Fetch articles matching your query and time window  
2. **De-duplicate syndication** (same story replicated across outlets)  
3. Build daily features:
   - **tension_core (tone)**: how negative the average coverage is  
   - **diplomacy_share**: fraction of coverage referencing negotiation/peace themes  
   - **articles**: deduplicated article count (used for confidence/uncertainty)  
4. Compute a daily time series:
   - smooth tension with a rolling window  
   - compute a **z-score** vs baseline  
5. Convert to **0–100** using a logistic function  

Tone + structured metadata is harder to game than headline keyword counting.

---

### 2) Quality controls

**Source diversity weighting**  
Signals carried by many independent outlets are weighted higher than those syndicated by only a few domains.

**Uncertainty band**  
Low volume or low diversity widens uncertainty. High agreement narrows it.

---

### 3) Air traffic anomaly (OpenSky + SQLite)

The app queries OpenSky for **aggregate airborne aircraft counts** over an Iran bounding box.

Stored data (SQLite):
- UTC timestamp  
- Aggregate airborne count only  

**Baseline logic**
- Z-score vs historical baseline for the same **hour-of-day** and **day-of-week**
- Baseline persists across restarts

A sharp traffic drop can optionally **increase risk** in the adjusted score.

Privacy note: no callsigns, no routes, no per-aircraft tracking.

---

### 4) Adjusted score (optional amplifiers)

The adjusted score starts with the base score, then optionally applies:

- **Market stress amplifier** (oil / volatility proxy)
- **Shipping disruption amplifier** (aggregated GDELT volume)
- **Air traffic anomaly amplifier**

These are **amplifiers**, not core drivers.

---

## UI notes (tooltips)

The **Latest-day drivers** table includes tooltips explaining:

- **value** — raw daily value  
- **z** — how unusual today is vs baseline  
- **weight** — how strongly the component is applied  
- **effect** — contribution before conversion to 0–100  

---

## Running locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
````

---

## Configuration

### OpenSky credentials

Provide via Streamlit secrets or environment variables:

* `OPENSKY_CLIENT_ID`
* `OPENSKY_CLIENT_SECRET`

---

## Project structure

```text
.
├── streamlit_app.py
├── gdelt_structured.py
├── market_stress.py
├── flights.db              # auto-created (SQLite flight baseline)
├── requirements.txt
└── README.md
```

---

## Flight DB maintenance

The sidebar includes a **Prune flight DB** control to limit storage size
(e.g., keep the last 120 days).

---

## Limitations

* Media coverage can be biased or cyclical
* GDELT tone/themes are imperfect
* OpenSky coverage may be incomplete or rate-limited
* No classified or private data is used

---

## Future improvements

* GDELT **Events / CAMEO / Goldstein** conflict coding
* Calibration & backtesting so score levels map to outcomes
* More robust regime-change detection
* Additional hard-to-spin aggregated indicators
