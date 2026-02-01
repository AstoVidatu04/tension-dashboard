---

---
title: USA–Iran Tension Dashboard
emoji: 🌍
colorFrom: blue
colorTo: red
sdk: streamlit
app_file: streamlit_app.py
pinned: false
---

# 🇺🇸🇮🇷 USA–Iran Tension Dashboard

A transparent indicator built from public news signals (GDELT). Not a literal probability-of-war predictor.


```md
# 🇺🇸🇮🇷 USA–Iran Tension Dashboard

A lightweight, interactive dashboard that tracks **publicly observable tension signals** between the United States and Iran using global news data from the **GDELT Project**.

⚠️ **Important:**  
This is **not** a prediction model and **not** a probability-of-war calculator.  
It is an **indicator** meant to show trends, volume, and tone of publicly reported events.

---

## 🌍 What this shows

The dashboard aggregates recent news articles and derives:

- 📈 A **Tension / Risk Score (0–100)**  
- 📊 Daily counts of:
  - Hostile signals
  - Military-related signals
  - Diplomatic signals
- 🧠 A breakdown of **what drove the latest score**
- 📰 Links to the most recent matching articles

All inputs are transparent and adjustable.

---

## 🧠 How the score works (high level)

1. News articles mentioning both the USA and Iran are pulled from **GDELT**
2. Headlines are classified using simple keyword heuristics into:
   - **Hostile**
   - **Military**
   - **Diplomatic**
3. Signals are:
   - aggregated daily
   - lightly smoothed
   - normalized (z-scores)
4. A weighted score is calculated and squashed into a **0–100 range**

Higher score = **more public escalation signals**, not inevitability.

---

## 🧩 What this is *not*

- ❌ Not a classified or intelligence-grade system
- ❌ Not a geopolitical prediction engine
- ❌ Not financial, political, or security advice

Media coverage ≠ intent, and silence ≠ de-escalation.

---

## 🚀 Live demo

If deployed on Streamlit Cloud, a public link will be available here:

```

https://<your-app-name>.streamlit.app

````

(Open in a browser — no account required.)

---

## 🛠 Tech stack

- **Python**
- **Streamlit**
- **Pandas / NumPy**
- **Plotly**
- **GDELT DOC 2.1 API**

---

## ▶️ Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
````

---

## 📦 Files

```text
.
├── streamlit_app.py   # Main Streamlit app
├── requirements.txt   # Python dependencies
└── README.md
```

---

## 🔒 Data & privacy

* Uses **publicly available news metadata**
* No user tracking
* No cookies
* No authentication

---

## 🧪 Known limitations

* Media-driven (subject to hype cycles)
* Keyword-based classification (v1 by design)
* GDELT coverage varies by region and language
* No access to classified or backchannel diplomacy

Future improvements could include event-code–based models or NLP classifiers.

---

## 📜 License

MIT — free to fork, modify, and share.
Just don’t claim it predicts wars 🙂

---
