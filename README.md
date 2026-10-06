# StockHolmes 🕵️‍♂️📈

**StockHolmes** is an AI-powered stock forecasting and market intelligence platform that combines time-series deep learning models (Kronos) with real-time news sentiment analysis (NLTK VADER) and direct Google Sheets sync automation.

---

## ✨ Features

- **Hybrid AI Forecasting:** Combines raw market data from `yfinance` with Kronos deep learning models for trend analysis.
- **Sentiment-Enhanced Predictions:** Scrapes live news feeds via Google News RSS and computes compound sentiment scores to detect upward momentum or downward drift.
- **Interactive UI Dashboard:** Built with Flask, HTML5, Plotly.js, and CSS dark theme.
- **Google Sheets Sync:** One-click integration to write stock forecasts and actual price updates directly to a tracking spreadsheet using Google Cloud Service Accounts.
- **Quantitative Signals:** Uses regulatory-aligned quantitative terms (`UPWARD MOMENTUM`, `DOWNWARD DRIFT`, `CONSOLIDATION`).

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.9+
- Git

### 2. Clone Repository & Install Dependencies
```bash
git clone https://github.com/jaygiradkar04/StockHolmes.git
cd StockHolmes

# Install required Python packages
pip install flask flask-cors pandas numpy yfinance plotly nltk google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

### 3. Setup Google Sheets Service Account (Optional)
1. Enable **Google Sheets API** in Google Cloud Console.
2. Create a Service Account and download the JSON key.
3. Rename the key file to `service_account.json` in the root project directory (see `service_account.json.example`).
4. Share your target Google Sheet with the service account `client_email` with **Editor** permissions.

### 4. Run the Application
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000`.

---

## 🛡️ License & Disclaimer

This project is for educational and research purposes only. Financial investments carry risk. Always perform independent due diligence before making investment decisions.
