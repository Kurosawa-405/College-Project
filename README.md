# 📊 AI Employability Forecast Dashboard

An advanced AI-powered dashboard for analyzing and forecasting employability trends across G20 countries using machine learning, time series forecasting, explainable AI, and local LLM integration.

---

## 🚀 Features

### 📈 Forecasting

* ARIMA-based time series forecasting
* Prophet-based forecasting with scenario tuning
* Multi-scenario comparison (Base, Optimistic, Pessimistic)

### 🧠 Explainable AI

* SHAP-based feature importance analysis
* Beeswarm and bar plots for interpretability

### 🤖 AI Analyst (Local LLM)

* Integrated with **Ollama**
* Ask questions about trends, countries, and policies
* Context-aware responses using real dataset

### 📊 Dashboard

* Clean Streamlit UI
* Country-wise rankings
* Historical trend visualization
* Downloadable dataset

---

## 🏗️ Tech Stack

* **Frontend:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib
* **Forecasting:** Prophet, ARIMA (statsmodels)
* **Machine Learning:** Scikit-learn
* **Explainability:** SHAP
* **LLM Integration:** Ollama

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-project-folder>
```

---

### 2. Create virtual environment

```bash
python -m venv env
env\Scripts\activate   # Windows
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. ⚠️ Important: Fix for Prophet Compatibility

Ensure NumPy version is compatible:

```bash
pip install numpy==1.26.4
```

---

### 5. Install and run Ollama

Download from: https://ollama.com

Pull model:

```bash
ollama pull llama3.2:3b
```

Run model:

```bash
ollama run llama3.2:3b
```

---

### 6. Run the app

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
├── app.py
├── requirements.txt
├── G20_Historical_Data_2010_2024_v2.xlsx
└── README.md
```

---

## 🧠 How It Works

### 🔹 AI Employability Index

Computed using:

* GDP per capita (20%)
* HDI (20%)
* Internet penetration (15%)
* Patents (15%)
* Startups (10%)
* Employment rate (10%)
* Automation risk (-10%)

(All values normalized)

---

### 🔹 Forecasting Models

* **ARIMA:** Statistical time-series model
* **Prophet:** Trend + seasonality based forecasting

---

### 🔹 AI Analyst

* Uses local LLM via Ollama
* Injects real-time dataset context
* Answers:

  * Trend analysis
  * Country comparisons
  * Policy suggestions

---

## ⚠️ Common Issues & Fixes

### ❌ Prophet / NumPy Error

```
AttributeError: module 'numpy' has no attribute 'float_'
```

✔ Fix:

```bash
pip install numpy==1.26.4
```

---

### ❌ Ollama model not detected

✔ Ensure:

```bash
ollama run llama3.2:3b
```

---

### ❌ Matplotlib / PIL permission error (Windows)

✔ Fix:

* Close all Python processes
* Restart system
* Reinstall dependencies

---

## 📌 Future Improvements

* Automated insight generation from charts
* Deployment (Streamlit Cloud / Docker)
* Multi-model LLM support
* Policy simulation module

---

## 👨‍💻 Author

**Vansh Grover**
B.Tech (AI/ML) | VIPS-TC

---

## ⭐ Acknowledgment

This project combines traditional statistical methods with modern AI systems, reflecting a hybrid approach to solving real-world economic forecasting problems.

---

## 📢 Note

This is a **locally powered AI system** — no external APIs required.
All AI inference runs on your machine via Ollama.

---
