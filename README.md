# 🥇 Athlete Medal Prediction App (BMI + IQ Based)

A Streamlit-based web app that predicts whether an athlete will win a medal using BMI, IQ, age, and Olympic performance history. The model auto-calculates BMI and IQ using domain-specific logic, and enhances prediction using historical performance trends.

---

## 🚀 Live Demo

> 🔗 *(Optional: Add your Streamlit Cloud / HuggingFace link here once deployed)*

---

## 📌 Features

- 🧠 **Custom IQ Prediction** – Combines age, BMI category, medal history, and academy ranking
- ⚖️ **BMI Calculation & Categorization** – Based on height/weight
- 🥇 **Medal Prediction (Yes/No)** – Based on ML model trained on Olympic data
- 📊 **Olympic Data Visuals** – Power BI dashboard or additional visual insights (optional)
- ⚙️ **Fully Interactive** – User can enter new data or select athletes

---

## 🧰 Tech Stack

- 🐍 Python
- 📊 Pandas, NumPy, scikit-learn
- 🌐 Streamlit
- 📈 Power BI (for dashboards)
- 📁 Excel (.xlsx) for input data

---

## 📁 Project Structure

```bash
Athlete-Medal-Prediction/
│
├── data/
│   ├── athlete.xlsx
│   ├── IQ.xlsx
│   ├── performances.xlsx
│   └── olympic_games.xlsx
│
├── model.py         # Model logic, IQ/BMI functions
├── app.py           # Streamlit app interface
├── requirements.txt # Python dependencies
├── dashboard-screenshots/
└── README.md
