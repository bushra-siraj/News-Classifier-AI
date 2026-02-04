# 📰 BBC News AI Classifier & Insights Dashboard

## 🚀 Project Overview
This is a **Full-Stack Machine Learning** application that classifies news articles into five categories: **Business, Entertainment, Politics, Sport, and Tech**. The project combines a robust Natural Language Processing (NLP) backend with an interactive Streamlit frontend.

---

## 🧠 Machine Learning Workflow
* **Data Cleaning:** Handled missing values using `.dropna()` to ensure data integrity.
* **Vectorization:** Implemented **TF-IDF** (Term Frequency-Inverse Document Frequency) to convert raw text into numerical features.
* **Modeling:** Utilized **Logistic Regression** for classification.
* **Deployment:** Built a web interface using **Streamlit** for real-time user predictions.

---

## 📊 Key Features
* **AI Predictor:** Paste any news snippet to get an instant category prediction.
* **Data Insights:** Visualizations of category distributions and article word counts.
* **Word Clouds:** Dynamic vocabulary exploration for each news category.
* **Interactive UI:** Smooth UX with Lottie animations and multi-tab navigation.

---

## 🛠️ Tech Stack
| Tool | Usage |
| :--- | :--- |
| **Python** | Core Logic |
| **Scikit-Learn** | Machine Learning & NLP |
| **Pandas** | Data Manipulation |
| **Streamlit** | Web Interface |
| **Matplotlib** | Data Visualization |

---

## 📂 Project Structure

```text
├── app.py                # Main Streamlit Application
├── model.pkl             # Saved Logistic Regression Model
├── tfidf.pkl             # Saved TF-IDF Vectorizer
├── label_encoder.pkl     # Saved Label Encoder
├── bbc-text.csv          # Dataset
├── requirements.txt      # Dependencies
└── *.png                 # Visualizations
---

## 👤 Author
**Bushra Siraj**
* **Data Science Student** (Exam 2026)
* [GitHub Profile](https://github.com/bushra-siraj)
* [LinkedIn Profile](https://www.linkedin.com/in/bushrasiraj/)
