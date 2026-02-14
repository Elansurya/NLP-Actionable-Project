# 🧠 Finding Message is Actionable or Not (Required Response or Not)

## 🚀 Project Overview

This project builds an NLP-based classification system that automatically identifies whether a customer message requires a response (Actionable) or not (Non-Actionable).

The solution combines statistical NLP techniques and Deep Learning approaches to create an intelligent message triage system for customer support automation.

Domain: Customer Intelligence & Customer Experience

---

## 🎯 Problem Statement

Organizations receive thousands of customer messages daily via email, chat, or social media.  
However, not all messages require a response — some are acknowledgments, greetings, or general comments.

The objective is to build a robust NLP classification model that:

- Detects Actionable (1) messages requiring response  
- Identifies Non-Actionable (0) messages  
- Reduces response backlog  
- Improves customer support efficiency  

(As described in the project documentation :contentReference[oaicite:1]{index=1})

---

## 📊 Business Use Cases

- 🎫 Customer Support Ticketing – Auto-flag messages requiring agent response  
- 📱 Social Media Monitoring – Detect actionable tweets/posts  
- 📧 Email Sorting – Separate informational emails from urgent ones  
- 🧾 CRM Systems – Prioritize customer queries  
- 🤖 Helpdesk Automation – Route actionable messages to human agents  

---

## 📊 Dataset Overview

Format: CSV  

Variables:
- message – Text content of customer message  
- label – 1 (Actionable), 0 (Non-Actionable)

Data includes:
- Emails
- Chat messages
- Tweets
- Support queries

Example:
- "My ticket hasn’t been confirmed yet, please help." → Actionable  
- "Thanks for your quick support!" → Non-Actionable  

(Example data described in project pages :contentReference[oaicite:2]{index=2})

---

## 🧹 Data Preprocessing

Implemented complete NLP preprocessing pipeline:

- Lowercase conversion  
- Stopword removal  
- Punctuation & special character cleaning  
- Lemmatization (SpaCy / NLTK)  
- Tokenization  

Ensured text cleaning before vectorization (as per project guidelines :contentReference[oaicite:3]{index=3})

---

## 🧠 Feature Extraction Techniques

### Statistical NLP
- Bag of Words (CountVectorizer)
- TF-IDF Vectorizer

### Semantic Representation
- Word2Vec / GloVe Embeddings

---

## 🔍 Feature Engineering

- Used RandomForestClassifier for feature importance ranking  
- Selected top contributing features  
- Reduced noise & improved classification performance  

---

## 🤖 Model Building

### Machine Learning Models
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

### Deep Learning Model
- LSTM using word embeddings

Compared ML vs DL approaches for performance evaluation.

---

## 📈 Model Evaluation Metrics

- Accuracy
- Precision (Actionable class)
- Recall (Optimized for actionable detection)
- F1-Score
- Confusion Matrix
- ROC-AUC (optional)

Focus: High Recall for actionable messages to avoid missing important queries.

---

## 🖥️ Deployment (Optional)

Developed Streamlit application where users can:

- Enter text message
- Receive prediction (Actionable / Non-Actionable)
- View confidence score

Acts as an intelligent message triage assistant.

---

## 🏗️ Workflow Architecture

Customer Messages  
↓  
Text Cleaning & Preprocessing  
↓  
Feature Extraction (TF-IDF / Embeddings)  
↓  
ML & DL Model Training  
↓  
Model Evaluation & Comparison  
↓  
Streamlit Deployment  

---

## ⚙️ Tech Stack

Python  
Scikit-learn  
TensorFlow / Keras  
NLTK / SpaCy  
TF-IDF  
CountVectorizer  
Word2Vec  
LSTM  
RandomForest  
XGBoost  
Streamlit  

---

## 📌 Key Learnings

- Difference between statistical NLP and deep learning NLP  
- Importance of recall in support automation  
- Feature importance interpretation  
- End-to-end ML pipeline design  
- NLP model deployment  

---

## 🔮 Future Enhancements

- Transformer-based models (BERT)
- Multi-language support
- Real-time API integration
- Automated ticket routing system
- Class imbalance optimization techniques

---

## 👨‍💻 Author
Elansurya K  
Data Scientist | Machine Learning | NLP | SQL
