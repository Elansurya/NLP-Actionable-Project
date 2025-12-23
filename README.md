 🛡️ ActionIntel: AI-Driven Message Prioritization System  
**Bridging the gap between massive social media noise and actionable customer intelligence**

 🌟 Executive Summary
In today’s fast-paced digital world, customer support teams are overwhelmed by thousands of social media messages every day. Manually identifying which messages require urgent attention is inefficient and error-prone.

**ActionIntel** is an end-to-end **NLP-based intelligent system** that automatically classifies incoming messages as **Actionable** (requires response) or **Non-Actionable** (spam, greetings, casual chatter).  
This solution helps organizations **prioritize critical issues faster**, reducing response time by up to **60%** and improving customer satisfaction.

🛠️ Tech Stack
- **Language:** Python 3.10+
- **NLP & Text Preprocessing:** Regex, NLTK (Tokenization, Lemmatization)
- **Machine Learning:** Random Forest Classifier (high interpretability)
- **Deep Learning:** LSTM Neural Network (context & sequence understanding)
- **Text Vectorization:** TF-IDF, Keras Word Embeddings
- **Web Interface:** Streamlit (Interactive Dashboard)

 📊 Project Workflow
 1️⃣ Exploratory Data Analysis (EDA)
- Analysis of customer sentiment
- Identification of platform-specific message patterns
- Actionable vs non-actionable message distribution

 2️⃣ Advanced Text Cleaning
- Removal of URLs, mentions, emojis, and stopwords
- Preservation of intent-critical keywords
- Normalization and lemmatization for better model performance

 3️⃣ Hybrid Modeling Approach
**🔹 Random Forest (ML Model)**
- Used for fast and interpretable classification
- Feature importance analysis to identify **“power words”** like *crash*, *delay*, *refund*, *help*

**🔹 LSTM (Deep Learning Model)**
- Captures long-term dependencies in messages
- Better understanding of context and sentence structure
- Improved classification of complex customer complaints

 4️⃣ Live Deployment
- Streamlit-based web application
- Real-time message classification
- Instant actionable vs non-actionable prediction


 📈 Performance Highlights

| Metric | Random Forest (ML) | LSTM (Deep Learning) |
|------|-------------------|---------------------|
| Accuracy | ~90% | **92%+** |
| Primary Strength | Keyword & feature-based detection | Contextual sequence understanding |
| Precision (Actionable) | High | **Superior** |



🔹 Clone the Repository
```bash
https://github.com/Elansurya/NLP-Actionable-Project.git
