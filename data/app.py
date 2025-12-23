import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# --- Page Config ---
st.set_page_config(page_title="NLP Actionable Detector", layout="wide", page_icon="🛡️")

# --- Helper Functions ---
@st.cache_data
def load_data():
    # Ensure this file is in your project folder
    return pd.read_csv('social_media_actionable_dataset.csv')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

# --- Sidebar ---
st.sidebar.title("Project Dashboard")
st.sidebar.info("This project uses NLP to automate customer service by identifying actionable messages.")
page = st.sidebar.radio("Navigate", ["Project Overview & EDA", "Model Evaluation", "Live Prediction"])

df = load_data()

# --- PAGE 1: EDA ---
if page == "Project Overview & EDA":
    st.title("📊 Data Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10))
        
    with col2:
        st.subheader("Actionable vs Non-Actionable")
        fig, ax = plt.subplots()
        sns.countplot(x='actionable', data=df, palette='viridis', ax=ax)
        ax.set_xticklabels(['Non-Actionable', 'Actionable'])
        st.pyplot(fig)

    st.subheader("Platform Activity")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.countplot(data=df, x='platform', hue='actionable', palette='magma', ax=ax2)
    st.pyplot(fig2)

# --- MODEL TRAINING ENGINE ---
@st.cache_resource
def train_models(df):
    df['clean_text'] = df['text'].apply(clean_text)
    X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['actionable'], test_size=0.2, random_state=42)
    
    # Random Forest
    tfidf = TfidfVectorizer(max_features=2000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_tfidf, y_train)
    
    # LSTM
    max_words, max_len = 5000, 100
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len)
    
    lstm = Sequential([
        Embedding(max_words, 128, input_length=max_len),
        LSTM(64, dropout=0.2),
        Dense(1, activation='sigmoid')
    ])
    lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    lstm.fit(X_train_seq, y_train, epochs=3, batch_size=64, verbose=0)
    
    return rf, tfidf, lstm, tokenizer, X_test, y_test

# --- PAGE 2: METRICS ---
if page == "Model Evaluation":
    st.title("⚙️ Technical Performance")
    rf, tfidf, lstm, tokenizer, X_test, y_test = train_models(df)
    
    X_test_tfidf = tfidf.transform(X_test)
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=100)
    
    # RF Metrics
    rf_acc = rf.score(X_test_tfidf, y_test)
    
    # LSTM Metrics
    y_pred_lstm = (lstm.predict(X_test_seq) > 0.5).astype(int).flatten()
    lstm_acc = accuracy_score(y_test, y_pred_lstm)
    
    m1, m2 = st.columns(2)
    m1.metric("Random Forest Accuracy", f"{rf_acc:.2%}")
    m2.metric("LSTM Accuracy", f"{lstm_acc:.2%}")

    st.subheader("Word Importance (Random Forest)")
    features = tfidf.get_feature_names_out()
    importances = rf.feature_importances_
    indices = np.argsort(importances)[-10:]
    fig3, ax3 = plt.subplots()
    ax3.barh(range(10), importances[indices], color='skyblue')
    ax3.set_yticks(range(10))
    ax3.set_yticklabels([features[i] for i in indices])
    st.pyplot(fig3)

# --- PAGE 3: PREDICTION ---
if page == "Live Prediction":
    st.title("🔮 Model Inference")
    user_input = st.text_area("Analyze a Message:", "I haven't received my refund yet, please check!")
    
    if st.button("Predict Intent"):
        rf, tfidf, lstm, tokenizer, _, _ = train_models(df)
        cleaned = clean_text(user_input)
        
        # RF
        rf_prob = rf.predict_proba(tfidf.transform([cleaned]))[0][1]
        
        # LSTM
        input_seq = pad_sequences(tokenizer.texts_to_sequences([cleaned]), maxlen=100)
        lstm_prob = float(lstm.predict(input_seq)[0][0])
        
        c1, c2 = st.columns(2)
        c1.write(f"**Random Forest Score:** {rf_prob:.2%}")
        c2.write(f"**LSTM Score:** {lstm_prob:.2%}")
        
        final_prob = (rf_prob + lstm_prob) / 2
        if final_prob > 0.5:
            st.error(f"🚨 **ACTIONABLE** (Score: {final_prob:.2%})")
        else:
            st.success(f"✅ **NON-ACTIONABLE** (Score: {final_prob:.2%})")