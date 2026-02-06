import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

# --- Page Config ---
st.set_page_config(
    page_title="NLP Actionable Message Detector",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Better UI ---
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .actionable-box {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        color: #c62828;
    }
    .non-actionable-box {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        color: #2e7d32;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        padding: 12px;
        border-radius: 8px;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background-color: #1565C0;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_data
def load_data():
    """Load the dataset"""
    try:
        return pd.read_csv('social_media_actionable_dataset.csv')
    except FileNotFoundError:
        st.error("❌ Dataset file 'social_media_actionable_dataset.csv' not found!")
        st.info("Please ensure the CSV file is in the same directory as this script.")
        st.stop()

def clean_text(text):
    """Clean and preprocess text"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

# --- Load Data ---
df = load_data()

# --- Sidebar ---
st.sidebar.markdown("## 🎯 Navigation Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Choose a page:",
    ["🏠 Project Overview & EDA", "📊 Model Evaluation", "🔮 Live Prediction"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 About This Project")
st.sidebar.info(
    """
    This application uses **Natural Language Processing (NLP)** 
    to automatically detect actionable customer service messages 
    from social media platforms.
    
    **Models Used:**
    - Random Forest Classifier
    - LSTM Neural Network
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Dataset Info")
st.sidebar.metric("Total Messages", len(df))
st.sidebar.metric("Actionable Messages", df['actionable'].sum())
st.sidebar.metric("Non-Actionable", len(df) - df['actionable'].sum())

# --- MODEL TRAINING ENGINE ---
@st.cache_resource
def train_models(df):
    """Train both Random Forest and LSTM models"""
    # Clean text
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], 
        df['actionable'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['actionable']
    )
    
    # === RANDOM FOREST MODEL ===
    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    rf = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        max_depth=20,
        min_samples_split=5
    )
    rf.fit(X_train_tfidf, y_train)
    
    # === LSTM MODEL ===
    max_words, max_len = 5000, 100
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = pad_sequences(
        tokenizer.texts_to_sequences(X_train), 
        maxlen=max_len,
        padding='post'
    )
    X_test_seq = pad_sequences(
        tokenizer.texts_to_sequences(X_test), 
        maxlen=max_len,
        padding='post'
    )
    
    lstm = Sequential([
        Embedding(max_words, 128, input_length=max_len),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    lstm.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    
    lstm.fit(
        X_train_seq, 
        y_train, 
        epochs=5, 
        batch_size=32, 
        validation_split=0.1,
        verbose=0
    )
    
    return rf, tfidf, lstm, tokenizer, X_test, y_test, X_test_tfidf, X_test_seq

# ====================================
# PAGE 1: PROJECT OVERVIEW & EDA
# ====================================
if page == "🏠 Project Overview & EDA":
    st.markdown('<h1 class="main-header">📊 Data Insights & Exploratory Analysis</h1>', unsafe_allow_html=True)
    
    # Overview Section
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h3>Total Messages</h3>
                <h2>{}</h2>
            </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        actionable_count = df['actionable'].sum()
        st.markdown("""
            <div class="metric-card">
                <h3>Actionable</h3>
                <h2>{}</h2>
            </div>
        """.format(actionable_count), unsafe_allow_html=True)
    
    with col3:
        non_actionable_count = len(df) - df['actionable'].sum()
        st.markdown("""
            <div class="metric-card">
                <h3>Non-Actionable</h3>
                <h2>{}</h2>
            </div>
        """.format(non_actionable_count), unsafe_allow_html=True)
    
    with col4:
        platforms = df['platform'].nunique() if 'platform' in df.columns else 0
        st.markdown("""
            <div class="metric-card">
                <h3>Platforms</h3>
                <h2>{}</h2>
            </div>
        """.format(platforms), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dataset Preview and Visualization
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<p class="sub-header">📋 Dataset Preview</p>', unsafe_allow_html=True)
        st.dataframe(
            df.head(10),
            use_container_width=True,
            height=400
        )
        
        # Basic Statistics
        st.markdown('<p class="sub-header">📈 Basic Statistics</p>', unsafe_allow_html=True)
        actionable_pct = (df['actionable'].sum() / len(df)) * 100
        st.write(f"**Actionable Rate:** {actionable_pct:.2f}%")
        st.write(f"**Non-Actionable Rate:** {100-actionable_pct:.2f}%")
        
    with col2:
        st.markdown('<p class="sub-header">📊 Actionable vs Non-Actionable Distribution</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#4caf50', '#f44336']
        counts = df['actionable'].value_counts()
        ax.pie(
            counts.values, 
            labels=['Non-Actionable', 'Actionable'],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 12, 'weight': 'bold'}
        )
        ax.set_title('Message Classification Distribution', fontsize=14, weight='bold')
        st.pyplot(fig)
        plt.close()
    
    # Platform Analysis
    if 'platform' in df.columns:
        st.markdown("---")
        st.markdown('<p class="sub-header">🌐 Platform Activity Analysis</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.countplot(
                data=df, 
                x='platform', 
                hue='actionable', 
                palette=['#4caf50', '#f44336'],
                ax=ax
            )
            ax.set_title('Messages by Platform and Type', fontsize=14, weight='bold')
            ax.set_xlabel('Platform', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.legend(title='Type', labels=['Non-Actionable', 'Actionable'])
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 5))
            platform_stats = df.groupby('platform')['actionable'].mean() * 100
            platform_stats.plot(kind='bar', color='#1E88E5', ax=ax)
            ax.set_title('Actionable Rate by Platform (%)', fontsize=14, weight='bold')
            ax.set_xlabel('Platform', fontsize=12)
            ax.set_ylabel('Actionable Rate (%)', fontsize=12)
            plt.xticks(rotation=45)
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            plt.close()
    
    # Sample Messages
    st.markdown("---")
    st.markdown('<p class="sub-header">💬 Sample Messages</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔴 Actionable Messages:**")
        actionable_samples = df[df['actionable'] == 1].sample(min(5, df['actionable'].sum()))
        for idx, row in actionable_samples.iterrows():
            st.markdown(f"- {row['text'][:100]}...")
    
    with col2:
        st.markdown("**🟢 Non-Actionable Messages:**")
        non_actionable_samples = df[df['actionable'] == 0].sample(min(5, len(df) - df['actionable'].sum()))
        for idx, row in non_actionable_samples.iterrows():
            st.markdown(f"- {row['text'][:100]}...")

# ====================================
# PAGE 2: MODEL EVALUATION
# ====================================
elif page == "📊 Model Evaluation":
    st.markdown('<h1 class="main-header">⚙️ Model Performance & Metrics</h1>', unsafe_allow_html=True)
    
    with st.spinner('🔄 Training models... This may take a moment...'):
        rf, tfidf, lstm, tokenizer, X_test, y_test, X_test_tfidf, X_test_seq = train_models(df)
    
    st.success("✅ Models trained successfully!")
    
    st.markdown("---")
    
    # Model Accuracies
    rf_acc = rf.score(X_test_tfidf, y_test)
    y_pred_lstm = (lstm.predict(X_test_seq, verbose=0) > 0.5).astype(int).flatten()
    lstm_acc = accuracy_score(y_test, y_pred_lstm)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h3>🌲 Random Forest</h3>
                <h1>{:.2%}</h1>
                <p>Accuracy</p>
            </div>
        """.format(rf_acc), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <h3>🧠 LSTM Network</h3>
                <h1>{:.2%}</h1>
                <p>Accuracy</p>
            </div>
        """.format(lstm_acc), unsafe_allow_html=True)
    
    with col3:
        ensemble_acc = (rf_acc + lstm_acc) / 2
        st.markdown("""
            <div class="metric-card">
                <h3>🎯 Ensemble</h3>
                <h1>{:.2%}</h1>
                <p>Average Accuracy</p>
            </div>
        """.format(ensemble_acc), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Confusion Matrices
    st.markdown('<p class="sub-header">📊 Confusion Matrices</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Random Forest Confusion Matrix**")
        y_pred_rf = rf.predict(X_test_tfidf)
        cm_rf = confusion_matrix(y_test, y_pred_rf)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm_rf, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Non-Actionable', 'Actionable'],
            yticklabels=['Non-Actionable', 'Actionable'],
            ax=ax
        )
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_title('Random Forest', fontsize=14, weight='bold')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("**LSTM Confusion Matrix**")
        cm_lstm = confusion_matrix(y_test, y_pred_lstm)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm_lstm, 
            annot=True, 
            fmt='d', 
            cmap='Greens',
            xticklabels=['Non-Actionable', 'Actionable'],
            yticklabels=['Non-Actionable', 'Actionable'],
            ax=ax
        )
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_title('LSTM Network', fontsize=14, weight='bold')
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Classification Reports
    st.markdown('<p class="sub-header">📋 Detailed Classification Reports</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Random Forest Report**")
        report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
        st.dataframe(pd.DataFrame(report_rf).transpose(), use_container_width=True)
    
    with col2:
        st.markdown("**LSTM Report**")
        report_lstm = classification_report(y_test, y_pred_lstm, output_dict=True)
        st.dataframe(pd.DataFrame(report_lstm).transpose(), use_container_width=True)
    
    st.markdown("---")
    
    # Feature Importance
    st.markdown('<p class="sub-header">🔍 Top 15 Important Words (Random Forest)</p>', unsafe_allow_html=True)
    
    features = tfidf.get_feature_names_out()
    importances = rf.feature_importances_
    indices = np.argsort(importances)[-15:]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(15), importances[indices], color='skyblue', edgecolor='navy')
    ax.set_yticks(range(15))
    ax.set_yticklabels([features[i] for i in indices], fontsize=11)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('Most Important Words for Classification', fontsize=14, weight='bold')
    ax.grid(axis='x', alpha=0.3)
    st.pyplot(fig)
    plt.close()

# ====================================
# PAGE 3: LIVE PREDICTION
# ====================================
elif page == "🔮 Live Prediction":
    st.markdown('<h1 class="main-header">🔮 Live Message Classification</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">Enter a customer message below to analyze whether it requires action from customer service.</div>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner('🔄 Loading models...'):
        rf, tfidf, lstm, tokenizer, _, _, _, _ = train_models(df)
    
    st.markdown("---")
    
    # Input Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_area(
            "📝 Enter Message to Analyze:",
            value="I haven't received my refund yet, please check!",
            height=150,
            help="Type or paste a customer message here"
        )
    
    with col2:
        st.markdown("### 💡 Try These Examples:")
        
        example_messages = [
            "Great product, love it!",
            "Where is my order? It's been 2 weeks!",
            "Thanks for the awesome service!",
            "I need help with my account login",
            "Amazing quality, highly recommend"
        ]
        
        selected_example = st.selectbox(
            "Quick examples:",
            ["Select an example..."] + example_messages
        )
        
        if selected_example != "Select an example...":
            user_input = selected_example
    
    st.markdown("---")
    
    # Predict Button
    if st.button("🔍 Analyze Message", use_container_width=True):
        if not user_input or user_input.strip() == "":
            st.warning("⚠️ Please enter a message to analyze.")
        else:
            with st.spinner('🔄 Analyzing message...'):
                cleaned = clean_text(user_input)
                
                if not cleaned or cleaned.strip() == "":
                    st.warning("⚠️ The message doesn't contain valid text after cleaning. Please enter a meaningful message.")
                else:
                    # Random Forest Prediction
                    rf_input = tfidf.transform([cleaned])
                    rf_prob = rf.predict_proba(rf_input)[0][1]
                    rf_prediction = rf.predict(rf_input)[0]
                    
                    # LSTM Prediction
                    input_seq = pad_sequences(
                        tokenizer.texts_to_sequences([cleaned]), 
                        maxlen=100,
                        padding='post'
                    )
                    lstm_prob = float(lstm.predict(input_seq, verbose=0)[0][0])
                    lstm_prediction = 1 if lstm_prob > 0.5 else 0
                    
                    # Display Results
                    st.markdown("---")
                    st.markdown("## 📊 Analysis Results")
                    
                    # Individual Model Scores
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 🌲 Random Forest")
                        st.progress(rf_prob)
                        st.metric(
                            "Confidence Score", 
                            f"{rf_prob:.1%}",
                            delta="Actionable" if rf_prediction == 1 else "Non-Actionable"
                        )
                    
                    with col2:
                        st.markdown("### 🧠 LSTM Network")
                        st.progress(lstm_prob)
                        st.metric(
                            "Confidence Score", 
                            f"{lstm_prob:.1%}",
                            delta="Actionable" if lstm_prediction == 1 else "Non-Actionable"
                        )
                    
                    st.markdown("---")
                    
                    # Ensemble Prediction
                    final_prob = (rf_prob + lstm_prob) / 2
                    final_prediction = 1 if final_prob > 0.5 else 0
                    
                    st.markdown("### 🎯 Final Ensemble Prediction")
                    
                    # Progress bar for ensemble
                    st.progress(final_prob)
                    
                    if final_prediction == 1:
                        st.markdown(f"""
                            <div class="prediction-box actionable-box">
                                🚨 <strong>ACTIONABLE MESSAGE</strong><br>
                                Confidence: {final_prob:.1%}<br>
                                <small>This message requires customer service attention</small>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("#### 📌 Recommended Actions:")
                        st.markdown("""
                        - ✅ Route to customer service team
                        - ✅ Prioritize based on urgency
                        - ✅ Assign to appropriate department
                        - ✅ Track response time
                        """)
                    else:
                        st.markdown(f"""
                            <div class="prediction-box non-actionable-box">
                                ✅ <strong>NON-ACTIONABLE MESSAGE</strong><br>
                                Confidence: {(1-final_prob):.1%}<br>
                                <small>General feedback or positive comment</small>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("#### 📌 Message Classification:")
                        st.markdown("""
                        - ℹ️ General feedback or comment
                        - ℹ️ No immediate action required
                        - ℹ️ Can be logged for analytics
                        - ℹ️ May be used for sentiment analysis
                        """)
                    
                    st.markdown("---")
                    
                    # Message Analysis Details
                    with st.expander("🔍 View Message Analysis Details"):
                        st.markdown("**Original Message:**")
                        st.code(user_input, language=None)
                        
                        st.markdown("**Cleaned Message:**")
                        st.code(cleaned, language=None)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Original Length", f"{len(user_input)} chars")
                        col2.metric("Cleaned Length", f"{len(cleaned)} chars")
                        col3.metric("Word Count", f"{len(cleaned.split())} words")
                        
                        st.markdown("**Model Agreement:**")
                        if rf_prediction == lstm_prediction:
                            st.success("✅ Both models agree on classification")
                        else:
                            st.warning("⚠️ Models disagree - using ensemble average")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>NLP Actionable Message Detector</strong></p>
        <p>Powered by Random Forest & LSTM Neural Networks</p>
        <p>Built with Streamlit 🚀</p>
    </div>
""", unsafe_allow_html=True)